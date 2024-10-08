import argparse
import os
import numpy as np
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet import UNet
from unet_openai import create_openAI_Unet
from utils import load_mnist_dataset, load_cifar10_dataset, check_torch_cuda, sample_batch


class SimpleBrownianBridgeDiffusion:

    def __init__(self, T: float, model: nn.Module, device: str, trainY: List, deterministic: bool = False, interpolate_size = None):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device
        self.trainY = trainY

        self._EPS = 0.0000001
        self._deterministic = deterministic
        self.interpolate_size = interpolate_size

    def train(self, batch_size, optimizer):
        """
        Corollary 2.7 implementation
        """

        # sample target images y
        y = sample_batch(self.trainY, batch_size, self.device, interpolate_size=self.interpolate_size)
        # sample random times from [0,T)
        t = torch.rand(batch_size, device=self.device, dtype=torch.float32) * self.T
        # Unsqueeze t to align dimensions
        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, 1, 1, 1)
        # sample x as independent gaussians 
        x = torch.randn(y.shape, device=self.device)
        # compute E_t
        E = x + (y - x) * t / self.T
        # compute V_t
        V = t * (self.T - t) / self.T
        # compute xi_t sampling 
        xi = E + torch.randn(y.shape, device=self.device) * torch.sqrt(V)
        # Take one gradient descent step
        # The input to function_approximator must be 4D: (batch_size, channels, height, width)
        # t is a list of times [batch_size] (hence squeeze)
        img_predicted = self.function_approximator(xi, t.squeeze())
        mse = nn.functional.mse_loss(y, img_predicted)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        return mse.item()

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), n_steps=100,
                 use_tqdm=True):
        """
        Corollary 2.9 implementation
        """
        # create n_steps t
        # messing about with n_steps as small eps gets rounded away
        t_values = torch.linspace(0, self.T, steps=n_steps + 1, device=self.device) * (self.T - self._EPS) / self.T

        # sample x
        delta = (self.T - self._EPS) / n_steps
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                        device=self.device)

        xi_t = x
        #Z_t = x

        progress_bar = tqdm if use_tqdm else lambda x: x
        for idx, t in enumerate(progress_bar(t_values[:-1])):
            t = torch.tensor(t, device=self.device)
            inv_lambda = 1 / (self.T - t)
            # mu_t = x; Sigma_tt = t
            #Z = x + torch.randn(x.shape, device=self.device)*torch.sqrt(t)
            # need a list of repeated t with len same as n_samples
            t_list = [t for _ in range(n_samples)]
            #xi = xi_t + inv_lambda*(self.function_approximator(xi_t, torch.stack(t_list)) - xi_t )*delta + (Z - Z_t)
            xi = xi_t + inv_lambda * (self.function_approximator(xi_t, torch.stack(t_list)) - xi_t) * delta
            if not self._deterministic:
                dZ = torch.randn(x.shape, device=self.device) * torch.sqrt(torch.tensor(delta))
                xi += dZ

            xi_t = xi
            #Z_t = Z
        return xi_t

def save_samples(diffusion_model, nb_images, n_batch, path):
    samples = diffusion_model.sample_w_reverse_process(n_samples=nb_images, use_tqdm=False)
    plt.figure(figsize=(17, 17))
    for i in range(nb_images):
        plt.subplot(9, 9, 1 + i)
        plt.axis('off')
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap='gray')
    plt.savefig(f'{path}/samples_epoch_{n_batch}.png')
    plt.close()

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def p_mean_variance(
    self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
):
    """
    Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
    the initial x, x_0.

    :param model: the model, which takes a signal and a batch of timesteps
                  as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :param clip_denoised: if True, clip the denoised signal into [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample. Applies before
        clip_denoised.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict with the following keys:
             - 'mean': the model mean output.
             - 'variance': the model variance output.
             - 'log_variance': the log of 'variance'.
             - 'pred_xstart': the prediction for x_0.
    """
    if model_kwargs is None:
        model_kwargs = {}

    B, C = x.shape[:2]
    assert t.shape == (B,)
    model_output = model(x, self._scale_timesteps(t), **model_kwargs)

    if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)
        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
    else:
        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

    def process_xstart(x):
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            return x.clamp(-1, 1)
        return x

    if self.model_mean_type == ModelMeanType.PREVIOUS_X:
        pred_xstart = process_xstart(
            self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
        )
        model_mean = model_output
    elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
    else:
        raise NotImplementedError(self.model_mean_type)

    assert (
        model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    )
    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
    }

if __name__ == "__main__":

    check_torch_cuda()  # check CUDA

    parser = argparse.ArgumentParser(description="Set folders input and output")
    parser.add_argument('--data_folder', default='./data', type=str, required=False, help='Path to the data folder')
    parser.add_argument('--imgs_folder', default='./imgs/mnist/bridge', type=str, required=False, help='Path to the imgs folder')
    parser.add_argument('--models_folder', default='./aim-bridges-models/bridge', type=str, required=False, help='Path to the models folder')
    parser.add_argument('--dataset', default='cifar', help='Select dataset, mnist, cifar')
    parser.add_argument('--unet', default='openAI', help='Select UNet: berkan or openAI')
    parser.add_argument('--download_data', action='store_true', help='Download the data')
    parser.add_argument('--verbose', action='store_true', help='Download the data')

    args = parser.parse_args()
    check_point_folder = f'{args.models_folder}/checkpoints'
    # Create the "data" folder if it does not exist
    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)
    if not os.path.exists(args.imgs_folder):
        os.makedirs(args.imgs_folder)
    if not os.path.exists(check_point_folder):
        os.makedirs(check_point_folder)
    if args.dataset == 'mnist':
        trainY, testY = load_mnist_dataset(root=args.data_folder, download=args.download_data)
    elif args.dataset == 'cifar':
         trainY, testY = load_cifar10_dataset(root=args.data_folder, download=args.download_data)
    else:
        raise f"unknown dataset: {args.dataset}"

    # basic config
    device = 'cuda'
    batch_size = 64
    T = 0.1
    learning_rate = 1e-4

    deterministic = False
    checkpoint_steps = (1000 / batch_size) * 64
    num_batches = 40_000

    if args.dataset == 'mnist':
        if args.unet == 'berkan':
            model = UNet()
        elif args.unet == 'openAI':
            model = create_openAI_Unet()
        diffusion_model = SimpleBrownianBridgeDiffusion(T, model, device, trainY, deterministic=deterministic, interpolate_size=32)
    elif args.dataset == 'cifar':
        if args.unet == 'berkan':
            model = UNet(in_ch=3)
        elif args.unet == 'openAI':
            model = create_openAI_Unet()
        diffusion_model = SimpleBrownianBridgeDiffusion(T, model, device, trainY, deterministic=deterministic)
    else:
        raise f"unknown dataset: {args.dataset}"

    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    det_name = 'det' if deterministic else 'stoch'
    model_name = f'batch_{batch_size}_T_{T}_{det_name}'
    print(f'batch_size: {batch_size}; T: {T}; deterministic: {deterministic}')

    # Training
    writer = SummaryWriter(log_dir=f'{args.models_folder}/runs/{model_name}')  # Initialize TensorBoard writer
    for n_batch in tqdm(range(num_batches)):
        loss = diffusion_model.train(batch_size, optimizer)
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        writer.add_scalar('Loss/train', loss, n_batch)  # Log the loss to TensorBoard

        if n_batch % checkpoint_steps == 0:
            checkpoint_path = os.path.join(check_point_folder, f"{model_name}_checkpoint_n_batch_{n_batch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            if args.verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final model
    final_model_path = os.path.join(f'{args.models_folder}/checkpoints', f"{model_name}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()  # Close the writer when done
