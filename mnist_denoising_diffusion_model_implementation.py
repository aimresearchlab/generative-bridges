import os

import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from utils import load_mnist_dataset, check_torch_cuda
import argparse
from torch.utils.tensorboard import SummaryWriter


def unflatten_data(vector, shape=(28, 28)):
    """
    Function to unflatten an image
    :param vector: Flattened image tensor
    :param shape: Target shape
    :return: Unflattened image tensor
    """
    return vector.view(-1, 1, *shape)


def sample_batch(trainX, batch_size, device):
    indices = torch.randperm(len(trainX))[:batch_size]
    data = torch.stack([trainX[i] for i in indices]).to(device)
    data = unflatten_data(data)  # Reshape to (batch_size, 1, 28, 28)
    return torch.nn.functional.interpolate(data, size=(32, 32))


class DiffusionModel:

    def __init__(self, T: int, model: nn.Module, device: str, trainX: List):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)  # forward process variance schedule as per paper 4. Experiments section
        self.alpha = 1. - self.beta  # αt := 1 − βt
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # ᾱt definition as per Page 2 (before eq 4)
        self.trainX = trainX

    def train_w_fwd_process(self, batch_size, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        The algorithm samples a random real image x₀ from the distribution q(x₀), then sample time step uniformly between 1 to T,
        which determines the noise level and generates a random sample of noise from a Gaussian distribution.
        Using that noise, the image is corrupted to xₜ on the basis of x₀ and a known βₜ, and the neural network
        uses that in its training to predict the noise.
        """
        x0 = sample_batch(self.trainX, batch_size, self.device)
        t = torch.randint(1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long) # sample batch_size t's from uniform [1,...,T]
        eps = torch.randn_like(x0)  # return a tensor with the same size as input that is filled with random numbers from N(0,1).

        # Take one gradient descent step
        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x0_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps  # Algorithm 1 Step 5 input for of epsilon_theta (func approximator)

        # The input to function_approximator must be 4D: (batch_size, channels, height, width)
        eps_predicted = self.function_approximator(x0_t, t - 1)
        loss = nn.functional.mse_loss(eps, eps_predicted)  # ((eps - eps_predicted)**2).mean(), the derived loss from -logLik and KL div, with the scaling factor ommitted
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()  # as it is not part of training
    def sample_w_reverse_process(self, n_samples=1, image_channels=1, img_size=(32, 32),
                                 use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        For sampling from the reverse process and get a new image (Sample from p_θ(xₜ₋₁|xₜ) to get xₜ₋₁ according to Algo 2)
        We start with a pure standard Gaussian noise (t=T), and we gradually denoise it using the prediction of our model.
        In the end (t=0), we get x₀, which should be a clean image from the real distribution.
        """

        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                        device=self.device)  # pure Gaussian noise
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)  # sample z as per Algo 2
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Step 4 x_t-1
            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * self.function_approximator(x, t - 1))  # Step 4 (x_t-1) mean
            sigma = torch.sqrt(beta_t)  # Step 4 (x_t-1) variance
            x = mean + sigma * z
        return x


def save_samples(diffusion_model, nb_images, n_batch, img_dir):
    samples = diffusion_model.sample_w_reverse_process(n_samples=nb_images, use_tqdm=False)
    plt.figure(figsize=(17, 17))
    for i in range(nb_images):
        plt.subplot(9, 9, 1 + i)
        plt.axis('off')
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap='gray')
    plt.savefig(os.path.join(img_dir, f'samples_epoch_{n_batch}.png'))
    plt.close()


if __name__ == "__main__":

    check_torch_cuda()  # check CUDA

    parser = argparse.ArgumentParser(description="Set folders input and output")
    parser.add_argument('--data_folder', default='./data', type=str, required=False, help='Path to the data folder')
    parser.add_argument('--imgs_folder', default='./imgs/mnist/denoise', type=str, required=False, help='Path to the imgs folder')
    parser.add_argument('--models_folder', default='./aim-bridges-models/denoise/', type=str, required=False, help='Path to the models folder')
    parser.add_argument('--download_data', action='store_true', help='Download the data')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_images', action='store_false', help='Save images')

    args = parser.parse_args()

    # Create the "data" folder if it does not exist
    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)
    if not os.path.exists(args.imgs_folder):
        os.makedirs(args.imgs_folder)
    if not os.path.exists(args.models_folder):
        os.makedirs(args.models_folder)

    trainY, testY = load_mnist_dataset(root=args.data_folder, download=args.download_data)

    device = 'cuda'
    batch_size = 64
    t_steps = 1000
    deterministic = False
    checkpoint_steps = int((1000 / batch_size) * 64)
    num_batches = 40_000

    model = UNet()
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    diffusion_model = DiffusionModel(t_steps, model, device, trainY)

    # Training
    det_name = 'det' if deterministic else 'stoch'
    model_name = f'batch_{batch_size}_t_steps_{t_steps}_{det_name}'

    print(f'batch_size: {batch_size}; t_steps: {t_steps}; deterministic: {deterministic}')
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'{args.models_folder}/runs/{model_name}')
    for n_batch in tqdm(range(num_batches)):
        loss = diffusion_model.train_w_fwd_process(batch_size, optimizer)

        # Save sampled images every 2000 epochs
        if args.save_images and (n_batch + 1) % checkpoint_steps == 0:
            save_samples(diffusion_model, nb_images=81, n_batch=n_batch + 1, img_dir=args.imgs_folder)
        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', loss, n_batch)

        if n_batch % checkpoint_steps == 0:
            checkpoint_path = os.path.join(f'{args.models_folder}/checkpoints', f"{model_name}_checkpoint_n_batch_{n_batch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            if args.verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final model
    final_model_path = os.path.join(f'{args.models_folder}/checkpoints', f"{model_name}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()  # Close the writer when done

    # Save final results
    if args.save_images:
        save_samples(diffusion_model, nb_images=81, n_batch='final', img_dir=args.imgs_folder)
