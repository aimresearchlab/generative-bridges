import os
import torch
import torch.distributions as dist
import torch.nn as nn
from typing import List
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import load_mnist_dataset, load_cifar10_dataset, view_image, check_torch_cuda
from unet import get_timestep_embedding, UNet

import matplotlib.pyplot as plt
import numpy as np
import argparse


class SimpleBrownianBridgeDiffusion:

    def __init__(self, T: float, model: nn.Module, device: str, trainY: List, deterministic: bool = False):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device
        self.trainY = trainY

        self._EPS = 0.0000001
        self._deterministic = deterministic

    def train(self, batch_size, optimizer):
        """
        Corollary 2.7 implementation
        """

        # sample target images y
        y = sample_batch(self.trainY, batch_size, self.device)
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
        loss = nn.functional.mse_loss(y, img_predicted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

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

# def sample_batch(trainY, batch_size, device):
#     indices = torch.randperm(len(trainY))[:batch_size]
#     data = torch.stack([trainY[i] for i in indices]).to(device)
#     data = unflatten_data(data)  # Reshape to (batch_size, 1, 28, 28)
#     return torch.nn.functional.interpolate(data, size=(32, 32))

def sample_batch(trainY, batch_size, device):
    indices = torch.randperm(len(trainY))[:batch_size]
    data = torch.stack([trainY[i] for i in indices]).to(device)

    return data


def save_samples(diffusion_model, nb_images, n_batch, path):
    samples = diffusion_model.sample_w_reverse_process(n_samples=nb_images, use_tqdm=False)
    plt.figure(figsize=(17, 17))
    for i in range(nb_images):
        plt.subplot(9, 9, 1 + i)
        plt.axis('off')
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap='gray')
    plt.savefig(f'{path}/samples_epoch_{n_batch}.png')
    plt.close()


if __name__ == "__main__":

    check_torch_cuda()  # check CUDA

    parser = argparse.ArgumentParser(description="Set folders input and output")
    parser.add_argument('--data_folder', default='./data', type=str, required=False, help='Path to the data folder')
    parser.add_argument('--imgs_folder', default='./imgs/mnist/bridge', type=str, required=False, help='Path to the imgs folder')
    parser.add_argument('--models_folder', default='./aim-bridges-models/bridge', type=str, required=False, help='Path to the models folder')
    parser.add_argument('--dataset', default='mnist', help='Select dataset, mnist, cifar')
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
    batch_size = 128
    T = 0.1

    deterministic = False
    checkpoint_steps = (1000 / batch_size) * 64
    num_batches = 40_000

    if args.dataset == 'mnist':
        model = UNet()
    elif args.dataset == 'cifar':
        model = UNet(in_ch=3)
    else:
        raise f"unknown dataset: {args.dataset}"
        
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    diffusion_model = SimpleBrownianBridgeDiffusion(T, model, device, trainY, deterministic=deterministic)

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
