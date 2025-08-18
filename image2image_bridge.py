import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet import UNet
from unet_openai import create_openAI_Unet, model_and_diffusion_defaults_cifar10
from utils import load_mnist_dataset, load_cifar10_dataset, check_torch_cuda, sample_batch, load_cifar10grey_dataset


class SimpleBrownianBridgeDiffusion:

    # (Sean): Added trainX as input for an initial distribution
    def __init__(self, T: float, model: nn.Module, device: str, trainY: List, trainX: List, deterministic: bool = False, interpolate_size = None):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device
        self.trainY = trainY
        self.trainX = trainX  # (Sean): Added trainX for initial distribution

        self._EPS = 0.0000001
        self._deterministic = deterministic
        self.interpolate_size = interpolate_size

    def train(self, batch_size, optimizer):
        """
        Corollary 2.7 implementation
        """

        # sample target images y
        # (Sean): Step 1 in 2.2.1
        y = sample_batch(self.trainY, batch_size, self.device, interpolate_size=self.interpolate_size)

        # sample random times from [0,T)
        # (Sean): Step 1 in 2.2.1
        t = torch.rand(batch_size, device=self.device, dtype=torch.float32) * self.T

        # Unsqueeze t to align dimensions
        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, 1, 1, 1)

        # (Sean): sample x from initial distribution 
        # (Sean): Step 1 in 2.2.1
        x = sample_batch(self.trainX, batch_size, self.device, interpolate_size=self.interpolate_size)
        
        # compute E_t
        # (Sean): Step 2 in 2.2.1
        E = x + (y - x) * t / self.T
        
        # compute V_t
        # (Sean): Step 2 in 2.2.1
        V = t * (self.T - t) / self.T

        # compute xi_t sampling 
        # (Sean): Step 2 in 2.2.1
        xi = E + torch.randn(y.shape, device=self.device) * torch.sqrt(V)


        # Take one gradient descent step
        # The input to function_approximator must be 4D: (batch_size, channels, height, width)
        # t is a list of times [batch_size] (hence squeeze)

        # (Sean): Step 3 in 2.2.1
        img_predicted = self.function_approximator(xi, t.squeeze())
        loss = nn.functional.mse_loss(y, img_predicted)

        # (Sean): Step 4 in 2.2.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    # (Sean): n_steps here represents sampling steps, I've set it to 10 to reduce computational load
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), n_steps=10,
                 use_tqdm=True):
        """
        Corollary 2.9 implementation
        """
        # create n_steps t
        # messing about with n_steps as small eps gets rounded away
        # (Sean): Step 1 in 2.2.2
        t_values = torch.linspace(0, self.T, steps=n_steps + 1, device=self.device) * (self.T - self._EPS) / self.T

        # sample x
        # (Sean): Step 4 in 2.2.2
        delta = (self.T - self._EPS) / n_steps

        # (Sean): Step 2 in 2.2.2
        # (Sean): Changed to sample from trainX for initial distribution
        x = sample_batch((self.trainX, n_samples, image_channels, img_size[0], img_size[1]), 
                         device=self.device)

        xi_t = x
        #Z_t = x


        progress_bar = tqdm if use_tqdm else lambda x: x
        for idx, t in enumerate(progress_bar(t_values[:-1])):

            # (Sean): Step 4a in 2.2.2
            t = torch.tensor(t, device=self.device)
            inv_lambda = 1 / (self.T - t)

            # mu_t = x; Sigma_tt = t
            #Z = x + torch.randn(x.shape, device=self.device)*torch.sqrt(t)
            # need a list of repeated t with len same as n_samples

            t_list = [t for _ in range(n_samples)]
            #xi = xi_t + inv_lambda*(self.function_approximator(xi_t, torch.stack(t_list)) - xi_t )*delta + (Z - Z_t)

            # (Sean): Step 4b and 4c in 2.2.2
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


if __name__ == "__main__":

    check_torch_cuda()  # check CUDA

    parser = argparse.ArgumentParser(description="Set folders input and output")
    parser.add_argument('--data_folder', default='./data', type=str, required=False, help='Path to the data folder')
    parser.add_argument('--imgs_folder', default='./imgs/mnist/bridge', type=str, required=False, help='Path to the imgs folder')
    parser.add_argument('--models_folder', default='./aim-bridges-models/bridge', type=str, required=False, help='Path to the models folder')
    parser.add_argument('--dataset', default='mnist', help='Select dataset, mnist, cifar')
    parser.add_argument('--unet', default='openAI', help='Select UNet: berkan or openAI')
    parser.add_argument("--T", type=float, default=0.1, help="Diffusion time")
    parser.add_argument("--batch_size", type=int, default=72, help="Batch Size")
    parser.add_argument('--download_data', action='store_true', help='Download the data')
    parser.add_argument('--verbose', action='store_true', help='Download the data')

    args = parser.parse_args()
    print(args)
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

    # (Sean): Added support for cifar-greyscale dataset
    elif args.dataset == 'cifar-greyscale':
         trainY, testY = load_cifar10grey_dataset(root=args.data_folder, download=args.download_data)
    else:
        raise f"unknown dataset: {args.dataset}"

    # basic config
    device = 'cuda'
    batch_size = args.batch_size
    T = args.T
    learning_rate = 1e-4

    deterministic = False
    checkpoint_steps = (10000 / batch_size) * 64
    
    # (Sean): Made num_batches smaller instead of 40000 so tests can run quicker
    num_batches = 2500

    if args.dataset == 'mnist':
        if args.unet == 'berkan':
            print('running mnist simple unet')
            model = UNet()
        elif args.unet == 'openAI':
            print('running mnist openAI unet')
            model = create_openAI_Unet(model_config='mnist')
        print('creating diffusion model')
        diffusion_model = SimpleBrownianBridgeDiffusion(T, model, device, trainY, deterministic=deterministic, interpolate_size=32)
    elif args.dataset == 'cifar':
        if args.unet == 'berkan':
            print('running cifar simple unet')
            model = UNet(in_ch=3)
        elif args.unet == 'openAI':
            print('running cifar OpenAI unet')
            model = create_openAI_Unet(model_config='cifar')
        diffusion_model = SimpleBrownianBridgeDiffusion(T, model, device, trainY, deterministic=deterministic)

    # (Sean): UNET for CIFAR-10 grayscale dataset
    elif args.dataset == 'cifar-greyscale':
        if args.unet == 'berkan':
            print('running cifar simple unet')
            model = UNet(in_ch=1)
        elif args.unet == 'openAI':
            print('running cifar-greyscale OpenAI unet')
            model = create_openAI_Unet(model_config='cifar-greyscale')
        diffusion_model = SimpleBrownianBridgeDiffusion(T, model, device, trainY, deterministic=deterministic)
    
    else:
        raise f"unknown dataset: {args.dataset}"
    
    print('data parallel')
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    det_name = 'det' if deterministic else 'stoch'
    model_name = f'{args.dataset}_batch_{batch_size}_T_{T}_{det_name}_{args.unet}_50M'
    print(f'data_set: {args.dataset}; batch_size: {batch_size}; T: {T}; deterministic: {deterministic}')

    # Training
    writer = SummaryWriter(log_dir=f'{args.models_folder}/runs/{model_name}')  # Initialize TensorBoard writer
    for n_batch in tqdm(range(num_batches)):
        loss = diffusion_model.train(batch_size, optimizer)
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        writer.add_scalar('Loss/train', loss, n_batch)  # Log the loss to TensorBoard

        if n_batch % checkpoint_steps == 0 or n_batch == 1000:
            checkpoint_path = os.path.join(check_point_folder, f"{model_name}_checkpoint_n_batch_{n_batch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            if args.verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final model
    final_model_path = os.path.join(f'{args.models_folder}/checkpoints', f"{model_name}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()  # Close the writer when done

    #python simple_bridge.py --data_folder /workspace/aim-bridges-data/cifar10/ --imgs_folder /workspace/aim-bridges-img/ --models_folder /workspace/aim-bridges-models/bridge/ --dataset cifar --unet openAI
