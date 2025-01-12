import os

import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from utils import load_mnist_dataset

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

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.trainX = trainX

    def train(self, batch_size, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """
        x0 = sample_batch(self.trainX, batch_size, self.device)
        t = torch.randint(1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long)
        eps = torch.randn_like(x0)

        # Take one gradient descent step
        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x0_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        # The input to function_approximator must be 4D: (batch_size, channels, height, width)
        eps_predicted = self.function_approximator(x0_t, t - 1)
        loss = nn.functional.mse_loss(eps, eps_predicted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32),
                 use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """

        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                        device=self.device)
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * self.function_approximator(x, t - 1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        return x


def save_samples(diffusion_model, nb_images, n_batch):
    samples = diffusion_model.sample_w_reverse_process(n_samples=nb_images, use_tqdm=False)
    plt.figure(figsize=(17, 17))
    for i in range(nb_images):
        plt.subplot(9, 9, 1 + i)
        plt.axis('off')
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(), cmap='gray')
    plt.savefig(f'imgs/samples_epoch_{n_batch}.png')
    plt.close()

# Based on: https://papers-100-lines.medium.com/diffusion-models-from-scratch-mnist-data-tutorial-in-100-lines-of-pytorch-code-a609e1558cee
if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")

    trainX, testX = load_mnist_dataset()

    device = 'cuda'
    batch_size = 64  # max batch size on my 6GB VRAM laptop
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    diffusion_model = DiffusionModel(1000, model, device, trainX)

    # Create the "Img" folder if it does not exist
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    # Training
    for n_batch in tqdm(range(40_000)):
        loss = diffusion_model.train(batch_size, optimizer)

       # Save sampled images every 2000 epochs
        if (n_batch + 1) % 1000 == 0:
            save_samples(diffusion_model, nb_images=81, n_batch=n_batch + 1)

    # Save final results
    save_samples(diffusion_model, nb_images=81, n_batch='final')
