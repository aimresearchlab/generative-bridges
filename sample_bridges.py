import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming these imports are available in your environment
from unet_openai import create_openAI_Unet
from unet import UNet
from simple_bridge import SimpleBrownianBridgeDiffusion

def save_image(image, save_path):
    """
    Save an image to the specified path, handling both single-channel and multi-channel images.
    
    Args:
    image (np.ndarray): The image to save. Can be single-channel or multi-channel.
    save_path (str): The full path where the image should be saved.
    """
    # Ensure the image is a numpy array
    image_np = np.asarray(image)
    
    # Check the number of dimensions
    if image_np.ndim == 2:
        # It's a single-channel image (grayscale)
        plt.imsave(save_path, image_np, cmap='gray')
    elif image_np.ndim == 3:
        if image_np.shape[2] == 1:
            # It's a single-channel image with an extra dimension
            plt.imsave(save_path, image_np[:,:,0], cmap='gray')
        elif image_np.shape[2] in [3, 4]:
            # It's an RGB or RGBA image
            plt.imsave(save_path, image_np)
        else:
            raise ValueError(f"Unexpected number of channels: {image_np.shape[2]}")
    else:
        raise ValueError(f"Unexpected number of dimensions: {image_np.ndim}")

def remove_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def process_image(image):
    # Implement your image processing logic here
    # For now, we'll just return the image as is
    return image

def main(args):
    # Create model
    if args.dataset == 'mnist':
        image_channels=1
        image_size = 28
        if args.unet == 'berkan':
            print('running mnist simple unet')
            model = UNet()
        elif args.unet == 'openAI':
            print('running mnist openAI unet')
            model = create_openAI_Unet(model_config='mnist')
        print('creating diffusion model')
    elif args.dataset == 'cifar':
        image_channels=3
        image_size = 32
        if args.unet == 'berkan':
            print('running cifar simple unet')
            model = UNet(in_ch=3)
        elif args.unet == 'openAI':
            print('running cifar OpenAI unet')
            model = create_openAI_Unet(model_config='cifar')
    else:
        raise f"unknown dataset: {args.dataset}"
    
    # Load checkpoint
    print(f"Loading checkpoint '{args.model}'")
    checkpoint = torch.load(args.model)
    model.load_state_dict(remove_module_prefix(checkpoint))

    # Create diffusion model
    diffusion_model = SimpleBrownianBridgeDiffusion(args.T, model, 'cuda', trainY=None, deterministic=True)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.imgs_folder):
        os.makedirs(args.imgs_folder)

    # Generate samples in batches
    batch_size = args.batch_size
    num_batches = args.num_samples // batch_size
    remaining_samples = args.num_samples % batch_size

    for i in tqdm(range(num_batches + (1 if remaining_samples > 0 else 0)), desc="Generating batches"):
        current_batch_size = batch_size if i < num_batches else remaining_samples
        batch_samples = diffusion_model.sampling(n_samples=current_batch_size, image_channels=image_channels, use_tqdm=False, n_steps=args.n_steps)
        
        for j, sample in enumerate(batch_samples):
            sample_idx = i * batch_size + j
            sample_np = sample.cpu().permute(1, 2, 0).numpy()
            sample_np = process_image(sample_np)
            save_image(sample_np, os.path.join(args.imgs_folder, f'sample_{sample_idx}.png'))
            #plt.imsave(os.path.join(args.imgs_folder, f'sample_{sample_idx}.png'), sample_np)

    print(f"Generated {args.num_samples} samples in {args.imgs_folder}")

    # Create .npz file
    img_files = [f for f in os.listdir(args.imgs_folder) if f.endswith('.png')]
    images = np.zeros((len(img_files), image_size, image_size, image_channels), dtype=np.uint8)

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(args.imgs_folder, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        images[i] = img_array[:,:,:image_channels]

    np.savez(os.path.join(args.imgs_folder, 'samples.npz'), images=images)
    print(f"Saved .npz file: {os.path.join(args.imgs_folder, 'samples.npz')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a diffusion model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--imgs_folder", type=str, required=True, help="Folder to save generated images")
    parser.add_argument('--dataset', default='mnist', help='Select dataset, mnist, cifar')
    parser.add_argument('--unet', default='openAI', help='Select UNet: berkan or openAI')
    parser.add_argument("--T", type=float, default=0.1, help="Diffusion time")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise level")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for sampling")

    args = parser.parse_args()
    main(args)