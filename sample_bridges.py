import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming these imports are available in your environment
from unet import UNet, create_openAI_Unet
from diffusion import SimpleBrownianBridgeDiffusion

def remove_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def process_image(image):
    # Implement your image processing logic here
    # For now, we'll just return the image as is
    return image

def main(args):
    # Create model
    model = create_openAI_Unet()
    
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
        batch_samples = diffusion_model.sampling(n_samples=current_batch_size, image_channels=3, use_tqdm=False, n_steps=args.n_steps)
        
        for j, sample in enumerate(batch_samples):
            sample_idx = i * batch_size + j
            sample_np = sample.cpu().permute(1, 2, 0).numpy()
            sample_np = process_image(sample_np)
            plt.imsave(os.path.join(args.imgs_folder, f'sample_{sample_idx}.png'), sample_np)

    print(f"Generated {args.num_samples} samples in {args.imgs_folder}")

    # Create .npz file
    img_files = [f for f in os.listdir(args.imgs_folder) if f.endswith('.png')]
    images = np.zeros((len(img_files), 32, 32, 3), dtype=np.uint8)

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(args.imgs_folder, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        images[i] = img_array[:,:,:3]

    np.savez(os.path.join(args.imgs_folder, 'samples.npz'), images=images)
    print(f"Saved .npz file: {os.path.join(args.imgs_folder, 'samples.npz')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a diffusion model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--imgs_folder", type=str, required=True, help="Folder to save generated images")
    parser.add_argument("--T", type=float, default=0.1, help="Diffusion time")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise level")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for sampling")

    args = parser.parse_args()
    main(args)