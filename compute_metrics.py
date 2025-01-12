import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchmetrics.image.fid import FrechetInceptionDistance
from utils import load_mnist_dataset, flatten_image, unflatten_image, view_image, load_cifar10_dataset, remove_module_prefix 
from tqdm import tqdm

def sample_batch(trainY, batch_size, device):
    indices = torch.randperm(len(trainY))[:batch_size]
    data = torch.stack([trainY[i] for i in indices]).to(device)
    return data

def process_image(image, mono_channel=False):
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = (image * 255).clamp(0, 255).byte()  # Scale to [0, 255] and convert to uint8
    if mono_channel:
        image = image.repeat(3, 1, 1)
    return image

def trivial_process_npz_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image)

def calculate_fid(model_samples_path, train_dataset_path, max_images, dataset):
    # Load model samples
    data = np.load(model_samples_path)
    key = data.files[0]
    print(f"Using key '{key}' for images.")
    bridge_images = data[key]

    # Load dataset
    if dataset.lower() == 'cifar10':
        trainY, _ = load_cifar10_dataset(root=train_dataset_path, download=True)
        mono_channel = False
    elif dataset.lower() == 'mnist':
        trainY, _ = load_mnist_dataset(root=train_dataset_path, download=True)
        mono_channel = True
    else:
        raise ValueError("Invalid dataset. Choose either 'cifar10' or 'mnist'.")

    # Limit the number of images processed
    num_images = min(max_images, len(bridge_images), len(trainY))
    bridge_images = bridge_images[:num_images]
    trainY = trainY[:num_images]

    # Process model samples
    npz_images_processed = torch.stack([trivial_process_npz_image(image) for image in bridge_images]).to('cuda')
    npz_images_processed_int = torch.stack([process_image(samp, mono_channel) for samp in npz_images_processed])

    # Process train samples
    train_Y_int = torch.stack([process_image(samp, mono_channel) for samp in trainY]).to('cuda')

    # Initialize FID computation
    fid = FrechetInceptionDistance().to('cuda')

    chunk_size = 6
    num_batches = (num_images + chunk_size - 1) // chunk_size  # Calculate total number of batches


    # Update FID with chunks of data
    with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
        for batch_start in range(0, num_images, chunk_size):
            batch_end = min(batch_start + chunk_size, num_images)
            
            batch_chunk_train = train_Y_int[batch_start:batch_end]
            fid.update(batch_chunk_train, real=True)

            batch_chunk_model = npz_images_processed_int[batch_start:batch_end]
            fid.update(batch_chunk_model, real=False)
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"Images Processed": batch_end})

    # Compute FID
    fid_value = fid.compute()
    return fid_value.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID between model samples and chosen dataset")
    parser.add_argument("--model_samples_path", type=str, required=True, help="Path to the .npz file containing model samples")
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the directory containing the dataset")
    parser.add_argument("--max_images", type=int, default=50000, help="Maximum number of images to process")
    parser.add_argument("--dataset", type=str, choices=['cifar10', 'mnist'], required=True, help="Dataset to use (cifar10 or mnist)")


    args = parser.parse_args()

    fid_score = calculate_fid(args.model_samples_path, args.train_dataset_path, args.max_images, args.dataset)
    print(f"FID: {fid_score}")

    #python compute_metrics.py --model_samples_path /workspace/aim-bridges-img/samples_1k_50000x32x32x3.npz --train_dataset_path /workspace/aim-bridges-data/cifar --dataset cifar10 --max_images 50000