import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchmetrics.image.fid import FrechetInceptionDistance

def load_cifar10_dataset(root, download=True):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = CIFAR10(root=root, train=True, download=download, transform=transform)
    testset = CIFAR10(root=root, train=False, download=download, transform=transform)
    return trainset.data, testset.data

def sample_batch(trainY, batch_size, device):
    indices = torch.randperm(len(trainY))[:batch_size]
    data = torch.stack([trainY[i] for i in indices]).to(device)
    return data

def process_image(image):
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = (image * 255).clamp(0, 255).byte()  # Scale to [0, 255] and convert to uint8
    return image

def trivial_process_npz_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image)

def calculate_fid(model_samples_path, train_dataset_path, max_images):
    # Load model samples
    data = np.load(model_samples_path)
    bridge_images = data['images']

    # Load CIFAR10 dataset
    trainY, _ = load_cifar10_dataset(root=train_dataset_path, download=True)

    # Limit the number of images processed
    num_images = min(max_images, len(bridge_images), len(trainY))
    bridge_images = bridge_images[:num_images]
    trainY = trainY[:num_images]

    # Process model samples
    npz_images_processed = torch.stack([trivial_process_npz_image(image) for image in bridge_images]).to('cpu')
    npz_images_processed_int = torch.stack([process_image(samp) for samp in npz_images_processed])

    # Process train samples
    train_Y_int = torch.stack([process_image(samp) for samp in trainY])

    # Initialize FID computation
    fid = FrechetInceptionDistance().to('cpu')

    chunk_size = 1000

    # Update FID with chunks of data
    for batch_start in range(0, num_images, chunk_size):
        batch_end = min(batch_start + chunk_size, num_images)
        
        batch_chunk_train = train_Y_int[batch_start:batch_end]
        fid.update(batch_chunk_train, real=True)

        batch_chunk_model = npz_images_processed_int[batch_start:batch_end]
        fid.update(batch_chunk_model, real=False)

    # Compute FID
    fid_value = fid.compute()
    return fid_value.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID between model samples and CIFAR10 dataset")
    parser.add_argument("model_samples_path", type=str, help="Path to the .npz file containing model samples")
    parser.add_argument("train_dataset_path", type=str, help="Path to the directory containing CIFAR10 dataset")
    parser.add_argument("--max_images", type=int, default=50000, help="Maximum number of images to process")

    args = parser.parse_args()

    fid_score = calculate_fid(args.model_samples_path, args.train_dataset_path, args.max_images)
    print(f"FID: {fid_score}")