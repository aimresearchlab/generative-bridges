import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchmetrics.image.fid import FrechetInceptionDistance
from utils import load_mnist_dataset, flatten_image, unflatten_image, view_image, load_cifar10_dataset, remove_module_prefix 
from tqdm import tqdm
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def manifold_based_precision_recall(generated_features, real_features, k=3, threshold=None):
    """
    Compute precision and recall using manifold approximation via k-NN
    https://arxiv.org/abs/1904.06991 Improved Precision and Recall Metric for Assessing Generative Models
    Args:
        generated_features: Features from generated images (n_samples x feature_dim)
        real_features: Features from real images (n_samples x feature_dim)
        k: Number of nearest neighbors to use
        threshold: Distance threshold (if None, estimated from real data)
    """
    # Convert to numpy if needed
    if torch.is_tensor(generated_features):
        generated_features = generated_features.cpu().numpy()
    if torch.is_tensor(real_features):
        real_features = real_features.cpu().numpy()
    
    # Fit k-NN for both distributions
    real_nearest = NearestNeighbors(n_neighbors=k).fit(real_features)
    gen_nearest = NearestNeighbors(n_neighbors=k).fit(generated_features)
    
    # If threshold not provided, estimate it from real data
    if threshold is None:
        real_distances, _ = real_nearest.kneighbors(real_features)
        threshold = np.max(real_distances[:, 1:])  # Exclude distance to self
    
    # Calculate precision (how many generated samples are in real manifold)
    gen_to_real_distances, _ = real_nearest.kneighbors(generated_features)
    precision = np.mean(gen_to_real_distances[:, 1:] <= threshold)
    
    # Calculate recall (how many real samples are in generated manifold)
    real_to_gen_distances, _ = gen_nearest.kneighbors(real_features)
    recall = np.mean(real_to_gen_distances[:, 1:] <= threshold)
    
    return precision, recall

# Usage example:
def calculate_precision_recall(generated_images, real_images, feature_extractor):
    """
    Calculate precision and recall for generated images
    
    Args:
        generated_images: Generated images (n_samples x channels x height x width)
        real_images: Real images (n_samples x channels x height x width)
        feature_extractor: Model to extract features (e.g., pretrained CNN)
    """
    # Extract features
    with torch.no_grad():
        gen_features = feature_extractor(generated_images)
        real_features = feature_extractor(real_images)
    
    # Calculate metrics
    precision, recall = manifold_based_precision_recall(gen_features, real_features)
    
    return precision, recall

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

def process_image_auto(image):
    # Input validation
    if not torch.is_tensor(image):
        raise TypeError("Input must be a torch tensor")
    if not (len(image.shape) in [2, 3]):
        raise ValueError("Image must be 2D or 3D tensor")
        
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).clamp(0, 255).byte()
    
    # Check dimensionality and channels
    if len(image.shape) == 2:  # HxW
        image = image.unsqueeze(0).repeat(3, 1, 1)
    elif len(image.shape) == 3 and image.shape[0] == 1:  # 1xHxW
        image = image.repeat(3, 1, 1)
    elif len(image.shape) == 3 and image.shape[0] not in [1, 3]:
        raise ValueError(f"Unexpected number of channels: {image.shape[0]}")
    
    return image

def trivial_process_npz_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image)

def find_nearest_neighbor(generated_img, training_set, feature_extractor):
    # Extract features
    gen_features = feature_extractor(generated_img)
    train_features = feature_extractor(training_set)
    
    # Calculate distances
    distances = torch.cdist(gen_features, train_features)
    min_dist, nearest_idx = torch.min(distances, dim=1)
    
    return min_dist, nearest_idx

def calculate_fid(model_samples_path, train_dataset_path, max_images, dataset):
    # Load model samples
    data = np.load(model_samples_path)
    key = data.files[0]
    print(f"Using key '{key}' for images.")
    bridge_images = data[key]

    # Load dataset
    if dataset.lower() == 'cifar10':
        trainY, testY = load_cifar10_dataset(root=train_dataset_path, download=True)
        mono_channel = False
    elif dataset.lower() == 'mnist':
        trainY, testY = load_mnist_dataset(root=train_dataset_path, download=True)
        mono_channel = True
    else:
        raise ValueError("Invalid dataset. Choose either 'cifar10' or 'mnist'.")

    # Limit the number of images processed
    num_images = min(max_images, len(bridge_images), len(trainY))
    bridge_images = bridge_images[:num_images]
    trainY = trainY[:num_images]

    num_images_test = min(max_images, len(bridge_images), len(testY))
    bridge_images_test = bridge_images[:num_images_test]
    testY = testY[:num_images_test]

    # Process model samples
    npz_images_processed = torch.stack([trivial_process_npz_image(image) for image in bridge_images]).to('cuda')
    npz_images_processed_int = torch.stack([process_image_auto(samp) for samp in npz_images_processed])

    npz_images_processed_test = torch.stack([trivial_process_npz_image(image) for image in bridge_images_test]).to('cuda')
    npz_images_processed_int_test = torch.stack([process_image_auto(samp) for samp in npz_images_processed_test])

    print(f'num images train {num_images}; num images test {num_images_test}')

    # Process samples
    train_Y_int = torch.stack([process_image(samp, mono_channel) for samp in trainY]).to('cuda')
    test_Y_int = torch.stack([process_image(samp, mono_channel) for samp in testY]).to('cuda')

    # Initialize FID computation for train data
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
    fid_train = fid_value.item()
    print(f'fid train {fid_value}')

    # Initialize FID computation for test data
    fid = FrechetInceptionDistance().to('cuda')
    
    chunk_size = 6
    num_batches = (num_images_test + chunk_size - 1) // chunk_size  # Calculate total number of batches

    # Update FID with chunks of data
    with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
        for batch_start in range(0, num_images_test, chunk_size):
            batch_end = min(batch_start + chunk_size, num_images_test)
            
            batch_chunk_test = test_Y_int[batch_start:batch_end]
            fid.update(batch_chunk_test, real=True)

            batch_chunk_model = npz_images_processed_int_test[batch_start:batch_end]
            fid.update(batch_chunk_model, real=False)
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"Images Processed": batch_end})

    # Compute FID
    fid_value = fid.compute()
    fid_test = fid_value.item()
    print(f'fid test {fid_value}')

    return {'fid_train': fid_train, 'fid_test': fid_test,}

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