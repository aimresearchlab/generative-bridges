import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict

CIFAR_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

def sample_batch(trainY, batch_size, device, interpolate_size=None):
    indices = torch.randperm(len(trainY))[:batch_size]
    data = torch.stack([trainY[i] for i in indices]).to(device)
    if interpolate_size:
        return torch.nn.functional.interpolate(data, size=(interpolate_size, interpolate_size))
    else:
        return data


def load_mnist_dataset(root: str = './data', download=True):
    """
    Function to load the MNIST dataset
    :return:
    """

    # Transform to tensor and flatten the image to a single vector
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the pixel values
    ])

    # Load the MNIST dataset
    train_set = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    # Extract only the images from datasets
    train_images = [data[0] for data in train_set]
    test_images = [data[0] for data in test_set]

    return train_images, test_images

def load_cifar10_dataset(root: str = './data', download=True):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        # NOT normalising as we loose detail
        #transforms.Normalize((0.5,), (0.5,)),  # Normalize the pixel values
    ])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)

    # Extract only the images from datasets
    train_images = [data[0] for data in train_set]
    test_images = [data[0] for data in test_set]

    return train_images, test_images


def view_image(image):
    """
    Function to view an image
    :param image:
    :return:
    """
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()


def flatten_image(image):
    """
    Function to flatten an image
    :param image:
    :return:
    """
    return image.view(-1)


def unflatten_image(vector, shape=(28, 28)):
    """
    Function to unflatten an image
    :param vector:
    :param shape:
    :return:
    """
    return vector.view(shape)


def check_torch_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        print(f"Number of available GPUs: {num_gpus}")

        # Iterate over each GPU and print its name and type
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs available")

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys.
    it gets added when using DataParallel"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict
