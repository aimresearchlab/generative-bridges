import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.distributions as dist

def load_mnist_dataset():
    """
    Function to load the MNIST dataset
    :return:
    """

    # Transform to tensor and flatten the image to a single vector
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the pixel values
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])

    # Load the MNIST dataset
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_set, test_set

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

def sample_multivariate_gaussian(mean, covariance, num_samples=1):
    """
    Function to sample from an N-dimensional multivariate Gaussian distribution
    :param mean:
    :param covariance:
    :param num_samples:
    :return:
    """
    # Create a multivariate normal distribution
    mvn = dist.MultivariateNormal(mean, covariance)
    # Sample from the distribution
    samples = mvn.sample((num_samples,))
    return samples

if __name__ == '__main__':

    train_set, test_set = load_mnist_dataset()
    sample_image = train_set[0][0]
    flattened_image = flatten_image(sample_image)
    restored_image = unflatten_image(flattened_image)
    view_image(restored_image)

    print(len(train_set), len(test_set))

    # Sample showcase
    vector_len = len(flattened_image)
    mean_vector = torch.zeros(vector_len)  # mean = 0
    covariance_matrix = torch.eye(vector_len) * 0.01  # cov = 1
    samples = sample_multivariate_gaussian(mean_vector, covariance_matrix, num_samples=1)
    samples.shape