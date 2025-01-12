import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
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