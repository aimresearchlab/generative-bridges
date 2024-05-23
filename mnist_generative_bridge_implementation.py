import torch
import torch.distributions as dist

from utils import load_mnist_dataset, flatten_image, unflatten_image, view_image


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

    print(f"GPU available: {torch.cuda.is_available()}")

    trainX, testX = load_mnist_dataset()
    sample_image = trainX[0]
    unflattened_image = unflatten_image(sample_image)
    restored_flattened_image = flatten_image(sample_image)

    view_image(unflattened_image)

    print(len(trainX), len(testX))

    # Sampling showcase
    vector_len = len(restored_flattened_image)
    mean_vector = torch.zeros(vector_len)  # mean = 0
    covariance_matrix = torch.eye(vector_len) * 0.01  # cov = 1 (isotropic)
    samples = sample_multivariate_gaussian(mean_vector, covariance_matrix, num_samples=1)
    samples.shape
