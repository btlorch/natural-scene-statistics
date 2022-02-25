"""
Helper methods to extract a 125-dimensional features using a natural scene statistics (NSS)-based model.
These features can be used to predict the upscaling factor.

Goodall, T. R., Katsavounidis, I., Li, Z., Aaron, A., & Bovik, A. C. (2016). Blind picture upscaling ratio prediction. IEEE Signal Processing Letters, 23(12), 1801-1805.
https://live.ece.utexas.edu/publications/2016/goodall2016blind.pdf
"""
from imageio import imread
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import convolve2d
from skimage.color import rgb2gray


def sample_patches(img_filenames, num_patches_per_image=2000, patch_size=5):
    """
    Randomly crop patches from an array of images given by their filenames
    :param img_filenames: filenames
    :param num_patches_per_image: number of patches to crop per image
    :param patch_size: isotropic patch size
    :return: ndarray of shape [num_patches, patch_size, patch_size]
    """
    patches = []

    for img_filename in tqdm(img_filenames, desc="Loop filenames"):
        try:
            img = load_img(img_filename)
        except ValueError as e:
            print(f"Failed to decode image \"{img_filename}\". Skipping this image.")
            continue

        height, width = img.shape
        for _ in range(num_patches_per_image):
            offset_y = np.random.randint(0, height - patch_size + 1)
            offset_x = np.random.randint(0, width - patch_size + 1)

            patch = img[offset_y:offset_y + patch_size, offset_x:offset_x + patch_size]
            patch_height, patch_width = patch.shape
            assert patch_height == patch_width == patch_size

            patches.append(patch)

    return np.stack(patches, axis=0)


def gaussian_kernel(kernel_size=5, sigma=2.):
    """
    Creates a 2D Gaussian kernel
    """
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def multiply_gaussian_kernel(patches, kernel_size):
    kernel = gaussian_kernel(kernel_size=kernel_size, sigma=2)

    # Normalize to unit maximum value
    kernel = kernel / np.max(kernel)

    return patches * kernel[None, :, :]


def compute_basis_functions(img_filenames, whiten=False, patch_size=5):
    # Derive basis functions from image patches
    patches = sample_patches(img_filenames, patch_size=patch_size)

    # Multiply with 5x5 Gaussian mask to reduce energy at the patch boundaries
    weighted_patches = multiply_gaussian_kernel(patches, kernel_size=patch_size)

    # Obtain basis functions using PCA
    pca = PCA(whiten=whiten)
    pca.fit(weighted_patches.reshape(-1, patch_size * patch_size))

    return pca.components_.reshape(-1, patch_size, patch_size)


def compute_mscn_maps(img, basis_functions, epsilon=1e-9):
    """
    Given an image and the model's basis functions, compute mean-subtracted contrast-normalized (MSCN) maps
    :param img: grayscale image
    :param basis_functions: ndarray of shape [num_basis_functions, 5, 5]
    :param epsilon: tiny constant to avoid division by zero
    :return: MSCN maps of shape [num_basis_functions, height, width]
    """
    num_basis_functions = len(basis_functions)

    # 2D circularly-symmetric Gaussian weighting sampled out to 3 standard deviations and normalized to unit volume
    weighting = gaussian_kernel(kernel_size=11, sigma=3)

    mscn_maps = []
    # For each of the 25 basis functions
    for i in range(num_basis_functions):
        # Convolve image with i-th basis function to obtain a response map
        response_map = convolve2d(img, basis_functions[i], mode="same")

        # Convolve the response map with a circular Gaussian filter
        mean_map = convolve2d(response_map, weighting, mode="same")
        std_map = np.sqrt(convolve2d((response_map - mean_map) ** 2, weighting, mode="same"))

        # Divisive normalization of response map
        # The result is called mean-subtracted contrast-normalized (MSCN) map
        mscn_map = (response_map - mean_map) / (std_map + epsilon)
        mscn_maps.append(mscn_map)

    # Stack MSCN maps for each of the basis functions
    return np.stack(mscn_maps, axis=0)


def paired_product_maps(mscn_maps):
    """
    Compute paired product maps from MSCN maps
    :param mscn_maps: ndarray of shape [num_basis_functions, height, width]
    :return: 4-tuple with paired product maps for horizontal, vertical, main-diagonal, and second-diagonal directions
    """
    # The remaining features are computed from the coefficient maps of shifted paired product maps (PPM)
    ppm_horizontal = mscn_maps[:, :, :-1] * mscn_maps[:, :, 1:]

    ppm_vertical = mscn_maps[:, :-1, :] * mscn_maps[:, 1:, :]

    # Main diagonal
    ppm_diagonal1 = mscn_maps[:, :-1, :-1] * mscn_maps[:, 1:, 1:]

    # Second diagonal
    ppm_diagonal2 = mscn_maps[:, :-1, 1:] * mscn_maps[:, 1:, :-1]

    return ppm_horizontal, ppm_vertical, ppm_diagonal1, ppm_diagonal2


def features_from_mscn_maps(mscn_maps):
    """
    Extract features from MSCN maps
    :param mscn_maps: ndarray of shape [num_basis_functions, height, width]
    :return: 125-dimensional feature vector
    """

    ppm_horizontal, ppm_vertical, ppm_diagonal1, ppm_diagonal2 = paired_product_maps(mscn_maps)

    # Compute standard deviation of each map
    stds = np.std(mscn_maps, axis=(1, 2))

    # The remaining features are computed from the coefficient maps of shifted paired product maps (PPM)
    std_ppm_horizontal = np.std(ppm_horizontal, axis=(1, 2))
    std_ppm_vertical = np.std(ppm_vertical, axis=(1, 2))
    std_ppm_diagonal1 = np.std(ppm_diagonal1, axis=(1, 2))
    std_ppm_diagonal2 = np.std(ppm_diagonal2, axis=(1, 2))

    # Concatenate all features
    return np.concatenate([stds, std_ppm_horizontal, std_ppm_vertical, std_ppm_diagonal1, std_ppm_diagonal2], axis=0)


def extract_features_from_image(img, basis_functions):
    """
    Extracts 125 features from the given image
    :param img: grayscale image in range [0, 255]
    :param basis_functions: ndarray of shape [num_basis_functions, 5, 5]
    :return: feature vector of shape [125]
    """
    # Convert image to MSCN maps
    mscn_maps = compute_mscn_maps(img, basis_functions)

    # Extract features from MSCN maps
    features = features_from_mscn_maps(mscn_maps)

    return features


def load_img(filename):
    img = imread(filename).astype(np.float32)

    if len(img.shape) > 2:
        img = rgb2gray(img)

    return img


def extract_features_from_file(filename, basis_functions):
    img = load_img(filename)

    return extract_features_from_image(img=img, basis_functions=basis_functions)
