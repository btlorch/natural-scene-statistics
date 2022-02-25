import argparse
import numpy as np
from glob import glob
import pandas as pd
from features import load_img, extract_features_from_image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
import os
import cv2
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def rescale_opencv(img, scale_factor, interpolation_order):
    """
    Rescale image by given scale factor
    :param img:
    :param scale_factor:
    :param interpolation_order:
        1 == cv2.INTER_LINEAR
        2 == cv2.INTER_CUBIC
        4 == cv2.INTER_LANCZOS4
    :return:
    """
    old_height, old_width = img.shape
    new_height = int(np.round(old_height * scale_factor))
    new_width = int(np.round(old_width * scale_factor))

    # Works with float images. Can produce float values outside the input range
    new_image = cv2.resize(img, (new_width, new_height), interpolation=interpolation_order)
    return new_image


def process_file_opencv(image_file_and_scale_factor, basis_functions, interpolation_order):
    # Destruct tuple argument
    image_file, scale_factor = image_file_and_scale_factor

    # Image is in range [0, 255]
    img = load_img(image_file)

    # Rescale
    rescaled = rescale_opencv(img, scale_factor, interpolation_order)

    # Extract features
    return extract_features_from_image(rescaled, basis_functions=basis_functions)


def extract_features_multiprocessing(image_files, scale_factors, basis_functions, interpolation_order):
    num_files = len(image_files)

    num_threads = min(mp.cpu_count(), len(image_files))
    with mp.Pool(processes=num_threads) as p:
        features_buffer = []

        # imap preserves the order
        for features in tqdm(p.imap(partial(process_file_opencv, basis_functions=basis_functions, interpolation_order=interpolation_order), zip(image_files, scale_factors)), desc="Feature extraction", total=num_files):
            features_buffer.append(features)

        # Stack all features along first axis
        return np.stack(features_buffer, axis=0)


def calculate_correlations(features, scale_factors):
    """
    Calculate absolute Spearman rank-order correlation coefficient between each of the 125 features and the scale factors
    :param features: ndarray of shape [num_samples, 125]
    :param scale_factors: scale factors array of shape [num_samples]
    :return: data frame containing all the correlations
    """

    buffer = []

    # Iterate over basis functions
    num_basis_functions = 25
    for basis_function_idx in range(num_basis_functions):
        # 5 features corresponding to the current basis function
        basis_function_features = features[:, basis_function_idx::num_basis_functions]

        # Calculate absolute Spearman correlation coefficients
        mscn_correlation = np.abs(spearmanr(scale_factors, basis_function_features[:, 0])[0])
        ppm_horizontal_correlation = np.abs(spearmanr(scale_factors, basis_function_features[:, 1])[0])
        ppm_vertical_correlation = np.abs(spearmanr(scale_factors, basis_function_features[:, 2])[0])
        ppm_diagonal1_correlation = np.abs(spearmanr(scale_factors, basis_function_features[:, 3])[0])
        ppm_diagonal2_correlation = np.abs(spearmanr(scale_factors, basis_function_features[:, 4])[0])

        records = {
            "mscn_correlation": mscn_correlation,
            "ppm_horizontal_correlation": ppm_horizontal_correlation,
            "ppm_vertical_correlation": ppm_vertical_correlation,
            "ppm_diagonal1_correlation": ppm_diagonal1_correlation,
            "ppm_diagonal2_correlation": ppm_diagonal2_correlation,
            "basis_function": basis_function_idx
        }
        buffer.append(records)

    correlations_df = pd.DataFrame(buffer)
    return correlations_df


def calculate_correlations_with_files(image_files, basis_functions, scale_factor_min=1., scale_factor_max=3.):
    """
    Scale each image by a randomly drawn scale factor in the given range. Extract natural scene statistics features from the scaled image.
    Repeat this procedure for bilinear, bicubic, and Lanczos interpolation.
    :param image_files: list of image filenames
    :param basis_functions: ndarray of shape [num_basis_functions, 25]
    :param scale_factor_min: lower end of uniform scale factor distribution
    :param scale_factor_max: upper end of uniform scale factor distribution
    :return: data frame with the correlations between the 125 features and the scale factor
    """
    # Bilinear interpolation
    scale_factors_bilinear = np.random.uniform(low=scale_factor_min, high=scale_factor_max, size=len(image_files))

    features_bilinear = extract_features_multiprocessing(
        image_files=image_files,
        scale_factors=scale_factors_bilinear,
        basis_functions=basis_functions,
        interpolation_order=cv2.INTER_LINEAR,
    )

    # Bicubic interpolation
    scale_factors_bicubic = np.random.uniform(low=scale_factor_min, high=scale_factor_max, size=len(image_files))

    features_bicubic = extract_features_multiprocessing(
        image_files=image_files,
        scale_factors=scale_factors_bicubic,
        basis_functions=basis_functions,
        interpolation_order=cv2.INTER_CUBIC,
    )

    # Lanczos interpolation
    scale_factors_lanczos = np.random.uniform(low=scale_factor_min, high=scale_factor_max, size=len(image_files))

    features_lanczos = extract_features_multiprocessing(
        image_files=image_files,
        scale_factors=scale_factors_lanczos,
        basis_functions=basis_functions,
        interpolation_order=cv2.INTER_LANCZOS4,
    )

    # Concatenate features: The result is of shape [number of images * 3, 125]
    features = np.concatenate([features_bilinear, features_bicubic, features_lanczos], axis=0)

    # Concatenate corresponding scale factors
    scale_factors = np.concatenate([scale_factors_bilinear, scale_factors_bicubic, scale_factors_lanczos], axis=0)

    # Calculate correlations between all 125 features and the scale factors
    return calculate_correlations(features, scale_factors)


def plot_correlations(df):
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    ax.scatter(df["basis_function"], df["mscn_correlation"], marker="o", label=r"$\sigma_m$", color="k")
    ax.scatter(df["basis_function"], df["ppm_horizontal_correlation"], marker="x", label=r"$pp_H$", color="green")
    ax.scatter(df["basis_function"], df["ppm_vertical_correlation"], marker="D", label=r"$pp_V$", color="red")
    ax.scatter(df["basis_function"], df["ppm_diagonal1_correlation"], marker="^", label=r"$pp_{D1}$", color="orange")
    ax.scatter(df["basis_function"], df["ppm_diagonal2_correlation"], marker="*", label=r"$pp_{D2}$", color="purple")
    ax.set_xticks(df["basis_function"])

    ax.set_xlabel("PCA basis filter number")
    ax.set_ylabel("Absolute value of SROCC")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="/tmp/BSR/BSDS500/data/images")
    parser.add_argument("--output_dir", type=str, help="Folder where to store results", default="/tmp")
    parser.add_argument("--basis_functions_file", type=str, help="Where to find basis functions", default="/tmp/basis_functions.npz")
    args = vars(parser.parse_args())

    # Retrieve stored basis functions
    basis_functions = np.load(args["basis_functions_file"])["basis_functions"]

    # Search for JPEG files
    img_filenames = glob(os.path.join(args["image_root"], "**/*.jpg"))

    # Upscale images and calculate correlation between features and scale factors
    correlations_df = calculate_correlations_with_files(
        image_files=img_filenames,
        basis_functions=basis_functions,
    )

    # Save correlation to csv file
    output_csv = os.path.join(args["output_dir"], f"{time.strftime('%Y_%m_%d')}-eval_features_upscaling_ratio.csv")
    correlations_df.to_csv(output_csv, index=False)
    print(f"Saved results to \"{output_csv}\"")

    # Create scatter plot
    fig, axes = plot_correlations(correlations_df)
    fig.savefig("results/correlations.png", dpi=300)
    plt.show()
