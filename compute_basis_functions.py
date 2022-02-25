import argparse
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from features import compute_basis_functions


def plot_basis_functions(basis_functions):
    assert len(basis_functions) <= 25
    fig, axes = plt.subplots(5, 5, figsize=(9, 9))

    for i in range(len(basis_functions)):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(basis_functions[i], cmap="gray")
        axes[row, col].set_title(f"Basis {i+1}")
        axes[row, col].set_axis_off()

    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, help="Image folder", default="/tmp/BSR/BSDS500/data/images")
    parser.add_argument("--output_file", type=str, help="Where to store basis functions as compressed numpy array", default="/tmp/basis_functions.npz")
    parser.add_argument("--whiten", dest="whiten", action="store_true", help="Whiten basis functions")
    parser.set_defaults(whiten=False)
    args = vars(parser.parse_args())

    # Make sure that basis functions file does not exist yet
    if os.path.exists(args["output_file"]):
        raise ValueError("Given basis functions file exists already")

    # Search for JPEG files
    img_filenames = glob(os.path.join(args["image_root"], "**/*.jpg"))

    # Compute basis functions
    basis_functions = compute_basis_functions(img_filenames=img_filenames, whiten=args["whiten"])

    # Save basis functions in compressed format
    np.savez_compressed(args["output_file"], basis_functions=basis_functions)
    print(f"Saved basis functions to {args['output_file']}")

    # Plot basis functions
    fig, axes = plot_basis_functions(basis_functions.reshape(-1, 5, 5))
    fig.savefig("results/basis_functions.png", dpi=300)
    plt.show()
