import glob
import os
import sys
import matplotlib.pyplot as plt
from enhancement.filters import isotropic_wavelet_filter
from enhancement.image_utils import extract_green_channel
from utils.io_utils import natural_keys

def main():
    input_dir = "inputs/tumu-13-mini1/"
    output_dir = "outputs/tumu-13-mini1/"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*")), key=natural_keys)

    for i, file in enumerate(files):
        green = extract_green_channel(file)
        enhanced = isotropic_wavelet_filter(green)
        plt.imsave(os.path.join(output_dir, f"{i}.png"), enhanced, cmap='gray')
        sys.stdout.write(f"\rProcessing... ({i + 1}/{len(files)})")
        sys.stdout.flush()

if __name__ == "__main__":
    main()