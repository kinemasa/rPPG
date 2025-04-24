import glob
import os
from integrate.integration import integrate_in_groups
from utils.io_utils import natural_keys

def main():
    input_dir = "inputs/tumu13/"
    output_dir = "outputs/tumu13_addmean_grouped/"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*')), key=natural_keys)
    integrate_in_groups(files, group_size=5, output_path=output_dir)

if __name__ == "__main__":
    main()