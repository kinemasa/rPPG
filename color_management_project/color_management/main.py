import glob
import os
from pathlib import Path
from color_management.calibration import calibrate_image
from color_management.utils import select_folder
from color_management_project.color_management.config import observer, illumination, colorchecker, colorspace, output_foldername,method

def main():
    folder_num = 1
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]

    input_dir_list = []
    for _ in range(folder_num):
        input_dir = select_folder("補正画像フォルダを選択してください")
        input_dir_list.append(input_dir)

    for input_dir in input_dir_list:
        output_dir = os.path.join(input_dir, output_foldername)
        os.makedirs(output_dir, exist_ok=True)

        input_image_list = []
        for ext in image_extensions:
            input_image_list.extend(glob.glob(os.path.join(input_dir, ext)))

        for image_path in input_image_list:
            calibrate_image(image_path, output_dir, observer, illumination, colorchecker, colorspace,method)

    print("補正完了")

if __name__ == "__main__":
    main()