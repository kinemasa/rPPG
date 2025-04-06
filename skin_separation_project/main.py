# main.py
from skin_separation.separation import makeSkinSeparation
from skin_separation.utils import select_folder
from config import vector, mask_type, bias_flag, bias_fixed, bias_Mel, bias_Hem, output_foldername
from pathlib import Path
import glob
import os

def main():
    folder_num = 1
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif"]

    input_dir_list = []
    for _ in range(folder_num):
        input_dir = select_folder("画像フォルダを選んでください")
        input_dir_list.append(input_dir)

    for input_dir in input_dir_list:
        input_image_list = []
        for ext in image_extensions:
            input_image_list.extend(glob.glob(os.path.join(input_dir, ext)))

        output_dir = os.path.join(input_dir, output_foldername)
        os.makedirs(output_dir, exist_ok=True)

        makeSkinSeparation(
            INPUT_DIR=input_dir,
            input_image_list=input_image_list,
            OUTPUT_DIR=output_dir,
            vector=vector,
            mask_type=mask_type,
            bias_flag=bias_flag,
            bias_fixed=bias_fixed,
            bias_Mel=bias_Mel,
            bias_Hem=bias_Hem
        )

    print("Done.")

if __name__ == "__main__":
    main()