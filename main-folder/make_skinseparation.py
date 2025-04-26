import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skin_separation_project.skin_separation.separation import makeSkinSeparation
from mycommon.select_folder import select_folder
from pathlib import Path
import glob
import os

##   ============== config=======================
melanin = [0.2203, 0.4788, 0.8499]
hemoglobin = [0.4350, 0.6929, 0.5750]
shading = [1.0, 1.0, 1.0]
vector = [melanin, hemoglobin, shading]

mask_type = "black"
bias_flag = True
bias_fixed = True
bias_Mel = -0.5
bias_Hem = -0.5
output_foldername = "result-bias-5"
##   ============== config=======================


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