import cv2
import glob
import os
from utils.io_utils import natural_keys
from utils.template_utils import match_template

if __name__ == "__main__":
    input_dir = "inputs/video_frames/"
    output_dir = "outputs/cropped_frames/"
    template_path = "templates/tumu-1-template.png"

    files = sorted(glob.glob(os.path.join(input_dir, "*")), key=natural_keys)
    temp = cv2.imread(template_path, 0)
    os.makedirs(output_dir, exist_ok=True)

    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        pt = match_template(img, temp)
        th, tw = temp.shape[:2]
        cropped = img[pt[1]:pt[1]+th, pt[0]:pt[0]+tw]
        cv2.imwrite(os.path.join(output_dir, f"{i}.png"), cropped)
        print(f"\rProcessing... ({i+1}/{len(files)})", end="")