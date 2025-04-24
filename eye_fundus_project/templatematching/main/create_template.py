import cv2
import glob
import os
from utils.template_utils import select_roi_from_image

if __name__ == "__main__":
    dir_name = 'D:\\129\\masa5\\'
    files = glob.glob(os.path.join(dir_name, '*'))

    img = cv2.imread(files[0])
    roi = select_roi_from_image(img)

    cropped = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    cv2.rectangle(img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0,0,200), 3)

    output_dir = "outputs/templates/"
    os.makedirs(output_dir, exist_ok=True)
    subject = "tumu-2"

    cv2.imwrite(os.path.join(output_dir, f"{subject}-template.png"), cropped)
    cv2.imwrite(os.path.join(output_dir, f"{subject}-checktemplate.png"), img)