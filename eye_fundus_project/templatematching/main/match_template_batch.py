import cv2
import os
from utils.template_utils import match_template

if __name__ == "__main__":
    img = cv2.imread("inputs/sample.png")
    temp = cv2.imread("templates/template.png")

    pt = match_template(img, temp)
    h, w = temp.shape[:2]
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 0, 200), 3)

    output_dir = "outputs/results/"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "sample-templateresulted.png"), img)