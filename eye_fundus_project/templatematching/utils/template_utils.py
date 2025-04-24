import cv2

def select_roi_from_image(img):
    cv2.namedWindow("ROIselector", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(img)
    cv2.destroyAllWindows()
    return roi

def match_template(img, template):
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc