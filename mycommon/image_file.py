import rawpy
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
def read_cr2_image(file_path):
    """
    .CR2ファイルを読み込み、RGB画像に変換する
    """
    # CR2を読み込み
    with rawpy.imread(file_path) as raw:
        rgb_image = raw.postprocess()  # デモザイク処理してRGB画像に変換

    return rgb_image

def save_pigment_images(path, image):
    """"
    画像を保存するための関数
    """
    image_out = np.clip(image, 0, 1)
    image_out = cv2.cvtColor((image_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_out)
    return

def get_image_rotation_info(image_path):
    """"
    iphoneで撮影した画像の回転情報を読み込む
    意図しない回転に対応するため
    入力:image_path 画像のパス
    出力：orientation 回転情報
    """
    img = Image.open(image_path)
    # EXIFデータを取得
    exif_data = img.getexif()

    if not exif_data:
        return None
    exif = {TAGS.get(tag): value for tag, value in exif_data.items()}

    # Orientationタグを取得（回転情報が含まれている）
    orientation = exif.get('Orientation', None)
    return orientation
