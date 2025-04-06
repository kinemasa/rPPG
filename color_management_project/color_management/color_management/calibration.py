import os
import numpy as np
import imageio
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
from .utils import select_folder

colour.plotting.colour_style()


def calibrate_image(input_image_path, OUTPUT_DIR, observer, illumination, colorchecker, colorspace,method):
    """
    カラーチャートを用いた色補正処理を行い、補正画像を保存する関数
    """
    image_basename = os.path.splitext(os.path.basename(input_image_path))[0]
    output_path = os.path.join(OUTPUT_DIR, image_basename + ".png")

    image = colour.cctf_decoding(colour.io.read_image(input_image_path))
    SWATCHES = []

    for colour_checker_data in detect_colour_checkers_segmentation(image, additional_data=True):
        swatch_colours = colour_checker_data.swatch_colours
        swatch_masks = colour_checker_data.swatch_masks
        colour_checker_image = colour_checker_data.colour_checker
        SWATCHES.append(swatch_colours)

        # 可視化
        masks_i = np.zeros(colour_checker_image.shape)
        for mask in swatch_masks:
            masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1

        colour.plotting.plot_image(
            colour.cctf_encoding(np.clip(colour_checker_image + masks_i * 0.25, 0, 1))
        )

    if not SWATCHES:
        print(f"No colour checker detected in {input_image_path}")
        return

    swatches = SWATCHES[0]

    illumination_setting = colour.CCS_ILLUMINANTS[observer][illumination]
    colorchecker_setting = colour.CCS_COLOURCHECKERS[colorchecker]

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(colorchecker_setting.data.values())),
        colorspace,
        colorchecker_setting.illuminant
    )

    corrected_image = colour.colour_correction(
        image, swatches, REFERENCE_SWATCHES, method=method
    )
    encoded_image = colour.cctf_encoding(corrected_image)
    encoded_image_uint8 = np.clip(encoded_image * 255, 0, 255).astype(np.uint8)
    # `imageio` を使って保存
    imageio.imwrite(output_path, encoded_image_uint8)
    print(f"Saved calibrated image to {output_path}")
