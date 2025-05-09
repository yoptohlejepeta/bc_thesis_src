import mahotas as mh
import numpy as np
from PIL import Image, ImageFilter


def unsharp_mask_img(
    img: np.ndarray,
    radius: int = 10,
    percent: int = 300,
    threshold: int = 3,
) -> np.ndarray:
    """Unsharp mask image.

    Args:
    ----
        img (np.ndarray): Image to be processed.
        output_path (str): Path to save the processed image.
        radius (int, optional): Radius of the filter. Defaults to 10.
        percent (int, optional): Percentage of the sharpening. Defaults to 300.
        threshold (int, optional): Threshold of the filter. Defaults to 3.

    Returns:
    -------
        np.ndarray: Processed image.

    """
    img_pil = Image.fromarray(img, "RGB")

    bmp = img_pil.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    return np.array(bmp)


def morph_operations(
    img: np.ndarray, mask_size: int = 3, iterations: int = 5
) -> np.ndarray:
    img_bin = mh.close_holes(img)
    mask = np.ones((mask_size, mask_size))

    for _ in range(iterations):
        img_bin = mh.erode(img_bin, mask)

    for _ in range(iterations):
        img_bin = mh.dilate(img_bin, mask)

    return img_bin


def remove_small_regions(img, min_size=200, is_bin=False):
    if is_bin:
        img, _ = mh.label(img)

    sizes = mh.labeled.labeled_size(img)

    img_without_small_regions = mh.labeled.remove_regions_where(img, sizes < min_size)

    return img_without_small_regions
