"""Neoptimalizovaná segmentace jader."""

import warnings
from pathlib import Path

import mahotas as mh
import numpy as np
import pandas as pd
from rich import print  # noqa: A004
from rich.progress import Progress
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score

from src.utils import morph_operations, unsharp_mask_img

warnings.simplefilter(action="ignore", category=FutureWarning)


def nucleus_segment(
    img_path: Path,
    label_path: Path,
) -> tuple[np.ndarray, float | None, int]:
    """Watershed of nucleis.

    If label is provided, it will calculate the f1 score.

    Args:
        img_path (Path): Original image
        mask_size (int): Mask for noise removal
        iterations (int): Number of iterations for noise removal
        min_size (int): Minimum size of region to keep
        radius (int): Radius for unsharp mask
        percent (float): Percent for unsharp mask
        threshold (float): Threshold for unsharp mask
        threshold_func (callable): Threshold function
        label_path (Path): Ground truth labels. Defaults to None.

    Returns:
        np.ndarray, float, int: Segmented image, f1 score and number of objects

    """
    img = mh.imread(img_path)
    ground_truth = np.load(label_path)

    img_unsharp = unsharp_mask_img(img)
    _, _, b1 = img_unsharp[:, :, 0], img_unsharp[:, :, 1], img_unsharp[:, :, 2]

    b_bin_otsu = b1 < threshold_otsu(b1)
    b_bin_otsu_morp = morph_operations(b_bin_otsu)

    b_bin_otsu_morp = remove_small_objects(b_bin_otsu_morp)

    labeled_img = label(b_bin_otsu_morp)
    n_objects = len(regionprops(labeled_img))

    if ground_truth.any():
        iou = jaccard_score(ground_truth.flatten(), b_bin_otsu_morp.flatten())
        return b_bin_otsu_morp, iou, n_objects

    return b_bin_otsu_morp, None, n_objects


if __name__ == "__main__":
    output_path = Path("results_data/segmentation/nucleis/nonopt/")
    output_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(columns=["image", "jaccard", "n_objects"])

    images = list(Path("images").glob("*.png"))

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing images...", total=len(images))

        for image in images:
            ground_truth = Path("labels/orig/nucleus") / f"{image.stem}.npy"
            segmented, jaccard, n_objects = nucleus_segment(
                image,
                label_path=ground_truth,
            )

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "image": [image.stem],
                            "jaccard": [jaccard],
                            "n_objects": [n_objects],
                        }
                    ),
                ]
            )
            progress.update(task, advance=1)

    results_df = results_df.sort_values(by="image")
    results_df.to_csv(output_path / "results.csv", index=False)
    print("[green]Results saved.[/green]")
