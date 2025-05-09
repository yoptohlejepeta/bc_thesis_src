from pathlib import Path
from typing import Literal

import numpy as np
from skimage import io
from skimage.filters import threshold_li, threshold_otsu, threshold_yen
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score

from optuna import create_study
from src.utils import morph_operations, unsharp_mask_img


def nucleus_segmentation(
    img_path: Path,
    mask_size: int,
    iterations: int,
    min_size: int,
    radius: int,
    percent: float,
    threshold: float,
    threshold_func: Literal["li", "otsu", "yen"],
    label_path: Path,
) -> tuple[np.ndarray, float]:
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
        np.ndarray, float: Segmented image and f1 score

    """
    img = io.imread(img_path)
    label = np.load(label_path)

    img_unsharp = unsharp_mask_img(
        img, radius=radius, percent=percent, threshold=threshold
    )
    _, _, b = img_unsharp[:, :, 0], img_unsharp[:, :, 1], img_unsharp[:, :, 2]

    if threshold_func == "li":
        b_bin_otsu = b < threshold_li(b)
    elif threshold_func == "otsu":
        b_bin_otsu = b < threshold_otsu(b)
    elif threshold_func == "yen":
        b_bin_otsu = b < threshold_yen(b)

    b_bin_otsu_morp = morph_operations(
        b_bin_otsu, mask_size=mask_size, iterations=iterations
    )

    b_bin_otsu_morp = remove_small_objects(b_bin_otsu_morp, min_size=min_size)

    if label.any():
        iou = jaccard_score(label.flatten(), b_bin_otsu_morp.flatten())
        return b_bin_otsu_morp, iou

    return b_bin_otsu_morp, None


def objective(trial):
    imgs_dir = Path("images/")

    radius = trial.suggest_int("radius", 1, 100)
    percent = trial.suggest_int("percent", 1, 1000)
    threshold = trial.suggest_int("threshold", 1, 10)
    mask_size = trial.suggest_int("mask_size", 1, 10)
    iterations = trial.suggest_int("iterations", 1, 10)
    min_size = trial.suggest_int("min_size", 1, 1500)
    threshold_func = trial.suggest_categorical("threshold_func", ["li", "otsu", "yen"])

    scores = []

    for img_path in imgs_dir.glob("*.png"):
        _, iou = nucleus_segmentation(
            img_path=img_path,
            mask_size=mask_size,
            iterations=iterations,
            min_size=min_size,
            radius=radius,
            percent=percent,
            threshold=threshold,
            threshold_func=threshold_func,
            label_path="labels/orig/nucleus/" + img_path.stem + ".npy",
        )

        scores.append(iou)

    return np.mean(scores)


if __name__ == "__main__":
    study = create_study(
        storage="sqlite:///nucleus_params_jaccard.db",
        load_if_exists=True,
        study_name="nucleus_segmentation",
        direction="maximize",
    )
    study.optimize(objective, n_trials=1000)
