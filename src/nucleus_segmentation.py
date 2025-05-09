"""Skript pro segmentaci jader.

- Skript načte optimální parametry z sqlite databáze vygenerovanou Optunou (`src/optuna/nucleus_cytoplasm`)
- Skript vytvoří složku pro výsledky
"""

import warnings
from pathlib import Path
from typing import Callable

import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print  # noqa: A004
from rich.panel import Panel
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score

from src.utils import morph_operations, unsharp_mask_img

warnings.simplefilter(action="ignore", category=FutureWarning)


def nucleus_segment(
    img_path: Path,
    mask_size: int,
    iterations: int,
    min_size: int,
    radius: int,
    percent: float,
    threshold: float,
    threshold_func: Callable,
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

    img_unsharp = unsharp_mask_img(
        img, radius=radius, percent=percent, threshold=threshold
    )
    _, _, b1 = img_unsharp[:, :, 0], img_unsharp[:, :, 1], img_unsharp[:, :, 2]

    b_bin_otsu = b1 < threshold_func(b1)
    b_bin_otsu_morp = morph_operations(
        b_bin_otsu, mask_size=mask_size, iterations=iterations
    )

    b_bin_otsu_morp = remove_small_objects(b_bin_otsu_morp, min_size=min_size)

    labeled_img = label(b_bin_otsu_morp)
    n_objects = len(regionprops(labeled_img))

    f1 = jaccard_score(ground_truth.flatten(), b_bin_otsu_morp.flatten())

    if ground_truth.any():
        f1 = jaccard_score(ground_truth.flatten(), b_bin_otsu_morp.flatten())
        return b_bin_otsu_morp, f1, n_objects

    return b_bin_otsu_morp, None, n_objects


if __name__ == "__main__":
    from optuna import load_study

    output_path = Path("results_data/segmentation/nucleis")
    output_path.mkdir(parents=True, exist_ok=True)
    Path(output_path / "png").mkdir(parents=True, exist_ok=True)

    study = load_study(
        storage="sqlite:///nucleus_params.db", study_name="nucleus_segmentation"
    )
    params = study.best_params
    params_str = "\n".join(
        [
            f"[green]{key}[/green]: [yellow]{value}[/yellow]"
            for key, value in params.items()
        ]
    )

    panel = Panel(
        params_str, title="[bold blue]Best Parameters[/bold blue]", expand=False
    )
    print(panel)

    images_dir = Path("images").glob("*.png")

    results_df = pd.DataFrame(columns=["image", "jaccard", "n_objects"])

    for image in images_dir:
        ground_truth = Path("labels/orig/nucleus") / f"{image.stem}.npy"
        segmented, jaccard, n_objects = nucleus_segment(
            image,
            mask_size=params["mask_size"],
            iterations=params["iterations"],
            min_size=params["min_size"],
            radius=params["radius"],
            percent=params["percent"],
            threshold=params["threshold"],
            threshold_func=threshold_otsu,
            label_path=ground_truth,
        )

        print("Image: ", image.stem, "| IoU: ", jaccard, "| N objects: ", n_objects)
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

        plt.imsave(
            output_path / "png" / f"{image.stem}_segmented.png",
            segmented,
        )
        np.save(output_path / f"{image.stem}_segmented.npy", segmented)

    results_df = results_df.sort_values(by="image")
    results_df.to_csv(output_path / "results.csv", index=False)
    print("[green]Results saved.[/green]")
