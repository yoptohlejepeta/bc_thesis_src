"""Neoptimalizovaná segmentace cytoplazmy."""

from pathlib import Path

import colorcorrect.algorithm as cca
import mahotas as mh
import numpy as np
import pandas as pd
from rich import print  # noqa: A004
from rich.progress import Progress
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.metrics import f1_score

from src.utils import morph_operations, remove_small_regions


def cytoplasm_segment(
    img_path: Path,
    label_path: Path | None = None,
) -> tuple[np.ndarray, float | None, int]:
    """Watershed.

    If label is provided, it will calculate the f1 score.

    Args:
        img_path (Path): Original image
        label_path (Path): Ground truth labels. Defaults to None.

    Returns:
        np.ndarray, float, int: Segmented image, f1 score and number of objects
    """
    img = mh.imread(img_path)
    if label_path:
        ground_truth = np.load(label_path)

    img_c_corrected = cca.automatic_color_equalization(img)
    img_blurred = gaussian_filter(img_c_corrected, sigma=5)

    _, g, _ = img_blurred[:, :, 0], img_blurred[:, :, 1], img_blurred[:, :, 2]

    bin_cyto_nuclei = g < threshold_otsu(g)

    bin_cyto_nuclei = morph_operations(bin_cyto_nuclei)
    cytoplasm = remove_small_regions(bin_cyto_nuclei)

    labeled_cytoplasm = label(cytoplasm)
    regions = regionprops(labeled_cytoplasm)

    if label_path:
        f1 = f1_score(ground_truth.flatten(), cytoplasm.flatten())
        return cytoplasm, f1, len(regions)

    return cytoplasm, None, len(regions)


if __name__ == "__main__":
    labels_dir = Path("labels/orig/all").glob("*.npy")
    output_path = Path("results_data/segmentation/cytoplasm/nonopt")
    output_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(columns=["image", "f1", "n_objects"])

    images = list(Path("images").glob("*.png"))

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing images...", total=len(list(images)))
        for image in images:
            label_path = Path("labels/orig/cytoplasm/") / f"{image.stem}.npy"
            segmented, f1, n_objects = cytoplasm_segment(
                img_path=image,
                label_path=label_path,
            )

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {"image": [image.stem], "f1": [f1], "n_objects": [n_objects]}
                    ),
                ]
            )

    results_df = results_df.sort_values(by="image")
    results_df.to_csv(output_path / "results.csv", index=False)
    print("[green]Results saved.[/green]")
