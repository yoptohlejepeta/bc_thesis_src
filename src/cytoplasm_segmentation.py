"""Skript pro segmentaci cytoplasmy.

- Skript načte optimální parametry z sqlite databáze vygenerovanou Optunou (`src/optuna/optuna_cytoplasm`)
"""

from pathlib import Path

import colorcorrect.algorithm as cca
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print  # noqa: A004
from rich.panel import Panel
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.metrics import f1_score

from src.utils import morph_operations, remove_small_regions


def cytoplasm_segment(
    img_path: Path,
    mask_size: int,
    iterations: int,
    min_size: int,
    slope: int,
    limit: int,
    samples: int,
    sigma: float,
    label_path: Path | None = None,
) -> tuple[np.ndarray, float | None, int]:
    """Segmentace cytoplasmy.

    Pokud je poskytnuta cesta k labelu, vrátí F1 skóre.

    Args:
        img_path (Path): Cesta k obrázku
        mask_size (int): Velikost masky pro morfologické operace
        iterations (int): Počet iterací morfologických operací
        min_size (int): Minimální velikost objektů
        slope (int): Slope pro `automatic_color_equalization`
        limit (int): Limit pro `automatic_color_equalization`
        samples (int): Samples pro `automatic_color_equalization`
        sigma (float): Sigma gaussův filtr
        label_path (Path): Cesta k labelu. Defaults to None.

    Returns:
        np.ndarray, float, int: Segmentovaný obrázek, F1 skóre a počet objektů

    """
    img = mh.imread(img_path)
    if label_path:
        ground_truth = np.load(label_path)

    img_c_corrected = cca.automatic_color_equalization(
        img, slope=slope, limit=limit, samples=samples
    )
    img_blurred = gaussian_filter(img_c_corrected, sigma=(sigma, sigma, 0))

    _, g, _ = img_blurred[:, :, 0], img_blurred[:, :, 1], img_blurred[:, :, 2]
    bin_cyto_nuclei = g < threshold_otsu(g)

    bin_cyto_nuclei = morph_operations(
        bin_cyto_nuclei, mask_size=mask_size, iterations=iterations
    )
    cytoplasm = remove_small_regions(bin_cyto_nuclei, min_size=min_size)

    labeled_cytoplasm = label(cytoplasm)
    regions = regionprops(labeled_cytoplasm)

    if label_path:
        f1 = f1_score(ground_truth.flatten(), cytoplasm.flatten())
        return cytoplasm, f1, len(regions)

    return cytoplasm, None, len(regions)


if __name__ == "__main__":
    from optuna import load_study

    study = load_study(
        storage="sqlite:///cytoplasm_params.db", study_name="cytoplasm_segmentation"
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
    labels_dir = Path("labels/orig/all").glob("*.npy")

    results_df = pd.DataFrame(columns=["image", "f1", "n_objects"])

    for image in images_dir:
        label_path = Path("labels/orig/cytoplasm/") / f"{image.stem}.npy"
        segmented, f1, n_objects = cytoplasm_segment(
            img_path=image,
            mask_size=params["mask_size"],
            iterations=params["iterations"],
            min_size=params["min_size"],
            slope=params["slope"],
            limit=params["limit"],
            samples=params["samples"],
            sigma=params["sigma"],
            label_path=label_path,
        )

        print("Image: ", image.stem, "| F1: ", f1, "| N objects", n_objects)
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {"image": [image.stem], "f1": [f1], "n_objects": [n_objects]}
                ),
            ]
        )

        # save segmented image (png and npy)
        plt.imsave(
            Path("results_data/segmentation/cytoplasm/png")
            / f"{image.stem}_segmented.png",
            segmented,
        )
        np.save(
            Path("results_data/segmentation/cytoplasm/")
            / f"{image.stem}_segmented.npy",
            segmented,
        )

    results_df = results_df.sort_values(by="image")
    results_df.to_csv("results_data/segmentation/cytoplasm/results.csv", index=False)
    print("[green]Results saved.[/green]")
