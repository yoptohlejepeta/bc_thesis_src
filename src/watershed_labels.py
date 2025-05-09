"""
Skript pro segmentaci anotací pomocí metody Watershed.

Je dále použito pro vytváření bounding boxů.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich import print  # noqa: A004
from rich.progress import Progress
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed

nucleus_dir = Path("labels/orig/nucleus")
cytoplasm_dir = Path("labels/orig/cytoplasm")
output_dir = Path("labels/watershed")
output_dir.mkdir(exist_ok=True)
outping_dir_png = output_dir / "png"
outping_dir_png.mkdir(exist_ok=True)

nucleus_files = sorted(nucleus_dir.glob("*.npy"))

with Progress() as progress:
    task = progress.add_task("[cyan]Processing images...", total=len(nucleus_files))

    for nucleus_file in nucleus_files:
        pic_num = nucleus_file.stem

        nucleus = np.load(nucleus_file)
        cytoplasm_file = cytoplasm_dir / f"{pic_num}.npy"

        if not cytoplasm_file.exists():
            print(f"Cytoplasm file not found for {pic_num}, skipping...")
            continue

        cytoplasm = np.load(cytoplasm_file)

        nucleus = label(nucleus)

        cytoplasm_no_nucleus = cytoplasm.copy()
        cytoplasm_no_nucleus[nucleus > 0] = 0

        distance = distance_transform_edt(cytoplasm_no_nucleus)

        watershed_img = watershed(-distance, markers=nucleus, mask=cytoplasm)

        output_path = output_dir / f"{pic_num}_watershed.npy"
        np.save(output_path, watershed_img)

        plt.imsave(
            output_dir / f"png/{pic_num}_watershed.png",
            watershed_img,
            cmap="nipy_spectral",
        )
        progress.update(task, advance=1)


print("[green]Processing complete.[/green]")
