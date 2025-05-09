"""
Skript pro vygenerování binárních masek pro bounding boxy anotací.

- Vstupem jsou segmentované anotace z `input_directory`
- Výstupem jsou binární masky bounding boxů (AABB, OBB)
- Jsou odstraněny bounding boxy dotýkající se okraje.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from rich import print  # noqa: A004
from rich.progress import Progress
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

input_directory = "labels/watershed"
output_aabb_directory = "labels/filled_boxes/aabb"
output_obb_directory = "labels/filled_boxes/obb"
aabb_png_directory = "labels/filled_boxes/aabb/png"
obb_png_directory = "labels/filled_boxes/obb/png"

Path(output_aabb_directory).mkdir(parents=True, exist_ok=True)
Path(output_obb_directory).mkdir(parents=True, exist_ok=True)
Path(aabb_png_directory).mkdir(parents=True, exist_ok=True)
Path(obb_png_directory).mkdir(parents=True, exist_ok=True)

with Progress() as progress:
    npy_files = list(Path(input_directory).glob("*.npy"))
    task = progress.add_task("[cyan]Processing images...", total=len(npy_files))

    for label_file in npy_files:
        label_image = np.load(label_file)
        labeled = label(label_image)
        labeled = clear_border(labeled)

        aabb_output = np.zeros_like(label_image, dtype=np.uint8)
        obb_output = np.zeros_like(label_image, dtype=np.uint8)

        for region in regionprops(labeled):
            label_value = region.label
            coords = region.coords[:, ::-1]  # (x, y) format

            # AABB
            min_y, min_x = region.bbox[0], region.bbox[1]
            max_y, max_x = region.bbox[2], region.bbox[3]
            w, h = max_x - min_x, max_y - min_y
            aabb_output[min_y:max_y, min_x:max_x] = label_value

            # OBB
            rect = cv2.minAreaRect(coords.astype(np.float32))
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            cv2.fillPoly(obb_output, [box], int(label_value))

        np.save(
            Path(output_aabb_directory) / f"{label_file.stem}_aabb.npy", aabb_output
        )
        np.save(Path(output_obb_directory) / f"{label_file.stem}_obb.npy", obb_output)

        # AABB PNG
        fig, ax = plt.subplots()
        ax.set_xlim(0, label_image.shape[1])
        ax.set_ylim(label_image.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        for region in regionprops(labeled):
            min_y, min_x = region.bbox[0], region.bbox[1]
            max_y, max_x = region.bbox[2], region.bbox[3]
            w, h = max_x - min_x, max_y - min_y
            rect = Rectangle((min_x, min_y), w, h, edgecolor="none", facecolor="red")
            ax.add_patch(rect)
        plt.axis("off")
        plt.savefig(
            Path(aabb_png_directory) / f"{label_file.stem}_aabb.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0,
        )
        plt.close(fig)

        # OBB PNG
        fig, ax = plt.subplots()
        ax.set_xlim(0, label_image.shape[1])
        ax.set_ylim(label_image.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        for region in regionprops(labeled):
            coords = region.coords[:, ::-1]
            rect = cv2.minAreaRect(coords.astype(np.float32))
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            poly = Polygon(box, closed=True, edgecolor="none", facecolor="red")
            ax.add_patch(poly)
        plt.axis("off")
        plt.savefig(
            Path(obb_png_directory) / f"{label_file.stem}_obb.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0,
        )
        plt.close(fig)

        progress.update(task, advance=1)

print("[green]Processing complete.[/green]")
