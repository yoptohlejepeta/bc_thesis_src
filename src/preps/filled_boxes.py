"""
Skript pro vygenerování binárních masek pro bounding boxy segmentovaných snímků.

- Vstupem jsou segmentované snímky z `input_directory`
- Výstupem jsou binární masky bounding boxů (AABB, OBB)
- Jsou odstraněny bounding boxy dotýkající se okraje.
- Bounding boxy jsou zvětšeny o `offset_percent`
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

input_directory = "results_data/watershed"
output_aabb_directory = "results_data/filled_boxes/aabb"
output_obb_directory = "results_data/filled_boxes/obb"
aabb_png_directory = "results_data/filled_boxes/aabb/png"
obb_png_directory = "results_data/filled_boxes/obb/png"

Path(output_aabb_directory).mkdir(parents=True, exist_ok=True)
Path(output_obb_directory).mkdir(parents=True, exist_ok=True)
Path(aabb_png_directory).mkdir(parents=True, exist_ok=True)
Path(obb_png_directory).mkdir(parents=True, exist_ok=True)

offset_percent = 5

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
            coords = region.coords[:, ::-1]

            min_y, min_x = region.bbox[0], region.bbox[1]
            max_y, max_x = region.bbox[2], region.bbox[3]
            w, h = max_x - min_x, max_y - min_y
            offset_x = int(w * offset_percent / 100)
            offset_y = int(h * offset_percent / 100)
            min_x = max(0, min_x - offset_x)
            min_y = max(0, min_y - offset_y)
            max_x = min(label_image.shape[1], max_x + offset_x)
            max_y = min(label_image.shape[0], max_y + offset_y)
            w = max_x - min_x
            h = max_y - min_y
            aabb_output[min_y:max_y, min_x:max_x] = label_value

            rect = cv2.minAreaRect(coords.astype(np.float32))
            offset_w = rect[1][0] * offset_percent / 100
            offset_h = rect[1][1] * offset_percent / 100
            rect = (
                rect[0],
                (rect[1][0] + 2 * offset_w, rect[1][1] + 2 * offset_h),
                rect[2],
            )
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            cv2.fillPoly(obb_output, [box], int(label_value))

        np.save(
            Path(output_aabb_directory) / f"{label_file.stem}_aabb.npy", aabb_output
        )
        np.save(Path(output_obb_directory) / f"{label_file.stem}_obb.npy", obb_output)

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
            offset_x = int(w * offset_percent / 100)
            offset_y = int(h * offset_percent / 100)
            min_x = max(0, min_x - offset_x)
            min_y = max(0, min_y - offset_y)
            max_x = min(label_image.shape[1], max_x + offset_x)
            max_y = min(label_image.shape[0], max_y + offset_y)
            w = max_x - min_x
            h = max_y - min_y
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

        fig, ax = plt.subplots()
        ax.set_xlim(0, label_image.shape[1])
        ax.set_ylim(label_image.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        for region in regionprops(labeled):
            coords = region.coords[:, ::-1]
            rect = cv2.minAreaRect(coords.astype(np.float32))
            offset_w = rect[1][0] * offset_percent / 100
            offset_h = rect[1][1] * offset_percent / 100
            rect = (
                rect[0],
                (rect[1][0] + 2 * offset_w, rect[1][1] + 2 * offset_h),
                rect[2],
            )
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
