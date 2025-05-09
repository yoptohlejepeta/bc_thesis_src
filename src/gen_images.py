"""
Skript pro generování obrázků z datových sad.

Tento skript načte data z JSON souborů, které obsahují informace o obdélnících a obdélnících s orientací (AABB a OBB).
Následně provede oříznutí obrázků podle těchto informací a uloží oříznuté obrázky do specifikovaných adresářů.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2

aabb_json_path = Path("results_data/watershed_with_bbox/all_aabb.json")
obb_json_path = Path("results_data/watershed_with_obbs/all_obb.json")
images_dir = Path("images")
crops_base_dir = Path("results_data/bbox_crops")

crops_base_dir.mkdir(parents=True, exist_ok=True)

with open(aabb_json_path, "r") as f:
    aabb_data = json.load(f)

with open(obb_json_path, "r") as f:
    obb_data = json.load(f)

for pic_num in aabb_data:
    image_path = images_dir / f"{pic_num}.png"
    if not image_path.exists():
        print(f"Original image {image_path} not found. Skipping...")
        continue

    image = Image.open(image_path)
    image_np = np.array(image)
    img_height, img_width = image_np.shape[:2]

    aabb_crops_dir = crops_base_dir / pic_num / "aabb"
    obb_crops_dir = crops_base_dir / pic_num / "obb"
    aabb_crops_dir.mkdir(parents=True, exist_ok=True)
    obb_crops_dir.mkdir(parents=True, exist_ok=True)

    for idx, aabb in enumerate(aabb_data[pic_num]):
        minc, minr, maxc, maxr = map(int, aabb)
        minc = max(0, minc)
        minr = max(0, minr)
        maxc = min(img_width, maxc)
        maxr = min(img_height, maxr)

        if maxc <= minc or maxr <= minr:
            print(f"Invalid AABB for {pic_num} index {idx}. Skipping...")
            continue

        crop = image.crop((minc, minr, maxc, maxr))
        crop_path = aabb_crops_dir / f"{pic_num}_aabb_{idx}.png"
        crop.save(crop_path)
        print(f"Saved AABB crop: {crop_path}")

    for idx, obb in enumerate(obb_data.get(pic_num, [])):
        obb_points = np.array(obb, dtype=np.int32)

        minc = max(0, obb_points[:, 0].min())
        maxc = min(img_width, obb_points[:, 0].max())
        minr = max(0, obb_points[:, 1].min())
        maxr = min(img_height, obb_points[:, 1].max())

        if maxc <= minc or maxr <= minr:
            print(f"Invalid OBB for {pic_num} index {idx}. Skipping...")
            continue

        padding = max(maxc - minc, maxr - minr) // 2
        minc = max(0, minc - padding)
        minr = max(0, minr - padding)
        maxc = min(img_width, maxc + padding)
        maxr = min(img_height, maxr + padding)

        crop = image.crop((minc, minr, maxc, maxr))
        crop_np = np.array(crop)

        if crop_np.ndim == 2:
            crop_np = np.stack([crop_np] * 3, axis=-1)

        adjusted_points = obb_points - np.array([minc, minr])
        rect = cv2.minAreaRect(adjusted_points)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle = 90 + angle
        else:
            angle = angle

        crop_height, crop_width = crop_np.shape[:2]
        center = (crop_width // 2, crop_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_crop = cv2.warpAffine(
            crop_np,
            rotation_matrix,
            (crop_width, crop_height),
            flags=cv2.INTER_LANCZOS4,
        )

        mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
        cv2.fillPoly(mask, [adjusted_points], 255)
        rotated_mask = cv2.warpAffine(
            mask, rotation_matrix, (crop_width, crop_height), flags=cv2.INTER_NEAREST
        )
        rotated_mask = (rotated_mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print(f"No valid contour for OBB {pic_num} index {idx}. Skipping...")
            continue

        x, y, w, h = cv2.boundingRect(contours[0])
        final_crop = rotated_crop[y : y + h, x : x + w]

        if final_crop.size == 0:
            print(f"Empty crop for OBB {pic_num} index {idx}. Skipping...")
            continue

        crop = Image.fromarray(final_crop)
        crop_path = obb_crops_dir / f"{pic_num}_obb_{idx}.png"
        crop.save(crop_path)
        print(f"Saved OBB crop: {crop_path}")

print("Cropping complete.")
