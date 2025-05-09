"""
Skript pro segmentaci jednotlivých buněk.

- Spojení masky jader (použita jako markery) a cytoplazmy
- Segmentace pomocí algoritmu Markerbased Watershed
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed

nucleus_dir = Path("results_data/segmentation/nucleis")
cytoplasm_dir = Path("results_data/segmentation/cytoplasm")
output_dir = Path("results_data/watershed")
output_dir.mkdir(exist_ok=True)

nucleus_files = sorted(nucleus_dir.glob("*.npy"))

for nucleus_file in nucleus_files:
    pic_num = nucleus_file.stem
    print(f"Processing image: {pic_num}")

    # Load nucleus and cytoplasm
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


print("Processing complete.")
