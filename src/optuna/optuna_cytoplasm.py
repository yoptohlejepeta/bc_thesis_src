from pathlib import Path

import numpy as np

from src.cytoplasm_segmentation import cytoplasm_segment


def objective(trial):
    imgs_dir = Path("images/")

    sigma = trial.suggest_float("sigma", 1.0, 15.0)
    mask_size = trial.suggest_int("mask_size", 2, 5)
    iterations = trial.suggest_int("iterations", 1, 10)
    min_size = trial.suggest_int("min_size", 50, 5000)
    slope = trial.suggest_int("slope", 5, 50)
    limit = trial.suggest_int("limit", 500, 5000)
    samples = trial.suggest_int("samples", 100, 3000)

    scores = []

    for img_path in imgs_dir.glob("*.png"):
        _, f1 = cytoplasm_segment(
            img_path,
            mask_size=mask_size,
            iterations=iterations,
            min_size=min_size,
            slope=slope,
            limit=limit,
            samples=samples,
            sigma=sigma,
            label_path="labels/orig/cytoplasm/" + img_path.stem + ".npy",
        )

        scores.append(f1)

    return np.mean(scores)


if __name__ == "__main__":
    from optuna import create_study

    study = create_study(
        storage="sqlite:///cytoplasm_params.db",
        load_if_exists=True,
        study_name="cytoplasm_segmentation",
        direction="maximize",
    )
    study.optimize(objective, n_trials=1000)
