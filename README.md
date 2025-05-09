# Vytváření bounding boxů ve snímcích buněk pořízených optickým mikroskopem

> BAKALÁŘSKÁ PRÁCE

## Struktura projektu

### Data

Obrázky, které chceme segmentovat se musí nacházet ve složce `images/`. Anotace se pak nachází ve složce `labels/orig/`, kde jsou pak dále děleny do `all/` (anotace označující cytoplazmy a jádra) `cytoplasm/` a `nucleus/` (anotace obsahující pouze cytoplazmy resp. jádra).

### Skripty

Skripty se nachází ve složce `src/`.

- `cytoplasm_segmentation.py` - Skript pro segmentaci cytoplazmy buněk.
- `nucleus_segmentation.py` - Skript pro segmentaci jader buněk.
- `watershed.py` - Skript pro segmentaci buněk pomocí metody Watershed (používá masky cytoplazem a jader).
- `nucleus_nonopt.py` a `cytoplasm_nonopt.py` - Skripty pro segmentaci. Nepoužívají optimalizované parametry.
- `gen_images.py` - Skript pro generování obrázků jednotlivých buněk.

Nachází se zde ještě podsložky `optuna/` a `preps/`. Složka `optuna/` obsahuje skripty pro optimalizaci parametrů `opptuna_cytoplasm.py` a `optuna_nucleus.py`. Složka `preps/` obsahuje skripty pro generování binárních masek bounding boxů. `filled_boxes_labels.py` pro anotace a `filled_boxes.py` pro segmentované snímky.

### Notebooky

Složka `notebooks/` obsahuje Jupyter notebooky.
Jupyter notebooky obsahují analýzy a grafy použité v textu práce.

- `compare_algorithms.ipynb` - Porovnání algoritmu s SVM pro jádra a cytoplazmu.
- `compare_boxes.ipynb` - Analýza segmentace z pohledu počtu objektů a porovnání metrik dvou typů bounding boxů.
- `cytoplasm_segmentation.ipynb` a `nucleus_segmentation.ipynb` - Obsahují segmentační posstupy krok po kroku, včetně analýzy důležitosti parametrů.
- `watershed.ipynb` - Obsahuje spojení masek jader a cytoplazmy a segmentaci jednotlivých buněk pomocí metody Watershed.

## Použití

### Instalace knihoven

Pro použití skriptů je potřeba mít nainstalované knihovny.

```bash
uv sync
```

### Spuštění skriptů

Jednotlivé skripty se zpouští pomocí:

```bash
uv run python -m src.<název_skriptu>
```
Například pro spuštění skriptu `cytoplasm_segmentation.py` použijete:

```bash
uv run python -m src.cytoplasm_segmentation
```