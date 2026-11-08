# kSZCeell_sbi — verbatim extraction only

This directory is the requested folder structure with **existing source statements copied unchanged** from the two supplied notebooks. Code from both notebooks is preserved verbatim in `notebooks/`.

**Not yet a runnable package.** Cells in the old notebooks depended on notebook globals and execution order. Copying those statements to thematic `.py` files without changing code cannot create independent modules; imports, class/function encapsulation, configuration, and the notebook driver are intentionally NOT implemented. Importing a module may start long-running computation or fail because other notebook-defined names have not been initialised. **Do not import/run these modules as-is.** We can adapt them together in a later step.

The requested blank scaffolds are deliberately blank: `configs/baseline.yaml`, `src/ksz_sbi/pipeline.py`, `src/ksz_sbi/__init__.py`, `pyproject.toml`, and `notebooks/run_pipeline.ipynb` (valid notebook with zero cells). Output folders are empty except for placeholders to preserve directories in the zip.

## Location of original code

| File | Code currently placed there |
|---|---|
| `config.py` | Original notebook's imports and global numerical settings (not a new configuration implementation) |
| `theory.py` | Theory template loading, interpolation, noise and `make_theory_cls()` |
| `dataset.py` | Parameter draws, training and held-out theory banks, original flagged-row filtering, `a_lm` training selection |
| `simulator.py` | Original training Gaussian sky realisations, held-out `C_l` and map realisations, held-out `a_lm` generation |
| `features.py` | Measured `C_l` feature building, field flattening, training `a_lm` feature building |
| `models.py` | Existing priors, tensors and MAF/FCEmbedding construction for measured `C_l`, field and `a_lm` |
| `train.py` | Existing NPE fit and posterior-building statements for measured `C_l`, field and `a_lm` |
| `inference.py` | Existing loops for 1,000 held-out posterior medians/intervals for `C_l` and field inputs, plus `a_lm` medians |
| `pipeline.py` | Blank; no pipeline controller has been written yet |

Every copied section comes from a precise source-cell range shown in `CELL_PROVENANCE.csv`, including SHA-256 checksums. A few original notebook cells combine different responsibilities; these were split only at complete-line boundaries, without editing any line. The original notebook remains available for diagnostics, diagnostic figures, SBC, alternative experimental runs, and any sections omitted from the main/core modules. No mathematical expressions, seed values, paths, model parameters, statements or data transformations were corrected or rewritten.

## New notebook-facing settings and controller (7 November 2026)

- `src/ksz_sbi/settings.py` — NEW, standard-library-only validated `Settings` dataclass.
- `src/ksz_sbi/pipeline.py` — NEW, `Pipeline(input="cl", model="maf", ...)` interface, settings inspection, JSON saving and dry-run stage plan.
- `notebooks/run_pipeline.ipynb` — safe demonstration of the above interface.

The old `config.py` remains **unchanged**; its original imports and globals are not yet refactored.  All other original `.py` files and the two archived notebooks are also unchanged. `settings.py` does **not** retroactively control hard-coded paths, seeds or arguments in those files.  `Pipeline.run()` deliberately raises `NotImplementedError` until we convert the original top-level statements into import-safe functions and wire them in, one step at a time. This is not a functioning end-to-end SBI pipeline yet.

Use `Pipeline(input="cl")`, `Pipeline(input="map")` or `Pipeline(input="a_lm")`; the aliases `field` and `alm` work too. `embedding="auto"` records the existing C_l / map / a_lm embedding choice; these settings are not applied to the old modules yet.
