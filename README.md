Analysis of seismicity resulting from deglaciation.

**To reproduce the figures in the paper, run `gr_gia.py`.** The notebook `gr_gia.ipynb` is exploratory and not the official version.

## Installation

Requires [conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
git clone https://github.com/keliankaz/GrGIA_seismicity.git
cd GrGIA_seismicity
conda env create -f environment.yml
conda activate grGIA_seismicity_test
python gr_gia.py
```

Figures are saved to `figures/`.

## Repository structure

| Path | Description |
|------|-------------|
| `gr_gia.py` | Main script — reproduces all paper figures |
| `stat_utils.py` | Statistics: b-value, b-positive, bootstrapping |
| `geo_utils.py` | Geospatial helpers |
| `schuster.py` | Schuster spectrum (periodicity test) |
| `decluster.py` | Earthquake declustering |
| `utils.py` | Catalog loaders |
| `data/` | Earthquake catalogs, shapefiles, GIA strain tensors |
| `figures/` | Output figures (not tracked in git) |
| `environment.yml` | Conda environment |

## Plate boundary data

From Bird et al. (2004). Shapefiles are in `data/plate_boundaries/`.


