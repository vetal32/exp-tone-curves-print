# exp-tone-curves-print

Code for the experiments on exponential tone curves for dark images in print: curve family, mean-matched comparison
with gamma, logarithmic, rational curves and CLAHE, and a simulated halftone print (8x8 clustered-dot screen, 15% dot gain,
Yule-Nielsen spread).

## Run

    pip install -r requirements.txt
    python run_experiments.py

Put the test photographs in `data/` (see `data/SOURCES.md`). Images with mean luma above 0.30 are skipped.
Results are written to `results/`: `metrics.csv` and four figures.

## Files

- `model.py`: curves, parameter matching, halftone and print model, metrics.
- `run_experiments.py`: runs all methods on every image in `data/`.
- `results/`: output of the run described in the paper (six images).

## Licence

Code: MIT. Photographs keep their own licences (`data/SOURCES.md`).
