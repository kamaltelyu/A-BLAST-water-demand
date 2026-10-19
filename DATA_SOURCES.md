# Data sources used by the one-click rerun

## Chennai Water Management

Target construction uses the daily reservoir-level file with columns `Date`, `POONDI`, `CHOLAVARAM`, `REDHILLS`, and `CHEMBARAMBAKKAM`. The modeled target is their row-wise sum (`Total_Water`) in million cubic feet (mcft).

The one-click notebook downloads a public reproducibility mirror of the Chennai Water Management file:

`https://raw.githubusercontent.com/anubhav199/Water-Level-Prediction-in-Chennai/master/chennai_reservoir_levels.csv`

The source repository attributes the dataset to the Chennai Water Management dataset distributed via Kaggle. The exact downloaded file is hashed at runtime and recorded in `reproducibility/dataset_profiles.json`.

## Melbourne Water Supply

Target: `supplied_ML`, daily accumulated bulk water-supply volume in megalitres (ML).

Primary official source:

`https://data-melbournewater.opendata.arcgis.com/api/download/v1/items/ab54f1679c3d4dd6b0f4030634fce861/csv?layers=0`

If the direct CSV endpoint fails, the code falls back to the DataVic CKAN datastore resource `2f386033-f8c3-4e47-b206-29d433cc7e24` and reconstructs the same resource as CSV. The exact file used is hashed at runtime.

Raw data are not redistributed in the final results ZIP.
