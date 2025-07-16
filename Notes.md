


## MLFlow
```
conda activate chipiron3.12
mlflow ui --backend-store-uri sqlite:///chipiron/scripts/default_output_folder/mlflow_data/mlruns.db
```


## Pipy
 - Clean previous builds:
rm -rf dist/ build/ *.egg-info

 - Build wheel and sdist:
python -m build

 - Upload to PyPI:
twine upload dist/*
