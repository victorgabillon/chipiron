


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


## Docker

docker build -t chipiron-x11 .


sudo docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network=host \
  -u $(id -u):$(id -g) \
  chipiron-x11

