


## MLFlow
```
conda activate chipiron3.12
mlflow ui --backend-store-uri sqlite:///chipiron/scripts/default_output_folder/mlflow_data/mlruns.db
```

## Fask app
```
conda activate chipiron3.12
python flaskapp/flask_app.py
```

## Gcloud
🔍 2. Find the deployed service & URL
 - Open the GCP console: https://console.cloud.google.com/run
 - Look for a service name in the region you deployed to.
 - Click the service — the URL will be shown at the top.

In case login not working:
gcloud auth login

Finding services : gcloud run services list

Upload: gcloud run deploy   --allow-unauthenticated --source .

link : https://console.cloud.google.com/run/detail/europe-west2/chipiron/metrics?hl=fr&inv=1&invt=Ab1U8A&project=chipironchess


## Pipy

 - Clean previous builds:
rm -rf dist/ build/ *.egg-info

 - Build wheel and sdist:
python -m build

 - Upload to PyPI:
twine upload dist/*