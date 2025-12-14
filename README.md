# Projet Final : Data Engineering & Credit Scoring

## Requirements
```
uv sync
uv pip install -e .
```

## Run mlflow server
```commandline
uv run mlflow server --port 8080 --backend-store-uri sqlite:///mlruns/mlflow.db
```


## Run the server
```commandline
docker compose up -d
```