import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tempfile


class MLflowTracker:
    """
    Classe personnalisée pour ton projet :
    - tracking SQLite local
    - logging params, metrics, artefacts
    - enregistrement du modèle sklearn dans MLflow Registry
    """

    def __init__(self, experiment_name="home_credit_experiment"):
        working_dir = Path(os.getcwd())
        db_path = (working_dir.parent / "mlruns" / "mlflow.db").as_posix()

        # Tracking URI SQLite
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(experiment_name)

        self.client = MlflowClient()
        print(f"[MLflow] Tracking sur: sqlite:///{db_path}")

    def start_run(self, run_name=None, tags=None):
        self.run = mlflow.start_run(run_name=run_name)

        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        return self.run.info.run_id

    def log_params(self, params: dict):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    def log_artifact_figure(self, fig, name="figure.png"):
        """Sauvegarde temporaire + upload artefact"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, name)
            fig.savefig(path, bbox_inches="tight")
            mlflow.log_artifact(path)

    def log_model(self, model, artifact_path="model"):
        mlflow.sklearn.log_model(model, artifact_path)

    def end_run(self):
        mlflow.end_run()

    def register_model(self, run_id, model_name="home_credit_rf", artifact_path="model"):
        """
        Enregistre le modèle du run dans le Model Registry
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(f"[MLflow] Enregistrement du modèle : {model_uri}")

        registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        return registered
