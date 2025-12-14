import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pathlib import Path
import os
import tempfile
from datetime import datetime


class MLflowTracker:
    """
    Classe personnalisée pour ton projet :
    - tracking SQLite local
    - logging params, metrics, artefacts
    - enregistrement du modèle sklearn dans MLflow Registry

    Ajouts : possibilité de préciser `artifact_root` (dossier où seront stockés
    les artefacts/modèles) et `db_path` (fichier sqlite pour le backend store).
    """

    def __init__(self, experiment_name="home_credit_experiment", artifact_root=None, db_path=None, force_new_experiment=True):
        working_dir = Path(os.getcwd())
        project_root = working_dir.parent

        # Par défaut, on positionne la DB et les artefacts dans <project_root>/mlruns
        if db_path is None:
            db_path = (project_root / "mlruns" / "mlflow.db").as_posix()

        # Configure le backend store (SQLite)
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        # Prépare le chemin d'artefacts (emplacement pour log_model et log_artifact)
        if artifact_root is None:
            artifact_root_path = (project_root / "model").resolve()
        else:
            artifact_root_path = Path(artifact_root).expanduser().resolve()

        # Crée le dossier d'artefacts si nécessaire
        artifact_root_path.mkdir(parents=True, exist_ok=True)

        # URI d'artefact au format file:// pour MLflow
        artifact_uri = artifact_root_path.as_uri()

        # Initialise le client MLflow (après avoir set_tracking_uri)
        self.client = MlflowClient()

        # Conserver les chemins pour debug
        self.artifact_root_path = artifact_root_path
        self.artifact_uri = artifact_uri
        self.tracking_db = db_path

        # Crée l'expérience si elle n'existe pas en précisant l'artifact_location
        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
            self.experiment_id = exp_id
            print(f"[MLflow] Création de l'expérience '{experiment_name}' avec artifact_location={artifact_uri}")
        else:
            # Si l'expérience existe mais que l'artifact_location diffère, on peut créer
            # automatiquement une nouvelle expérience pour s'assurer que les artefacts
            # sont écrits dans le dossier demandé.
            if hasattr(exp, 'artifact_location') and exp.artifact_location != artifact_uri:
                if force_new_experiment:
                    new_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    new_id = mlflow.create_experiment(new_name, artifact_location=artifact_uri)
                    mlflow.set_experiment(new_name)
                    self.experiment_name = new_name
                    self.experiment_id = new_id
                    print(f"[MLflow] L'expérience existante '{experiment_name}' a un artifact_location différent.\n"
                          f"Création et basculement vers la nouvelle expérience '{new_name}' avec artifact_location={artifact_uri}.")
                else:
                    mlflow.set_experiment(experiment_name)
                    self.experiment_name = experiment_name
                    self.experiment_id = exp.experiment_id
                    print(f"[MLflow] Attention : l'expérience '{experiment_name}' existe déjà avec artifact_location={exp.artifact_location}.\n"
                          f"Le chemin demandé pour les artefacts est {artifact_uri} mais MLflow n'écrira pas dans ce nouvel emplacement pour une expérience existante.")
            else:
                mlflow.set_experiment(experiment_name)
                self.experiment_name = experiment_name
                self.experiment_id = exp.experiment_id

        print(f"[MLflow] Tracking sur: sqlite:///{db_path} ; artefacts -> {artifact_uri}")

    def start_run(self, run_name=None, tags=None):
        # Assure que l'expérience est bien définie
        if hasattr(self, 'experiment_name'):
            mlflow.set_experiment(self.experiment_name)

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
        # Enregistre le modèle dans le run courant (sous self.artifact_uri/<run_id>/<artifact_path>)
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

    # Méthodes utilitaires de debug
    def get_current_experiment(self):
        """Retourne un dict {name, id, artifact_location} de l'expérience courante."""
        try:
            exp = self.client.get_experiment_by_name(self.experiment_name)
            if exp is None:
                return None
            return {"name": exp.name, "id": exp.experiment_id, "artifact_location": exp.artifact_location}
        except Exception:
            return None

    def run_artifact_uri(self, run_id):
        """Retourne l'artifact_uri d'un run donné (utile pour debug)."""
        try:
            run = self.client.get_run(run_id)
            return run.info.artifact_uri
        except Exception as e:
            return str(e)

