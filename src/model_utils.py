import mlflow
import mlflow.sklearn
from datetime import datetime
import os
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from mlFlow.MLflowTracker import MLflowTracker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_training_with_mlflow(grid, pipeline,
                             X_train, y_train, X_test, y_test,
                             tracker: MLflowTracker,
                             run_name="rf_gridsearch",
                             model_name="home_credit_rf"):
    try:
        # --- MLflow START ---
        run_id = tracker.start_run(run_name, tags={"type": "gridsearch", "model": model_name,})

        # Fit gridsearch
        grid.fit(X_train, y_train)

        # Meilleur modèle
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Scores
        metrics = {
            "train_accuracy": accuracy_score(y_train, best_model.predict(X_train)),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "train_f1": f1_score(y_train, best_model.predict(X_train)),
            "test_f1": f1_score(y_test, y_pred),
            #"test_classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        tracker.log_metrics(metrics)

        # Log best params
        tracker.log_params(grid.best_params_)

        # --- Confusion matrix ---
        #cm = confusion_matrix(y_test, y_pred)
        #fig = plt.figure(figsize=(6, 4))
        #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        #plt.title("Matrice de confusion")
        #tracker.log_artifact_figure(fig, "confusion_matrix.png")
    #
        ## --- Feature importance ---
        #model = best_model.named_steps["model"]
        #preprocessor: ColumnTransformer = best_model.named_steps["preprocessing"]
        #feature_names = preprocessor.get_feature_names_out()
        #clean_names = [name.split("__")[-1] for name in feature_names]
    #
        #importances = pd.Series(model.feature_importances_, index=clean_names)
        #fig2 = importances.sort_values(ascending=False).head(25).plot(kind="barh", figsize=(12, 8)).get_figure()
        #plt.title("Top 25 Features importantes")
        #tracker.log_artifact_figure(fig2, "feature_importances.png")

        # --- Log modèle ---
        tracker.log_model(best_model, artifact_path="model")

        # --- Register Model ---
        registered = tracker.register_model(run_id, model_name=model_name)

        print("\n=== Résultats ===")
        print("Best params:", grid.best_params_)
        print(metrics)
        print("Registered version:", registered.version)

        return best_model, metrics
    except Exception as e:
        print("Erreur durant le training avec MLflow:", e)
        return None, None
    finally:
        tracker.end_run()


def get_pipeline(model: BaseEstimator, numeric_features, categorical_features) -> Pipeline:
    """
    Construit et retourne un pipeline scikit-learn avec préprocessing.
    """

    transformers = [("num", SimpleImputer(strategy="mean"), StandardScaler(), numeric_features),
        ("cat", SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"), categorical_features)]

    preprocessor = ColumnTransformer(transformers)
    pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])
    return pipeline
