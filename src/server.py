from flask import Flask, request, jsonify
import os
import pickle
import threading
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Chemin par défaut vers le modèle picklé (modifiable via MODEL_PATH)
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    Path(os.path.dirname(__file__)).parent / "model" / "models"
)

# On accepte aussi MODEL_FILE env var pour pointer directement sur le .pkl
MODEL_FILE = os.environ.get("MODEL_FILE")

_model = None
_model_lock = threading.Lock()
_model_path_resolved = None


def _resolve_model_path():
    """Résout le chemin complet du fichier modèle à charger.
    Priorité : MODEL_FILE env -> search for model.pkl under DEFAULT_MODEL_PATH -> error
    """
    global _model_path_resolved
    if _model_path_resolved:
        return _model_path_resolved

    if MODEL_FILE:
        p = Path(MODEL_FILE)
        if p.exists():
            _model_path_resolved = str(p)
            return _model_path_resolved
        else:
            raise FileNotFoundError(f"MODEL_FILE {MODEL_FILE} introuvable")

    base = Path(DEFAULT_MODEL_PATH)
    # si DEFAULT_MODEL_PATH est un répertoire contenant des sous-dossiers de modèles, on cherche un fichier .pkl
    if base.is_dir():
        # recherche récursive pour le premier fichier .pkl
        pkl_files = list(base.rglob("*.pkl"))
        if len(pkl_files) > 0:
            _model_path_resolved = str(pkl_files[0])
            return _model_path_resolved
    # sinon, si DEFAULT_MODEL_PATH lui-même est un fichier
    if base.exists() and base.suffix in ['.pkl', '.joblib']:
        _model_path_resolved = str(base)
        return _model_path_resolved

    raise FileNotFoundError(f"Aucun fichier modèle trouvé sous {base}. Défini MODEL_FILE ou MODEL_PATH pour pointer sur le modèle.")


def load_model():
    """Charge le modèle en mémoire (singleton thread-safe)."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        model_path = _resolve_model_path()
        # Try joblib then pickle
        try:
            from joblib import load as joblib_load
            _model = joblib_load(model_path)
        except Exception:
            with open(model_path, "rb") as f:
                _model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return _model


@app.route("/", methods=["GET"])
def index():
    return (
        "Bienvenue sur l'API d'inférence du modèle de classification de risque de crédit. "
        "Utilisez /ping pour le health-check et /invocations pour les inférences.",
        200,
    )

@app.route("/ping", methods=["GET"])
def ping():
    """Endpoint de health-check."""
    try:
        _ = load_model()
        return ("pong", 200)
    except Exception as e:
        return (str(e), 500)


@app.route("/invocations", methods=["POST"])
def invocations():
    """Endpoint d'inférence. Accepte JSON ou CSV.

    JSON formats accepted:
    - {"instances": [ {feature: value, ...}, ... ]}
    - list of dicts: [ {feature: value, ...}, ... ]
    - {"data": [[v1, v2,...], ...], "columns": ["c1","c2",...]}

    CSV: body contains CSV, parsed by pandas.read_csv
    """
    try:
        content_type = request.content_type or ""
        # parse input into a DataFrame
        if "json" in content_type:
            payload = request.get_json(force=True)
            if isinstance(payload, dict) and "instances" in payload:
                df = pd.DataFrame(payload["instances"])
            elif isinstance(payload, list):
                df = pd.DataFrame(payload)
            elif isinstance(payload, dict) and "data" in payload and "columns" in payload:
                df = pd.DataFrame(payload["data"], columns=payload["columns"])
            else:
                # fallback: try to interpret payload as a single row dict
                df = pd.DataFrame([payload])
        elif "csv" in content_type or request.data.startswith(b"SK_ID") or b"," in request.data[:100]:
            # attempt to read CSV from raw body
            from io import BytesIO
            body = request.data
            df = pd.read_csv(BytesIO(body))
        else:
            # fallback to JSON parsing
            payload = request.get_json(force=True)
            df = pd.DataFrame(payload)

    except Exception as e:
        return jsonify({"error": f"Impossible de parser l'entrée: {e}"}), 400

    # Ensure df is not empty
    if df is None or len(df) == 0:
        return jsonify({"error": "Aucune donnée fournie pour l'inférence"}), 400

    # Load model
    try:
        model = load_model()
    except Exception as e:
        return jsonify({"error": f"Impossible de charger le modèle: {e}"}), 500

    # Run prediction
    try:
        # If model expects a numpy array, pass df.values
        # Prefer predict_proba when available
        result = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            preds = model.predict(df)
            result["predictions"] = np.asarray(preds).tolist()
            # include probabilities for each class
            result["probabilities"] = np.asarray(proba).tolist()
        else:
            preds = model.predict(df)
            result["predictions"] = np.asarray(preds).tolist()

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {e}"}), 500


if __name__ == "__main__":
    # Pour exécuter en local : python src/api/server.py
    # On écoute sur toutes les interfaces pour faciliter les tests Docker / WSL
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

