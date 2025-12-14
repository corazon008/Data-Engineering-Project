# test/test_api.py

import io
import json
import pandas as pd
import pytest

# ...existing code...

def test_invocations_with_csv_excerpt():
    """Charge quelques lignes de `model/application_test_enriched.csv` et les envoie
    à l'endpoint `/invocations` du Flask app (utilise le test_client).
    Vérifie que la réponse HTTP est 200 et contient des prédictions JSON.
    """
    from src import server

    # Chemin vers le fichier de test enrichi fourni dans le repo
    csv_path = "model/application_test_enriched.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        pytest.skip(f"Fichier de test introuvable: {csv_path}")

    # Prendre un petit extrait (3 lignes) et s'assurer qu'il y a des colonnes
    if df.shape[0] == 0:
        pytest.skip("Le fichier CSV de test est vide")

    excerpt = df.head(3)

    # Convertir en CSV en mémoire
    csv_buffer = io.StringIO()
    excerpt.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode("utf-8")

    client = server.app.test_client()

    resp = client.post(
        "/invocations",
        data=csv_data,
        content_type="text/csv",
    )

    assert resp.status_code == 200, f"Status code inattendu: {resp.status_code}, body: {resp.data}"

    # tenter de parser le JSON renvoyé
    try:
        data = resp.get_json()
    except Exception as e:
        pytest.fail(f"La réponse n'est pas un JSON valide: {e}")

    assert isinstance(data, dict), "La réponse JSON doit être un objet/dictionnaire"
    assert "predictions" in data, "La réponse JSON doit contenir la clé 'predictions'"
    preds = data["predictions"]
    assert isinstance(preds, list), "'predictions' doit être une liste"
    assert len(preds) == excerpt.shape[0], "Le nombre de prédictions doit correspondre au nombre de lignes envoyées"

    # si probabilités renvoyées, vérifier leur taille
    if "probabilities" in data:
        probs = data["probabilities"]
        assert isinstance(probs, list)
        assert len(probs) == excerpt.shape[0]


if __name__ == "__main__":
    test_invocations_with_csv_excerpt()