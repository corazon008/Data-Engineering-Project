# Projet Final : Data Engineering & Credit Scoring


## Contexte
Ce projet s'inscrit au sein d'une société financière spécialisée dans le crédit à la consommation pour des clients ayant peu ou pas d'historique de prêt. L'objectif est de développer un outil de Credit Scoring capable de prédire la probabilité de remboursement d'un client et de classer automatiquement les demandes.
Contrainte Économique et Fonction de CoûtLa modélisation intègre une contrainte métier majeure : 
l'asymétrie du coût de l'erreur.\
Faux Négatif (FN) : Accorder un crédit à un client qui ne rembourse pas (Perte de capital).\
Faux Positif (FP) : Refuser un crédit à un client fiable (Manque à gagner).

D'un point de vue business, la perte liée à un défaut de paiement est jugée beaucoup plus critique. Le modèle est donc optimisé pour minimiser une fonction de coût personnalisée où le Faux Négatif est pénalisé 10 fois plus que le Faux Positif:
$$\text{Coût} = 10 \times \text{FN} + 1 \times \text{FP}$$
En conséquence, le seuil de décision (threshold) d'octroi de crédit n'est pas fixé par défaut à 0.5, mais optimisé pour minimiser ce coût métier spécifique.


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

## Seuil métier
Pour répondre à la contrainte d'asymétrie des coûts, le seuil de décision n'est pas fixé arbitrairement à 0.5. Il est déterminé par l'optimisation de la fonction de coût suivante :
$$\text{Coût} = 10 \times \text{FN} + 1 \times \text{FP}$$
L'analyse a permis d'identifier le seuil optimal qui minimise ce risque financier.\
Règle de décision :\
Si la probabilité de défaut est < Seuil : Crédit Accordé (Client sain).\
Si la probabilité de défaut est ≥ Seuil : Crédit Refusé (Client à risque).

## Structure
```
├── docker-compose.yml
├── Dockerfile
├── home-credit-default-risk
│   ├── application_test.csv
│   ├── application_train.csv
│   ├── bureau_balance.csv
│   ├── bureau.csv
│   ├── credit_card_balance.csv
│   ├── HomeCredit_columns_description.csv
│   ├── home_credit.png
│   ├── installments_payments.csv
│   ├── POS_CASH_balance.csv
│   ├── previous_application.csv
│   └── sample_submission.csv
├── mlruns
│   └── mlflow.db
├── model
│   ├── application_test_enriched.csv
│   ├── application_train_enriched.csv
│   └── submission.csv
├── notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_explainability.ipynb
│   ├── 04_mlflow_serving_test.ipynb
├── pyproject.toml
├── README.md
├── reports
│   └── figures
│       ├── postprocessing
│       │   ├── application_test_null_value_chart.png
│       │   ├── application_train_null_value_chart.png
│       │   ├── bureau_null_value_chart.png
│       │   ├── credit_card_balance_null_value_chart.png
│       │   ├── installments_payments_null_value_chart.png
│       │   ├── POS_CASH_balance_null_value_chart.png
│       │   └── previous_application_null_value_chart.png
│       └── preprocessing
│           ├── application_test_null_value_chart.png
│           ├── application_train_null_value_chart.png
│           ├── bureau_null_value_chart.png
│           ├── credit_card_balance_null_value_chart.png
│           ├── installments_payments_null_value_chart.png
│           ├── POS_CASH_balance_null_value_chart.png
│           └── previous_application_null_value_chart.png
├── src
│   ├── data_prep.py
│   ├── explainability.py
│   ├── metrics.py
│   ├── mlFlow
│   │   ├── MLflowTracker.py
│   ├── model_utils.py
│   └── server.py
├── Tasks.md
├── tests
│   ├── test_api.py
└── uv.lock
```