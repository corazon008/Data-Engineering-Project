# Étape 1 : Préparer, nettoyer et enrichir les données
Vous allez découvrir et préparer les données nécessaires à la construction de votre modèle de scoring.
Cela inclut le nettoyage, la fusion des différentes sources, l’encodage des variables et la création de nouvelles features pertinentes.
L’objectif est de constituer un dataset propre et enrichi, prêt pour l’entraînement. Vous devrez également analyser la qualité de vos variables et les déséquilibres dans les classes.

## Prérequis
- Avoir exploré les données brutes fournies.
- Avoir vérifié les formats et les valeurs manquantes.
- Avoir identifié les colonnes clés pour les jointures.
- Avoir pris en compte les enjeux métiers (par exemple : déséquilibre des classes).

## Résultat attendu
Un jeu de données propre, fusionné et enrichi, prêt à être utilisé pour l’entraînement.

## Recommandations
- Charger chaque fichier séparément et inspecter ses colonnes.
- Utiliser pandas pour fusionner les jeux de données.
- Visualiser la distribution des classes cibles.
- Créer de nouvelles features à partir des variables existantes si nécessaire.
- Éviter de supprimer trop rapidement les colonnes avec des valeurs manquantes : explorer les possibilités d’imputation.

## Points de vigilance
- Oublier de vérifier les doublons.
- Supprimer des colonnes sans analyser leur importance métier.
- Imputer sans documenter ni justifier.
- Fusionner sans gérer les duplications ou pertes de lignes.
- Encoder sans tenir compte du type de modèle prévu (ordinal vs nominal).

## Outils
- pandas
- matplotlib et seaborn pour la visualisation
- scikit-learn pour le preprocessing
- missingno pour visualiser les valeurs manquantes

# Étape 2 : Traquer les expérimentations avec MLflow
Vous allez tracer vos expériences de modélisation avec MLflow : métriques, hyperparamètres, versions de modèles, etc.
Vous utiliserez l’interface web pour visualiser vos runs et comparer les modèles.

## Prérequis
- Avoir installé MLflow.
- Avoir configuré votre projet localement.

## Résultat attendu
Des runs visibles dans l’UI MLflow avec les paramètres testés et les scores obtenus.

## Recommandations
- Commencer par intégrer mlflow.start_run() dans vos notebooks.
- Logger les métriques et les paramètres principaux.
- Utiliser mlflow.autolog() si vous utilisez des modèles compatibles.
- Activer l’interface UI avec mlflow ui pour visualiser les résultats.

## Points de vigilance
- Lancer MLflow sans environnement isolé peut créer des conflits de versions. Utiliser un environnement virtuel.
- Oublier d’annoter les expériences (tags, noms, commentaires) complique l’analyse dans l’interface MLflow.
- Ne pas versionner les modèles enregistrés empêche de reproduire les résultats et de gérer leur cycle de vie.
- Sauvegarder des fichiers inutiles ou trop volumineux dans MLflow encombre le système et ralentit l’interface.

## Outils
- MLflow

# Étape 3 : Modéliser et expérimenter avec plusieurs algorithmes
Vous allez entraîner différents modèles de classification et comparer leurs performances sur des métriques métiers et classiques.
L’objectif est de tester plusieurs familles de modèles (forêts, boosting, MLP, etc.) et de construire une première version de votre pipeline d’apprentissage.
Vous devez aussi intégrer une validation croisée pour évaluer leur robustesse.

## Prérequis
- Avoir préparé un dataset propre et prêt à l’entraînement.
- Avoir compris la nature déséquilibrée du jeu de données.
- Avoir identifié les variables cibles et explicatives.
- Avoir installé les bibliothèques de machine learning nécessaires.
- Avoir paramétré MLflow.

## Résultat attendu
Un ou plusieurs modèles entraînés, avec validation croisée et premières métriques d’évaluation.

## Recommandations
- Commencer par tester des modèles simples (Logistic Regression, Random Forest).
- Comparer ensuite avec des modèles plus puissants (XGBoost, LightGBM, MLP).
- Utiliser StratifiedKFold pour conserver la distribution des classes et garantir une évaluation robuste.
- Entraîner les modèles dans des notebooks clairs et documentés.
- Stocker les scores et les hyperparamètres testés.

## Points de vigilance
- Ne pas tester sans validation croisée. Une évaluation basée uniquement sur un split train/test unique peut produire des résultats très variables selon le hasard du découpage, et conduire à des conclusions erronées sur la performance réelle d’un modèle.
- Ne pas comparer les modèles avec des métriques inadaptées. Privilégier des métriques pertinentes, telles que :
- AUC-ROC,
- Recall sur la classe minoritaire,
- F1-score,
- Coût métier personnalisé (𝐹 𝑁 ≫ 𝐹𝑃).
- Ne pas oublier la stratification. Sans stratification, certains algorithmes peuvent se biaisent vers la classe majoritaire si le dataset contient beaucoup plus de bons que de mauvais clients.
- Ne pas ignorer le déséquilibre des classes. Utiliser un class_weight adapté ou du sur-échantillonnage (SMOTE, etc.) pour éviter de biaiser l’apprentissage.

## Outils
- scikit-learn
- XGBoost
- LightGBM

# Étape 4 : Optimiser les hyperparamètres et le seuil métier
Vous allez optimiser les hyperparamètres des modèles pour maximiser leurs performances selon des critères métier. Vous définirez également un seuil de décision optimal basé sur le coût des erreurs.
L’objectif est de minimiser le coût métier total (avec un poids plus fort sur les faux négatifs que sur les faux positifs).

## Prérequis
- Avoir entraîné plusieurs modèles.
- Avoir comparé leurs performances de base.
- Avoir compris la notion de coût d’erreur.
- Avoir défini une fonction de coût métier.

## Résultat attendu
Un modèle avec hyperparamètres optimisés et un seuil métier ajusté.

## Recommandations
- Utiliser GridSearchCV ou Optuna pour l’optimisation.
- Définir une fonction de coût pondérant les erreurs FN et FP.
- Tester différents seuils de classification (par exemple de 0.1 à 0.9).
- Tracer la courbe coût vs. seuil pour identifier la meilleure décision.

## Points de vigilance
- Garder le seuil par défaut (0.5) sans justification. Ce seuil ne reflète pas nécessairement les enjeux métiers : il doit être optimisé selon le ratio FN/FP.
- Oublier de tracer le score métier en fonction du seuil : cela empêche d’identifier la meilleure décision.
- Optimiser uniquement sur l’AUC ou l’accuracy : ces métriques ne reflètent pas toujours les pertes métier.
- Oublier d’adapter les métriques aux besoins business.
- Choisir un modèle sans tester sa robustesse.

## Outils
- scikit-learn (GridSearchCV)
- Optuna

## Résultat attendu :
un modèle final optimisé et justifié.