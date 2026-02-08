import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, LogisticRegressionCV, ElasticNetCV
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.exceptions import ConvergenceWarning

from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

from tqdm import tqdm

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.set_option('display.max_columns', None)

# -------------------------------------------------------------------
# CONFIGURATION GLOBALE & CONSTANTES
# -------------------------------------------------------------------
DEFAULT_N_BOOTSTRAP = 100       # Nombre d'itérations pour la stabilité
STABILITY_THRESHOLD = 0.65      # Seuil de sélection (une variable doit apparaître dans 65% des bootstraps)
RANDOM_SEED = 42

# Chemins
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../data/combined_covid_data.csv"

# -------------------------------------------------------------------
# DÉFINITION DE LA STRUCTURE CAUSALE
# -------------------------------------------------------------------
categorical_features = ["Age_at_ICI_start","Ethnicity","Gender","Immunotherapy_Agent"]
numeric_features = ["BRAF","CNS_disease","Concurrent_Chemo","ECOG",
                    "English_as_primary_language", "Previous_history_of_malignancy_at_ICI_start",
                    "Steroid_win_1_month_of_ICI_start","Steroid_win_1_month_of_Vaccine",
                    "Vaccine100"
                    ]
target_variables = ["OS_event", "OS_months"]

CAUSAL_LEVELS = {
    # Niveau 0 : Variables démographiques
    0: ["Gender", "Ethnicity", "Age_at_ICI_start", "English_as_primary_language"],
    
    # Niveau 1 : Caractéristiques baseline de la maladie
    1: ["Simplified_Stage", "ECOG", "CNS_disease", "Previous_history_of_malignancy_at_ICI_start"],
    
    # Niveau 2 : Traitements et interventions
    2: ["Vaccine100", "Steroid_win_1_month_of_Vaccine", "Concurrent_Chemo", "BRAF", "Immunotherapy_Agent", "Steroid_win_1_month_of_ICI_start"],
    
    # Niveau 3 : Outcomes intermédiaires
    3: ["PFS_", "PFS_Code"],
    
    # Niveau 4 : Outcome final
    4: ["OS_months", "OS_event"]
}

# Mapping inversé pour faciliter les lookups
VAR_TO_LEVEL = {}
for level, vars_list in CAUSAL_LEVELS.items():
    for var in vars_list:
        VAR_TO_LEVEL[var] = level

#On effectue une sélection de variables par stabilité (Stability Selection) seulement pour OS_event, 
#car OS_month peut etre = 8 même si le patient est en vie à 120 ==> prédiction faussé
def run_stability_selection(df, candidates, target="OS_event", n_bootstrap=100, threshold=0.6, random_state=42):
    """
    Exécute une sélection de variables par stabilité (Stability Selection).
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        target (str): Le nom de la colonne cible.
        candidates (list): La liste des colonnes prédictives candidates.
        n_bootstrap (int): Nombre de ré-échantillonnages bootstrap.
        threshold (float): Fréquence minimale de sélection (0.0 à 1.0).
        random_state (int): Graine aléatoire.
        
    Returns:
        dict: {
            'selected_features': liste des noms de features retenues,
            'stability_scores': dict {feature: score},
            'n_iterations': int
        }
    """
    cols_to_use = candidates + [target]
    data = df[cols_to_use].dropna()

    if len(data) < 50:
        print(f"   Trop peu de données pour {target} (n={len(data)}). Skip.")
        return {'selected_features': [], 'stability_scores': {}, 'n_iterations': 0}

    X = data[candidates]
    y = data[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [c for c in numeric_features if c in candidates]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
             [c for c in categorical_features if c in candidates])
        ],
        verbose_feature_names_out=True
    )

    model = LogisticRegressionCV(
        cv=3,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.1, 0.5, 0.9],
        scoring='roc_auc',
        max_iter=2000,
        random_state=random_state
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    def _map_encoded_to_original(encoded_name, candidates_list):
        # "num__Age_at_ICI_start" -> "Age_at_ICI_start"
        if encoded_name.startswith("num__"):
            return encoded_name.replace("num__", "")
        # "cat__Gender_Male" -> "Gender"
        if encoded_name.startswith("cat__"):
            tail = encoded_name.replace("cat__", "")
            for cand in sorted(candidates_list, key=len, reverse=True):
                if tail == cand or tail.startswith(cand + "_"):
                    return cand
        return None

    np.random.seed(random_state)
    seeds = np.random.randint(0, 100000, size=n_bootstrap)

    selection_counts = {feature: 0 for feature in candidates}
    n_successful_runs = 0

    print(f"   Running Stability Selection for target: {target} with {n_bootstrap} bootstraps...")
    for seed in tqdm(seeds, desc=f"Bootstrap {target}", total=len(seeds)):
        X_resampled, y_resampled = resample(
            X, y, replace=True, n_samples=len(X), random_state=seed
        )

        if y_resampled.nunique() < 2:
            continue

        try:
            pipeline.fit(X_resampled, y_resampled)
        except Exception:
            continue

        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        coefs = pipeline.named_steps['model'].coef_.ravel()

        selected_encoded = feature_names[np.abs(coefs) > 1e-6]
        selected_original = set()

        for enc in selected_encoded:
            orig = _map_encoded_to_original(enc, candidates)
            if orig is not None:
                selected_original.add(orig)

        for feature in selected_original:
            selection_counts[feature] += 1

        n_successful_runs += 1

    if n_successful_runs == 0:
        print(f"   Aucune itération réussie pour {target}.")
        return {'selected_features': [], 'stability_scores': {}, 'n_iterations': 0}

    stability_scores = {feature: count / n_successful_runs for feature, count in selection_counts.items()}
    selected_features = [feature for feature, score in stability_scores.items() if score >= threshold]

    print(f"   Selected {len(selected_features)} features for {target} (threshold={threshold}): {selected_features}")
    return {
        'selected_features': selected_features,
        'stability_scores': stability_scores,
        'n_iterations': n_successful_runs
    }


if __name__ == "__main__":
    # Test rapide de la stability selection
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Candidats = variables connues (num + cat) présentes dans le df
    candidates = [c for c in (numeric_features + categorical_features) if c in df.columns]

    result = run_stability_selection(
        df=df,
        candidates=candidates,
        target="OS_event",
        n_bootstrap=20,          # petit pour test rapide
        threshold=0.6,
        random_state=RANDOM_SEED
    )

    # Vérifications de base
    assert "selected_features" in result
    assert "stability_scores" in result
    assert "n_iterations" in result
    assert isinstance(result["selected_features"], list)
    assert isinstance(result["stability_scores"], dict)

    print("✅ Test OK")
    print(f"n_iterations: {result['n_iterations']}")
    print(f"selected_features: {result['selected_features']}")
# ...existing code...