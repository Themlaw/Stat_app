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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.exceptions import ConvergenceWarning

from lifelines import CoxPHFitter
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

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


def map_feature_to_original(feature_name, candidate_variables):
    """
    Ramène un nom de feature potentiellement encodé (ex: Gender_Male)
    vers la variable source (ex: Gender).
    """
    if feature_name in candidate_variables:
        return feature_name

    for base in sorted(candidate_variables, key=len, reverse=True):
        if feature_name.startswith(base + "_"):
            return base

    return feature_name

# -------------------------------------------------------------------
# HELPER FONCTIONS
# -------------------------------------------------------------------
def compute_e_value(coef, se, is_binary=True):
    """
    Calcule la E-value pour une estimation (Risk Ratio, Odds Ratio, Hazard Ratio)
    ou pour une différence de moyennes standardisée (si is_binary=False).

    Ref: VanderWeele, T. J., & Ding, P. (2017).
         Sensitivity Analysis in Observational Research: Introducing the E-Value.

    Args:
        coef (float): Le coefficient (beta) du modèle (log-scale pour binaire).
        se (float): L'erreur standard du coefficient.
        is_binary (bool): True si outcome binaire/survie (Logit/Cox),
                          False si continu (OLS, coef interprété comme SMD).

    Returns:
        dict: {
            'rr_point': estimation transformée sur une échelle RR-like,
            'rr_ci_low': borne basse IC95% (RR-like),
            'rr_ci_high': borne haute IC95% (RR-like),
            'e_value_point': E-value pour l'estimation ponctuelle,
            'e_value_ci': E-value pour la borne de l'IC95% la plus proche du nul
        }
    """
    # Validation minimale pour éviter des résultats non interprétables.
    if not np.isfinite(coef):
        raise ValueError("coef doit être un nombre fini")
    if (se is None) or (not np.isfinite(se)) or (se < 0):
        raise ValueError("se doit être un nombre fini >= 0")

    def _evalue_from_ratio(rr):
        """Calcule l'E-value à partir d'un ratio (RR/OR/HR-like)."""
        if rr <= 0 or not np.isfinite(rr):
            return np.nan
        rr_ref = rr if rr >= 1 else 1.0 / rr
        if rr_ref <= 1:
            return 1.0
        return float(rr_ref + np.sqrt(rr_ref * (rr_ref - 1.0)))

    # IC95% Wald: beta ± 1.96 * SE.
    z = 1.96

    if is_binary:
        # Cas logit/cox: le coef est sur l'échelle log-ratio.
        rr_point = float(np.exp(coef))
        rr_ci_low = float(np.exp(coef - z * se))
        rr_ci_high = float(np.exp(coef + z * se))
    else:
        # Cas continu: approximation SMD -> RR-like.
        # (VanderWeele & Ding, 2017): RR ≈ exp(0.91 * SMD)
        rr_point = float(np.exp(0.91 * coef))
        rr_ci_low = float(np.exp(0.91 * (coef - z * se)))
        rr_ci_high = float(np.exp(0.91 * (coef + z * se)))

    e_value_point = _evalue_from_ratio(rr_point)

    # E-value conservative: on prend la borne d'IC la plus proche du nul (RR=1).
    if rr_ci_low <= 1.0 <= rr_ci_high:
        e_value_ci = 1.0
    else:
        if rr_point >= 1.0:
            nearest_bound = rr_ci_low
        else:
            nearest_bound = rr_ci_high
        e_value_ci = _evalue_from_ratio(nearest_bound)

    return {
        'rr_point': rr_point,
        'rr_ci_low': rr_ci_low,
        'rr_ci_high': rr_ci_high,
        'e_value_point': e_value_point,
        'e_value_ci': e_value_ci
    }

def fit_unpenalized_model(df_train, features, is_binary, target="OS_event"):
    """
    Ré-entraîne un modèle non pénalisé (Logit/OLS) sur les features sélectionnées
    et renvoie les coefficients inférentiels (SE, p-values, IC95%, E-values).

    Args:
        df_train (pd.DataFrame): Données d'entraînement (ex: Fold B du cross-fitting).
        features (list): Variables sélectionnées à l'étape de sélection.
        is_binary (bool): True -> Logit, False -> OLS.
        target (str): Nom de la variable cible.

    Returns:
        dict: {
            'target': str,
            'is_binary': bool,
            'n_obs': int,
            'n_features_input': int,
            'n_features_encoded': int,
            'model': objet statsmodels ajusté,
            'results_df': pd.DataFrame des coefficients
        }
    """
    # Vérifications d'entrée pour sécuriser le pipeline cross-fit.
    if target not in df_train.columns:
        raise ValueError(f"La cible '{target}' est absente du DataFrame")

    if not features:
        return {
            'target': target,
            'is_binary': is_binary,
            'n_obs': 0,
            'n_features_input': 0,
            'n_features_encoded': 0,
            'model': None,
            'results_df': pd.DataFrame()
        }

    selected_features = [f for f in dict.fromkeys(features) if (f in df_train.columns and f != target)]
    if not selected_features:
        return {
            'target': target,
            'is_binary': is_binary,
            'n_obs': 0,
            'n_features_input': len(features),
            'n_features_encoded': 0,
            'model': None,
            'results_df': pd.DataFrame()
        }

    # Sous-ensemble utile puis suppression des NA.
    cols = selected_features + [target]
    data = df_train[cols].copy()
    data[target] = pd.to_numeric(data[target], errors='coerce')
    data = data.dropna()

    if data.empty:
        return {
            'target': target,
            'is_binary': is_binary,
            'n_obs': 0,
            'n_features_input': len(selected_features),
            'n_features_encoded': 0,
            'model': None,
            'results_df': pd.DataFrame()
        }

    y = data[target].copy()
    if is_binary:
        # En binaire, on force un encodage 0/1 pour statsmodels.Logit.
        unique_vals = sorted(pd.unique(y))
        if len(unique_vals) != 2:
            raise ValueError(
                f"La cible binaire '{target}' doit contenir exactement 2 classes après nettoyage, obtenu: {unique_vals}"
            )
        if set(unique_vals) != {0, 1}:
            y = y.map({unique_vals[0]: 0, unique_vals[1]: 1})

    # Encodage des catégorielles (one-hot)
    # IMPORTANT: drop_first=True évite la dummy trap AVEC add_constant
    # (si on fait drop_first=True, la première catégorie devient la référence implicite)
    X_raw = data[selected_features]
    X = pd.get_dummies(X_raw, drop_first=True, dtype=float)

    if X.shape[1] == 0:
        return {
            'target': target,
            'is_binary': is_binary,
            'n_obs': int(len(y)),
            'n_features_input': len(selected_features),
            'n_features_encoded': 0,
            'model': None,
            'results_df': pd.DataFrame()
        }

    # CORRECTION CRITIQUE: Ajouter la constante AVANT de vérifier la multicolinéarité
    # (add_constant avec has_constant='add' ajoute seulement si absent)
    X = add_constant(X, has_constant='add')

    # Vérifier la multicolinéarité et les valeurs extrêmes
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] > 0:
        # Retirer les colonnes avec variance nulle (vrai problème de multicolinéarité)
        zero_var_cols = X_numeric.columns[X_numeric.var() < 1e-10]
        if len(zero_var_cols) > 0:
            X = X.drop(columns=zero_var_cols)
            print(f"   Info: Colonnes à variance nulle retirées (multicolinéarité): {list(zero_var_cols)}")

        # Vérifier condition number (indicateur mais pas critique)
        try:
            cond_number = np.linalg.cond(X.values)
            # Note: Avec dummies, cond_number peut être élevé MAIS c'est normal
            # On log seulement si > 1e15 (singularité vraie)
            if cond_number > 1e15:
                print(f"   Warning: Matrice quasi-singulière détectée (cond={cond_number:.2e})")
        except:
            pass

    try:
        # STRATÉGIE: Séparer coefficients robustes et inférence statistique
        # ====================================================================
        # coef_model : Source des coefficients β (régularisés si besoin)
        # inference_model : Source de SE/p-values/CI (fit MLE non pénalisé, peut échouer)
        # has_inference : Flag indiquant si stats inférentielles sont disponibles

        if is_binary:
            # ÉTAPE 1: Fit régularisé L1 pour obtenir des coefficients stables
            # (même en cas de séparation parfaite ou quasi-séparation)
            coef_model = Logit(y, X).fit_regularized(
                method='l1',
                alpha=0.001,  # Très faible pénalité (minimal bias)
                disp=False,
                maxiter=1000,
                trim_mode='auto'
            )

            # ÉTAPE 2: Essayer un fit MLE classique (sans pénalité) pour l'inférence
            # Cela peut échouer si séparation/quasi-séparation → fallback sans SE/p-val
            try:
                inference_model = Logit(y, X).fit(
                    disp=False,
                    maxiter=500,
                    method='bfgs'
                )
                conf_int = inference_model.conf_int()
                has_inference = True  # Stats inférentielles disponibles ✓
            except Exception:
                # Fit MLE a échoué → on garde coef_model mais pas d'inférence
                inference_model = None
                conf_int = None
                has_inference = False  # Inférence indisponible, on skipera SE/p-val
        else:
            # Cas continu (OLS): un seul fit suffit (rarement d'instabilité numérique)
            coef_model = OLS(y, X).fit()
            inference_model = coef_model
            conf_int = coef_model.conf_int()
            has_inference = True  # Inférence toujours dispo en OLS

        rows = []

        # ITÉRATION: Extraire coefficients et stats pour chaque variable
        for param in coef_model.params.index:
            if param == 'const':
                continue

            # Source unique et fiable pour les coefficients
            beta = float(coef_model.params[param])

            # VALIDATION: Coefficient "raisonnable"
            # (beta > 50 en log-odds = OR > 5e21, non interprétable)
            if not np.isfinite(beta) or abs(beta) > 50:
                print(f"   Warning: Coefficient extrême pour {param}: {beta:.2f}")
                continue

            # INFÉRENCE: Récupérer SE/p-values/IC si disponibles, sinon NaN
            if has_inference and hasattr(inference_model, 'bse'):
                # Cas nominal: on a un fit MLE valide
                se = float(inference_model.bse[param])
                pval = float(inference_model.pvalues[param])
                ci_low = float(conf_int.loc[param, 0])
                ci_high = float(conf_int.loc[param, 1])

                # VALIDATION: SE doit être fini et positif (sinon on signale mais on garde le coef)
                if not np.isfinite(se) or se <= 0:
                    # Ne pas faire continue! On garde le coefficient même sans SE valide
                    print(f"   Info: SE invalide pour {param}: {se} (coef={beta:.3f} conservé)")
                    se = np.nan
                    pval = np.nan
                    ci_low = np.nan
                    ci_high = np.nan
            else:
                # Cas fallback: fit régularisé uniquement (pas de SE/p-val valides)
                # On met NaN pour indiquer "stats indisponibles"
                se = np.nan
                pval = np.nan
                ci_low = np.nan
                ci_high = np.nan

            # TRANSFORMATION: exp(coef) pour résultats interprétables (OR/HR)
            if is_binary:
                effect_point = float(np.exp(np.clip(beta, -20, 20)))
                effect_ci_low = float(np.exp(np.clip(ci_low, -20, 20)))
                effect_ci_high = float(np.exp(np.clip(ci_high, -20, 20)))
            else:
                effect_point = beta
                effect_ci_low = ci_low
                effect_ci_high = ci_high

            # E-VALUE: Sensibilité à une variable non-mesurée confondante
            # (Calcul seulement si SE disponible et valide)
            if np.isfinite(se) and se > 0:
                try:
                    e_val = compute_e_value(beta, se, is_binary=is_binary)
                except Exception:
                    # Fallback si compute_e_value échoue
                    e_val = {
                        'rr_point': np.nan, 'rr_ci_low': np.nan, 'rr_ci_high': np.nan,
                        'e_value_point': np.nan, 'e_value_ci': np.nan
                    }
            else:
                # SE invalide → E-value indisponible
                e_val = {
                    'rr_point': np.nan, 'rr_ci_low': np.nan, 'rr_ci_high': np.nan,
                    'e_value_point': np.nan, 'e_value_ci': np.nan
                }

            # ENREGISTREMENT: Une ligne = une variable avec toutes ses stats
            rows.append({
                'feature': param,
                'coef': beta,
                'se': se,
                'p_value': pval,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'effect_point': effect_point,
                'effect_ci_low': effect_ci_low,
                'effect_ci_high': effect_ci_high,
                'rr_like': e_val['rr_point'],
                'rr_like_ci_low': e_val['rr_ci_low'],
                'rr_like_ci_high': e_val['rr_ci_high'],
                'e_value_point': e_val['e_value_point'],
                'e_value_ci': e_val['e_value_ci']
            })
    except Exception as e:
        print(f"   Erreur lors de l'ajustement du modèle pour {target} avec {len(selected_features)} features: {e}")
        return {
            'target': target,
            'is_binary': is_binary,
            'n_obs': int(len(y)),
            'n_features_input': len(selected_features),
            'n_features_encoded': int(X.shape[1] - 1),
            'model': None,
            'results_df': pd.DataFrame()
        }
        
    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        # Tri par p-value pour lecture rapide des signaux les plus forts
        results_df = results_df.sort_values('p_value').reset_index(drop=True)

    return {
        'target': target,
        'is_binary': is_binary,
        'n_obs': int(len(y)),
        'n_features_input': len(selected_features),
        'n_features_encoded': int(X.shape[1] - 1),
        'model': coef_model,  # Retourner le modèle des coefficients (régularisé si nécessaire)
        'results_df': results_df
    }

def aggregate_results(results_list):
    """
    Prend les p-values trouvées sur le Fold A et le Fold B et les combine 
    mathématiquement (Méthode de Fisher) pour garder 100% de la puissance statistique.

    Args:
        results_list (list): Liste de sorties de `fit_unpenalized_model`.

    Returns:
        pd.DataFrame: tableau agrégé par feature, avec p-value combinée et q-value.
    """
    # On agrège uniquement les folds ayant réellement produit des résultats.
    if not results_list:
        return pd.DataFrame()

    collected = []
    for fold_idx, res in enumerate(results_list, start=1):
        if not isinstance(res, dict):
            continue

        df_res = res.get('results_df', pd.DataFrame())
        if df_res is None or df_res.empty:
            continue

        fold_df = df_res.copy()
        fold_df['fold_id'] = fold_idx
        fold_df['target'] = res.get('target', None)
        fold_df['is_binary'] = res.get('is_binary', None)
        collected.append(fold_df)

    if not collected:
        return pd.DataFrame()

    # Table longue: une ligne = (feature, fold).
    stacked = pd.concat(collected, ignore_index=True)

    numeric_cols = [
        'coef', 'se', 'p_value', 'ci_low', 'ci_high', 'effect_point',
        'effect_ci_low', 'effect_ci_high', 'rr_like', 'rr_like_ci_low',
        'rr_like_ci_high', 'e_value_point', 'e_value_ci'
    ]
    for col in numeric_cols:
        if col in stacked.columns:
            stacked[col] = pd.to_numeric(stacked[col], errors='coerce')

    aggregated_rows = []
    for feature, grp in stacked.groupby('feature', dropna=True):
        grp = grp.copy()
        pvals = grp['p_value'].dropna().to_numpy(dtype=float)

        if len(pvals) == 0:
            p_fisher = np.nan
            fisher_stat = np.nan
        else:
            # Méthode de Fisher: combine l'évidence statistique des folds.
            pvals = np.clip(pvals, 1e-300, 1.0)
            fisher_stat, p_fisher = combine_pvalues(pvals, method='fisher')
            fisher_stat = float(fisher_stat)
            p_fisher = float(p_fisher)

        coef_vals = grp['coef'].dropna().to_numpy(dtype=float)
        coef_mean = float(np.mean(coef_vals)) if len(coef_vals) else np.nan
        coef_std = float(np.std(coef_vals, ddof=1)) if len(coef_vals) > 1 else 0.0

        ci_low_mean = float(np.nanmean(grp['ci_low'])) if grp['ci_low'].notna().any() else np.nan
        ci_high_mean = float(np.nanmean(grp['ci_high'])) if grp['ci_high'].notna().any() else np.nan
        effect_point_mean = float(np.nanmean(grp['effect_point'])) if grp['effect_point'].notna().any() else np.nan
        effect_ci_low_mean = float(np.nanmean(grp['effect_ci_low'])) if grp['effect_ci_low'].notna().any() else np.nan
        effect_ci_high_mean = float(np.nanmean(grp['effect_ci_high'])) if grp['effect_ci_high'].notna().any() else np.nan

        rr_like_mean = float(np.nanmean(grp['rr_like'])) if grp['rr_like'].notna().any() else np.nan
        rr_like_ci_low_mean = float(np.nanmean(grp['rr_like_ci_low'])) if grp['rr_like_ci_low'].notna().any() else np.nan
        rr_like_ci_high_mean = float(np.nanmean(grp['rr_like_ci_high'])) if grp['rr_like_ci_high'].notna().any() else np.nan
        e_value_point_mean = float(np.nanmean(grp['e_value_point'])) if grp['e_value_point'].notna().any() else np.nan
        e_value_ci_mean = float(np.nanmean(grp['e_value_ci'])) if grp['e_value_ci'].notna().any() else np.nan

        aggregated_rows.append({
            'feature': feature,
            'target': grp['target'].dropna().iloc[0] if grp['target'].notna().any() else None,
            'is_binary': bool(grp['is_binary'].dropna().iloc[0]) if grp['is_binary'].notna().any() else None,
            'n_folds_present': int(grp['fold_id'].nunique()),
            'fisher_stat': fisher_stat,
            'p_value_fisher': p_fisher,
            'coef_mean': coef_mean,
            'coef_std': coef_std,
            'ci_low_mean': ci_low_mean,
            'ci_high_mean': ci_high_mean,
            'effect_point_mean': effect_point_mean,
            'effect_ci_low_mean': effect_ci_low_mean,
            'effect_ci_high_mean': effect_ci_high_mean,
            'rr_like_mean': rr_like_mean,
            'rr_like_ci_low_mean': rr_like_ci_low_mean,
            'rr_like_ci_high_mean': rr_like_ci_high_mean,
            'e_value_point_mean': e_value_point_mean,
            'e_value_ci_mean': e_value_ci_mean
        })

    out = pd.DataFrame(aggregated_rows)
    if out.empty:
        return out

    valid_mask = out['p_value_fisher'].notna()
    out['p_value_fdr_bh'] = np.nan
    out['is_significant_fdr_5pct'] = False

    if valid_mask.any():
        # Contrôle du risque de faux positifs sur l'ensemble des variables.
        reject, pvals_corr, _, _ = multipletests(
            out.loc[valid_mask, 'p_value_fisher'].values,
            alpha=0.05,
            method='fdr_bh'
        )
        out.loc[valid_mask, 'p_value_fdr_bh'] = pvals_corr
        out.loc[valid_mask, 'is_significant_fdr_5pct'] = reject

    # Sortie finale triée par signal statistique combiné.
    out = out.sort_values(['p_value_fisher', 'feature'], na_position='last').reset_index(drop=True)
    return out

def bootstrap_mediation_robust(df, cause, mediator, outcome, n_boot=1000, random_seed=42):
    """
    Estime un effet de médiation par bootstrap en gérant les cas continus/binaires.

    Chemins estimés:
      - a: cause -> mediator
      - b: mediator -> outcome (ajusté sur cause)
      - c': effet direct de cause sur outcome (ajusté sur mediator)
      - c: effet total de cause sur outcome

    Retourne des statistiques bootstrap pour l'effet indirect (a*b), direct (c')
    et total (c): moyenne, IC95%, p-valeur empirique.
    """
    if n_boot <= 0:
        raise ValueError("n_boot doit être > 0")

    for col in [cause, mediator, outcome]:
        if col not in df.columns:
            raise ValueError(f"Colonne absente: {col}")

    # Sous-ensemble utile et nettoyage de base.
    data = df[[cause, mediator, outcome]].copy()
    data = data.dropna()
    if data.empty:
        return None

    def _coerce_to_numeric(series):
        """Convertit en numérique; pour binaire catégoriel, mappe vers 0/1."""
        s_num = pd.to_numeric(series, errors='coerce')
        if s_num.notna().all():
            return s_num

        non_na = series.dropna().astype(str)
        unique_vals = sorted(non_na.unique())
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0.0, unique_vals[1]: 1.0}
            return series.astype(str).map(mapping)

        return s_num

    data[cause] = _coerce_to_numeric(data[cause])
    data[mediator] = _coerce_to_numeric(data[mediator])
    data[outcome] = _coerce_to_numeric(data[outcome])
    data = data.dropna()

    if data.empty:
        return None

    is_mediator_binary = data[mediator].nunique() == 2
    is_outcome_binary = data[outcome].nunique() == 2

    # Les valeurs binaires doivent être bien codées en 0/1 pour Logit.
    if is_mediator_binary and set(pd.unique(data[mediator])) != {0, 1}:
        vals = sorted(pd.unique(data[mediator]))
        data[mediator] = data[mediator].map({vals[0]: 0.0, vals[1]: 1.0})

    if is_outcome_binary and set(pd.unique(data[outcome])) != {0, 1}:
        vals = sorted(pd.unique(data[outcome]))
        data[outcome] = data[outcome].map({vals[0]: 0.0, vals[1]: 1.0})

    if data[cause].nunique() < 2 or data[mediator].nunique() < 2 or data[outcome].nunique() < 2:
        return None

    rng = np.random.default_rng(random_seed)

    def _single_mediation_boot(boot_seed, data_input, cause_col, mediator_col, outcome_col, is_med_binary, is_out_binary):
        """Exécute une seule itération bootstrap de médiation."""
        rng_boot = np.random.default_rng(boot_seed)
        boot_idx = rng_boot.integers(low=0, high=len(data_input), size=len(data_input))
        df_boot = data_input.iloc[boot_idx].copy()

        try:
            # Modèle a: mediator ~ cause
            X_a = add_constant(df_boot[[cause_col]], has_constant='add')
            y_a = df_boot[mediator_col]
            if is_med_binary:
                # Utiliser régularisation pour éviter séparation parfaite
                model_a = Logit(y_a, X_a).fit_regularized(method='l1', alpha=0.001, disp=False, maxiter=200)
            else:
                model_a = OLS(y_a, X_a).fit()

            a_coef = float(model_a.params[cause_col])
            if not np.isfinite(a_coef) or abs(a_coef) > 50:
                return {'success': False}

            # Modèle b/c': outcome ~ cause + mediator
            X_b = add_constant(df_boot[[cause_col, mediator_col]], has_constant='add')
            y_b = df_boot[outcome_col]
            if is_out_binary:
                model_b = Logit(y_b, X_b).fit_regularized(method='l1', alpha=0.001, disp=False, maxiter=200)
            else:
                model_b = OLS(y_b, X_b).fit()

            b_coef = float(model_b.params[mediator_col])
            c_prime = float(model_b.params[cause_col])

            if not np.isfinite(b_coef) or not np.isfinite(c_prime) or abs(b_coef) > 50 or abs(c_prime) > 50:
                return {'success': False}

            # Modèle total c: outcome ~ cause
            X_c = add_constant(df_boot[[cause_col]], has_constant='add')
            if is_out_binary:
                model_c = Logit(y_b, X_c).fit_regularized(method='l1', alpha=0.001, disp=False, maxiter=200)
            else:
                model_c = OLS(y_b, X_c).fit()
            c_total = float(model_c.params[cause_col])

            if not np.isfinite(c_total) or abs(c_total) > 50:
                return {'success': False}

            return {
                'indirect': a_coef * b_coef,
                'direct': c_prime,
                'total': c_total,
                'success': True
            }
        except Exception:
            return {'success': False}

    print(f"Bootstrap mediation {cause}->{mediator}->{outcome}: {n_boot} itérations (parallélisées)")
    seeds = np.random.randint(0, 100000, size=n_boot)

    # Parallélisation avec joblib (meilleur pour les stats)
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_single_mediation_boot)(seed, data, cause, mediator, outcome, is_mediator_binary, is_outcome_binary)
        for seed in seeds
    )

    indirect_effects = []
    direct_effects = []
    total_effects = []
    n_failed = 0

    for res in results:
        if res.get('success', False):
            indirect_effects.append(res['indirect'])
            direct_effects.append(res['direct'])
            total_effects.append(res['total'])
        else:
            n_failed += 1

    if len(indirect_effects) == 0:
        return None

    def _summarize_bootstrap(samples):
        arr = np.asarray(samples, dtype=float)
        mean_val = float(np.mean(arr))
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        prop_pos = float(np.mean(arr > 0))
        p_emp = float(2 * min(prop_pos, 1 - prop_pos))
        return {
            'mean': mean_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'prop_positive': prop_pos,
            'p_empirical': p_emp
        }

    indirect_stats = _summarize_bootstrap(indirect_effects)
    direct_stats = _summarize_bootstrap(direct_effects)
    total_stats = _summarize_bootstrap(total_effects)

    mean_total = total_stats['mean']
    mean_indirect = indirect_stats['mean']
    prop_mediated = np.nan
    if np.isfinite(mean_total) and abs(mean_total) > 1e-12:
        prop_mediated = float(mean_indirect / mean_total)

    return {
        'cause': cause,
        'mediator': mediator,
        'outcome': outcome,
        'is_mediator_binary': bool(is_mediator_binary),
        'is_outcome_binary': bool(is_outcome_binary),
        'indirect_effect': indirect_stats,
        'direct_effect': direct_stats,
        'total_effect': total_stats,
        'proportion_mediated_mean': prop_mediated,
        'n_boot_requested': int(n_boot),
        'n_boot_success': int(len(indirect_effects)),
        'n_boot_failed': int(n_failed)
    }

# -------------------------------------------------------------------
# FONCTIONS PRINCIPALES
# -------------------------------------------------------------------
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

    # Détection du type de cible pour choisir un modèle de sélection adapté.
    is_binary_target = y.nunique() == 2

    cat_candidates = [c for c in categorical_features if c in candidates]
    num_candidates = [c for c in numeric_features if c in candidates]

    # ElasticNetCV nécessite une matrice dense, alors que LogitCV gère bien le sparse.
    cat_sparse = True if is_binary_target else False

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_candidates),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=cat_sparse, drop='first'),
             cat_candidates)
        ],
        verbose_feature_names_out=True
    )

    if is_binary_target:
        model = LogisticRegressionCV(
            cv=2,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.1, 0.5, 0.9],
            scoring='roc_auc',
            max_iter=1000,
            n_jobs=-1,
            random_state=random_state
        )
    else:
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=3,
            max_iter=5000,
            n_jobs=-1,
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

    def _single_stability_iteration(seed, X_input, y_input, pipeline_obj, candidates_list):
        """Exécute une itération de stabilité selection."""
        try:
            X_resampled, y_resampled = resample(
                X_input, y_input, replace=True, n_samples=len(X_input), random_state=seed
            )

            if y_resampled.nunique() < 2:
                return set()

            pipeline_obj.fit(X_resampled, y_resampled)
            feature_names = pipeline_obj.named_steps['preprocessor'].get_feature_names_out()
            model_fitted = pipeline_obj.named_steps['model']
            coefs = np.asarray(model_fitted.coef_).ravel()

            selected_encoded = feature_names[np.abs(coefs) > 1e-6]
            selected_original = set()

            for enc in selected_encoded:
                orig = _map_encoded_to_original(enc, candidates_list)
                if orig is not None:
                    selected_original.add(orig)

            return selected_original
        except Exception:
            return set()

    model_name = "LogitCV(elasticnet)" if is_binary_target else "ElasticNetCV"
    print(f"   Running Stability Selection for target: {target} with {n_bootstrap} bootstraps [{model_name}]...")

    # Parallélisation avec joblib
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_single_stability_iteration)(seed, X, y, clone(pipeline), candidates)
        for seed in seeds
    )

    selection_counts = {feature: 0 for feature in candidates}
    n_successful_runs = 0

    for selected_original in results:
        if selected_original:  # Si le set n'est pas vide
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

def fit_cross_validated_inference(
    df,
    target,
    candidates,
    n_splits=2,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    stability_threshold=STABILITY_THRESHOLD
):
    """
    Orchestration du cross-fitting.
    À chaque itération: sélection de variables sur un fold, inférence non pénalisée
    sur un autre fold, puis agrégation des p-values entre itérations.

    Args:
        df (pd.DataFrame): Données complètes.
        target (str): Variable cible.
        candidates (list): Variables candidates pour la sélection.
        n_splits (int): Nombre de plis (2 recommandé ici).
        n_bootstrap (int): Nombre de bootstraps pour la stability selection.
        stability_threshold (float): Seuil de stabilité pour retenir une variable.

    Returns:
        dict: {
            'target': str,
            'is_binary': bool,
            'n_obs_used': int,
            'n_splits': int,
            'fold_details': list,
            'aggregated_results': pd.DataFrame
        }
    """
    if target not in df.columns:
        raise ValueError(f"La cible '{target}' est absente du DataFrame")
    if n_splits < 2:
        raise ValueError("n_splits doit être >= 2")

    available_candidates = [c for c in dict.fromkeys(candidates) if c in df.columns and c != target]
    if not available_candidates:
        return {
            'target': target,
            'is_binary': None,
            'n_obs_used': 0,
            'n_splits': n_splits,
            'fold_details': [],
            'aggregated_results': pd.DataFrame()
        }

    # Jeu de données propre commun à toutes les étapes du cross-fitting.
    cols = available_candidates + [target]
    data = df[cols].copy()
    data[target] = pd.to_numeric(data[target], errors='coerce')
    data = data.dropna()

    if len(data) < n_splits:
        return {
            'target': target,
            'is_binary': None,
            'n_obs_used': int(len(data)),
            'n_splits': n_splits,
            'fold_details': [],
            'aggregated_results': pd.DataFrame()
        }

    y = data[target]
    unique_vals = sorted(pd.unique(y))
    is_binary = len(unique_vals) == 2

    # Stratification pour conserver la balance des classes en binaire.
    if is_binary:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_indices = list(splitter.split(data, y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_indices = list(splitter.split(data))

    fold_results_for_aggregation = []
    fold_details = []

    def _process_fold(fold_idx, avail_cand, df_data, is_bin, split_inds, target_col, n_boot, stab_thresh):
        """Traite un seul fold de cross-validation."""
        select_idx = split_inds[fold_idx][1]
        infer_idx = split_inds[(fold_idx + 1) % len(split_inds)][1]

        df_select = df_data.iloc[select_idx].copy()
        df_infer = df_data.iloc[infer_idx].copy()

        selection = run_stability_selection(
            df=df_select,
            candidates=avail_cand,
            target=target_col,
            n_bootstrap=n_boot,
            threshold=stab_thresh,
            random_state=RANDOM_SEED + fold_idx
        )

        selected_features = selection.get('selected_features', [])
        fit_result = fit_unpenalized_model(
            df_train=df_infer,
            features=selected_features,
            is_binary=is_bin,
            target=target_col
        )

        fit_result['selector_fold'] = fold_idx
        fit_result['inference_fold'] = (fold_idx + 1) % len(split_inds)
        fit_result['selected_features'] = selected_features
        fit_result['stability_scores'] = selection.get('stability_scores', {})
        fit_result['n_bootstrap_success'] = selection.get('n_iterations', 0)

        fold_detail = {
            'selector_fold': fold_idx,
            'inference_fold': (fold_idx + 1) % len(split_inds),
            'n_select_obs': int(len(df_select)),
            'n_infer_obs': int(len(df_infer)),
            'n_selected_features': int(len(selected_features)),
            'selected_features': selected_features,
            'n_bootstrap_success': int(selection.get('n_iterations', 0))
        }

        return fit_result, fold_detail

    print(f"Cross-fit {target}: {n_splits} folds (parallélisés)")

    # Parallélisation avec joblib
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_process_fold)(i, available_candidates, data, is_binary, split_indices, target, n_bootstrap, stability_threshold)
        for i in range(n_splits)
    )

    for fit_result, fold_detail in results:
        fold_results_for_aggregation.append(fit_result)
        fold_details.append(fold_detail)

    aggregated = aggregate_results(fold_results_for_aggregation)

    return {
        'target': target,
        'is_binary': is_binary,
        'n_obs_used': int(len(data)),
        'n_splits': n_splits,
        'fold_details': fold_details,
        'aggregated_results': aggregated
    }

def run_cox_validation(df, all_links, base_levels_to_adjust=[0, 1]):
    """
    Prend les prédicteurs de survie trouvés par le DAG et les valide
    dans un modèle de Cox multivarié temporel.
    """
    if "OS_months" not in df.columns or "OS_event" not in df.columns:
        return {
            'status': 'skipped',
            'reason': "Colonnes OS_months/OS_event absentes",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    if not isinstance(all_links, (list, pd.DataFrame)) or len(all_links) == 0:
        return {
            'status': 'skipped',
            'reason': "Aucun lien fourni dans all_links",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    links_df = pd.DataFrame(all_links) if isinstance(all_links, list) else all_links.copy()
    if links_df.empty:
        return {
            'status': 'skipped',
            'reason': "all_links vide après conversion",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    def _pick_first_existing(row, candidates, default=None):
        for c in candidates:
            if c in row and pd.notna(row[c]):
                return row[c]
        return default

    normalized_rows = []
    for _, row in links_df.iterrows():
        src = _pick_first_existing(row, ['from', 'feature', 'predictor'])
        dst = _pick_first_existing(row, ['to', 'target'])
        coef = _pick_first_existing(row, ['coef', 'coef_mean', 'effect_point', 'effect_point_mean'], np.nan)
        if src is None or dst is None:
            continue
        normalized_rows.append({'from': src, 'to': dst, 'coef': coef})

    norm_links = pd.DataFrame(normalized_rows)
    if norm_links.empty:
        return {
            'status': 'skipped',
            'reason': "Impossible d'extraire des colonnes from/to depuis all_links",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    os_targets = {"OS", "OS_months", "OS_event"}
    dag_os_links = norm_links[norm_links['to'].isin(os_targets)].copy()
    dag_os_predictors = [p for p in dag_os_links['from'].dropna().unique().tolist() if p in df.columns]

    if not dag_os_predictors:
        return {
            'status': 'skipped',
            'reason': "Aucun prédicteur OS dérivé du DAG présent dans le DataFrame",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    adjust_vars = []
    for lvl in base_levels_to_adjust:
        for var in CAUSAL_LEVELS.get(lvl, []):
            if var in df.columns and var not in ['OS_months', 'OS_event']:
                adjust_vars.append(var)
    adjust_vars = [v for v in dict.fromkeys(adjust_vars)]

    predictor_set = [v for v in dict.fromkeys(dag_os_predictors + adjust_vars)]

    df_cox = df[['OS_months', 'OS_event']].copy()
    dropped_high_cardinality = []

    print(f"Préparation variables Cox: {len(predictor_set)} variables candidates")
    for col in predictor_set:
        s = df[col]
        s_num = pd.to_numeric(s, errors='coerce')

        if s_num.notna().mean() >= 0.95:
            df_cox[col] = s_num
            continue

        n_unique = s.nunique(dropna=True)
        if n_unique <= 8:
            dummies = pd.get_dummies(s.astype('category'), prefix=col, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                df_cox = pd.concat([df_cox, dummies], axis=1)
        else:
            dropped_high_cardinality.append(col)

    df_cox['OS_months'] = pd.to_numeric(df_cox['OS_months'], errors='coerce')
    df_cox['OS_event'] = pd.to_numeric(df_cox['OS_event'], errors='coerce')
    df_cox = df_cox.dropna()

    if len(df_cox) < 50:
        return {
            'status': 'skipped',
            'reason': f"Trop peu d'observations après nettoyage pour Cox (n={len(df_cox)})",
            'n_obs': int(len(df_cox)),
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame(),
            'dropped_high_cardinality': dropped_high_cardinality
        }

    cox = CoxPHFitter(penalizer=0.01)
    try:
        cox.fit(df_cox, duration_col='OS_months', event_col='OS_event')
        summary = cox.summary.sort_values('p', ascending=True).copy()
    except Exception as e:
        return {
            'status': 'error',
            'reason': f"Échec de convergence du modèle de Cox : {e}",
            'n_obs': int(len(df_cox)),
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }
    if not summary.empty:
        reject, p_adj, _, _ = multipletests(summary['p'].values, method='fdr_bh')
        summary['p_adj'] = p_adj
        summary['significant_fdr'] = reject

    def _map_to_original_variable(name):
        if name in predictor_set:
            return name
        for base in sorted(predictor_set, key=len, reverse=True):
            if str(name).startswith(base + '_'):
                return base
        return str(name)

    summary['original_var'] = summary.index.map(_map_to_original_variable)
    significant = summary[summary['p_adj'] < 0.05].copy() if 'p_adj' in summary.columns else pd.DataFrame()

    dag_effects = dag_os_links.groupby('from', dropna=True)['coef'].mean().to_dict()
    agreements = []
    disagreements = []

    for _, row in significant.iterrows():
        var = row['original_var']
        if var not in dag_effects:
            continue
        dag_coef = dag_effects[var]
        if pd.isna(dag_coef):
            continue
        cox_coef = row['coef']
        if (cox_coef * dag_coef) > 0:
            agreements.append(var)
        elif (cox_coef * dag_coef) < 0:
            disagreements.append(var)

    return {
        'status': 'ok',
        'n_obs': int(len(df_cox)),
        'n_predictors_from_dag': int(len(dag_os_predictors)),
        'n_adjustment_vars': int(len(adjust_vars)),
        'dropped_high_cardinality': dropped_high_cardinality,
        'n_significant_fdr': int(len(significant)),
        'agreement_rate': (len(set(agreements)) / len(significant)) if len(significant) > 0 else np.nan,
        'agreements': sorted(set(agreements)),
        'disagreements': sorted(set(disagreements)),
        'cox_summary': summary,
        'cox_significant': significant
    }

def main(fast_mode=False):
    print("=" * 80)
    print("PHASE 0 - INITIALISATION")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {DATA_PATH}")

    # Paramètres d'exécution: mode rapide optionnel pour itération/debug.
    n_bootstrap_dml = 30 if fast_mode else DEFAULT_N_BOOTSTRAP
    n_bootstrap_mediation = 300 if fast_mode else 1000
    n_splits_dml = 2

    if fast_mode:
        print("Mode FAST activé: bootstraps réduits pour accélérer l'exécution")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Correction des types sur les colonnes connues numériques.
    numeric_candidates = sorted(set(numeric_features + ["Age_at_ICI_start", "OS_months", "PFS_", "OS_event", "PFS_Code"]))
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Nettoyage minimal des valeurs extrêmes/non finies.
    df = df.replace([np.inf, -np.inf], np.nan)
    all_links = []

    print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    print("\n" + "=" * 80)
    print("PHASE 1 - DÉCOUVERTE DE STRUCTURE (DML)")
    print("=" * 80)

    all_targets_to_process = []
    for target_level in range(1, 5):
        targets = [t for t in CAUSAL_LEVELS[target_level] if t in df.columns and t != "OS_months"]
        predictors = [
            v
            for lvl in range(0, target_level)
            for v in CAUSAL_LEVELS[lvl]
            if v in df.columns
        ]
        predictors = [p for p in dict.fromkeys(predictors)]

        if not targets or not predictors:
            continue

        print(f"Niveau {target_level}: {len(targets)} cible(s), {len(predictors)} prédicteur(s) candidats")
        for target in targets:
            all_targets_to_process.append((target, predictors, target_level))

    def _process_target_inference(target_info):
        """Traite un seul target pour l'inférence DML."""
        target, predictors, level = target_info
        try:
            print(f"  -> Traitement {target} (niveau {level})")
            inference_output = fit_cross_validated_inference(
                df=df,
                target=target,
                candidates=predictors,
                n_splits=n_splits_dml,
                n_bootstrap=n_bootstrap_dml,
                stability_threshold=STABILITY_THRESHOLD
            )
            return (target, predictors, inference_output)
        except Exception as exc:
            print(f"    Échec inférence {target}: {exc}")
            return (target, predictors, None)

    # Parallélisation au niveau des targets
    print(f"\nParallélisation de {len(all_targets_to_process)} inférences DML...")
    inference_results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_process_target_inference)(target_info)
        for target_info in all_targets_to_process
    )

    for target, predictors, inference_output in inference_results:
        if inference_output is None:
            continue

        agg_df = inference_output.get('aggregated_results', pd.DataFrame())
        if agg_df is None or agg_df.empty:
            continue

        # Approximation de stabilité: fréquence de sélection par feature sur les folds.
        fold_details = inference_output.get('fold_details', [])
        selected_lists = [fd.get('selected_features', []) for fd in fold_details if isinstance(fd, dict)]
        n_fold = max(len(selected_lists), 1)
        stability_map = {}
        for selected in selected_lists:
            for feat in selected:
                stability_map[feat] = stability_map.get(feat, 0) + 1
        stability_map = {k: v / n_fold for k, v in stability_map.items()}

        sig_df = agg_df[agg_df['p_value_fdr_bh'] < 0.05].copy() if 'p_value_fdr_bh' in agg_df.columns else pd.DataFrame()

        print(f"    {target}: {len(sig_df)} variables significatives FDR<0.05")
        for _, row in sig_df.iterrows():
            src = row.get('feature')
            if pd.isna(src):
                continue

            src = str(src)
            src_original = map_feature_to_original(src, predictors)

            all_links.append({
                'from': src_original,
                'from_encoded': src,
                'to': target,
                'coef': float(row.get('coef_mean', np.nan)),
                'p_value': float(row.get('p_value_fdr_bh', np.nan)),
                'e_value': float(row.get('e_value_point_mean', np.nan)),
                'stability': float(stability_map.get(src_original, np.nan)),
                'method': 'cross_fitted_dml'
            })

    print(f"\nLiens causaux détectés avant Cox: {len(all_links)}")

    print("\n" + "=" * 80)
    print("PHASE 2 - VALIDATION TEMPORELLE (COX)")
    print("=" * 80)

    cox_result = run_cox_validation(df, all_links, base_levels_to_adjust=[0, 1])
    if cox_result.get('status') == 'ok':
        print(f"Cox OK (n={cox_result['n_obs']}, significant_fdr={cox_result['n_significant_fdr']})")

        cox_sig = cox_result.get('cox_significant', pd.DataFrame())
        cox_map = {}
        if cox_sig is not None and not cox_sig.empty:
            for _, row in cox_sig.iterrows():
                orig = row.get('original_var')
                if pd.isna(orig):
                    continue
                cox_map[str(orig)] = {
                    'coef': float(row.get('coef', np.nan)),
                    'p_value': float(row.get('p_adj', np.nan))
                }

        # Mise à jour du graphe: les liens de survie sont conservés seulement s'ils passent Cox.
        audited_links = []
        for link in all_links:
            destination = link.get('to')
            source = link.get('from')

            if destination not in ['OS', 'OS_months', 'OS_event']:
                audited_links.append(link)
                continue

            if source in cox_map:
                link_updated = dict(link)
                link_updated['coef'] = cox_map[source]['coef']
                link_updated['p_value'] = cox_map[source]['p_value']
                link_updated['to'] = 'Overall_Survival'
                link_updated['method'] = 'validated_by_cox'
                audited_links.append(link_updated)

        all_links = audited_links
    else:
        print(f"Cox skipped: {cox_result.get('reason', 'raison inconnue')}")

    print(f"Liens causaux après audit Cox: {len(all_links)}")

    print("\n" + "=" * 80)
    print("PHASE 3 - MÉDIATION BOOTSTRAP")
    print("=" * 80)

    mediation_result = None
    vaccine_col = 'Vaccine100' if 'Vaccine100' in df.columns else None
    mediator_col = 'PFS_' if 'PFS_' in df.columns else None
    outcome_col = 'OS_event' if 'OS_event' in df.columns else None

    if vaccine_col and mediator_col and outcome_col:
        mediation_result = bootstrap_mediation_robust(
            df=df,
            cause=vaccine_col,
            mediator=mediator_col,
            outcome=outcome_col,
            n_boot=n_bootstrap_mediation,
            random_seed=RANDOM_SEED
        )

        if mediation_result is not None:
            p_indirect = mediation_result['indirect_effect']['p_empirical']
            ci_low = mediation_result['indirect_effect']['ci_low']
            ci_high = mediation_result['indirect_effect']['ci_high']
            print(f"Médiation {vaccine_col} -> {mediator_col} -> {outcome_col}: p={p_indirect:.4f}, IC95%=[{ci_low:.4f}, {ci_high:.4f}]")
        else:
            print("Médiation non concluante (aucun bootstrap valide)")
    else:
        print("Médiation non exécutée (colonnes manquantes)")

    print("\n" + "=" * 80)
    print("PHASE 4 - EXPORT")
    print("=" * 80)

    export_path = SCRIPT_DIR / "../data/reinforced_causal_dag_structured.csv"
    links_df = pd.DataFrame(all_links)
    links_df.to_csv(export_path, index=False, encoding='utf-8-sig')

    print(f"Export CSV: {export_path}")
    print(f"Liens exportés: {len(links_df)}")
    print("Analyse causale terminée avec succès.")

    return {
        'links_df': links_df,
        'cox_result': cox_result,
        'mediation_result': mediation_result,
        'export_path': str(export_path)
    }


if __name__ == "__main__":
    main()
