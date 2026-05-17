import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from types import SimpleNamespace
import random

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, ElasticNetCV
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
import statsmodels.api as sm
from tabulate import tabulate


from joblib import Parallel, delayed

pd.set_option('display.max_columns', None)


def log_exception(context, exc):
    """Log an explicit exception message without stopping the caller.

    Parameters
    ----------
    context : str
        Human-readable context describing where the error happened.
    exc : Exception
        Exception instance to report.
    """
    print(f"[ERROR] {context} | {type(exc).__name__}: {exc}")


def log_warning(context, exc):
    """Log a warning message when a degraded path is used.

    Parameters
    ----------
    context : str
        Human-readable context describing where the warning happened.
    exc : Exception
        Exception instance to report.
    """
    print(f"[WARN] {context} | {type(exc).__name__}: {exc}")


# -------------------------------------------------------------------
# GLOBAL CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------
DEFAULT_N_BOOTSTRAP = 100       # Number of stability-selection bootstrap iterations
STABILITY_THRESHOLD = 0.65      # Selection threshold (feature must appear in 65% of bootstraps)
RANDOM_SEED = 42

# Solver hyperparameters
LOGIT_CV_MAX_ITER = 8000
LOGIT_CV_TOL = 3e-3
LOGIT_CV_CS = np.logspace(-1, 1, 5)
LOGIT_CV_L1_RATIOS = [0.5]
OHE_MIN_FREQUENCY = 0.02
OHE_MAX_CATEGORIES = 10
RARE_CATEGORY_MIN_COUNT = 20

# Solver mode parameters
USE_LOGIT_CV_IN_STABILITY = False
LOGIT_STABILITY_C = 1.0
LOGIT_STABILITY_L1_RATIO = 0.5
BINARY_STABILITY_MODEL = "liblinear_l1"  # options: "saga_elasticnet", "liblinear_l1", "lbfgs_l2"

# Parallelism parameters
TARGET_PARALLEL_N_JOBS = 1
FOLD_PARALLEL_N_JOBS = 1
BOOTSTRAP_PARALLEL_N_JOBS = 1

# Statsmodels parameters
STATSMODELS_LOGIT_ALPHA = 0.05
STATSMODELS_LOGIT_MAXITER = 2000
ENABLE_BINARY_MLE_INFERENCE = True
MAX_LOG_EXP_INPUT = 20.0

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../data/combined_covid_data.csv"

# -------------------------------------------------------------------
# CAUSAL STRUCTURE DEFINITION
# -------------------------------------------------------------------
categorical_features = ["Ethnicity", "Gender", "Immunotherapy_Agent","Simplified_Stage"]
numeric_features = ["BRAF","CNS_disease","Concurrent_Chemo","ECOG",
                    "English_as_primary_language", "Previous_history_of_malignancy_at_ICI_start",
                    "Steroid_win_1_month_of_ICI_start","Steroid_win_1_month_of_Vaccine",
                    "Vaccine100", "Age_at_ICI_start"
                    ]
target_variables = ["OS_event", "OS_months"]

CAUSAL_LEVELS = {
    # Level 0: Demographic variables
    0: ["Gender", "Ethnicity", "Age_at_ICI_start", "English_as_primary_language"],
    
    # Level 1: Baseline disease characteristics
    1: ["Simplified_Stage", "ECOG", "CNS_disease", "Previous_history_of_malignancy_at_ICI_start"],
    
    # Level 2: Treatments and interventions
    2: ["Vaccine100", "Steroid_win_1_month_of_Vaccine", "Concurrent_Chemo", "BRAF", "Immunotherapy_Agent", "Steroid_win_1_month_of_ICI_start"],
    
    # Level 3: Final outcomes
    3: ["OS_months", "OS_event"]
}

# Reverse mapping for fast lookups
VAR_TO_LEVEL = {}
for level, vars_list in CAUSAL_LEVELS.items():
    for var in vars_list:
        VAR_TO_LEVEL[var] = level


def map_feature_to_original(feature_name, candidate_variables):
    """Map an encoded feature name back to its original variable.

    Parameters
    ----------
    feature_name : str
        Feature name potentially expanded by one-hot encoding
        (e.g. ``Gender_Male``).
    candidate_variables : list[str]
        Candidate source variables used before encoding.

    Returns
    -------
    str
        Original variable name when a prefix match is found, otherwise
        the input ``feature_name``.
    """
    if feature_name in candidate_variables:
        return feature_name

    for base in sorted(candidate_variables, key=len, reverse=True):
        if feature_name.startswith(base + "_"):
            return base

    return feature_name


def coerce_numeric_predictors(df_in, candidate_variables):
    """Cast known numeric predictors to float with safe coercion.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input dataframe.
    candidate_variables : list[str]
        Candidate predictors to inspect.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df_in`` with numeric candidates converted using
        ``errors='coerce'``.
    """
    out = df_in.copy()
    numeric_in_data = [c for c in candidate_variables if c in out.columns and c in numeric_features]
    for col in numeric_in_data:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def normalize_categorical_predictors(df_in, candidate_variables, min_count=RARE_CATEGORY_MIN_COUNT):
    """Normalize categorical levels and group rare categories.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input dataframe.
    candidate_variables : list[str]
        Candidate predictors to inspect.
    min_count : int, default=RARE_CATEGORY_MIN_COUNT
        Minimum count for a category to remain explicit.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df_in`` with cleaned categorical values.
    """
    out = df_in.copy()
    cat_in_data = [c for c in candidate_variables if c in out.columns and c in categorical_features]

    if 'Gender' in cat_in_data:
        g = out['Gender'].astype('string').str.strip()
        out['Gender'] = g.replace({'M': 'Male', 'm': 'Male', 'F': 'Female', 'f': 'Female'})

    for col in cat_in_data:
        s = out[col].astype('string').str.strip()
        counts = s.value_counts(dropna=True)
        rare_levels = set(counts[counts < min_count].index)
        if rare_levels:
            s = s.where(~s.isin(rare_levels), other='__RARE__')
        out[col] = s

    return out

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def compute_e_value(coef, se, is_binary=True):
    """Compute E-values from model estimates and standard errors.

    Parameters
    ----------
    coef : float
        Model coefficient. For binary outcomes, expected on log-ratio scale.
    se : float
        Standard error associated with ``coef``.
    is_binary : bool, default=True
        If ``True``, interpret ``coef`` as log(OR/HR-like). If ``False``,
        use the SMD-to-RR approximation.

    Returns
    -------
    dict[str, float]
        Dictionary with RR-like point/interval values and associated
        E-values.

    Raises
    ------
    ValueError
        If ``coef`` is not finite or ``se`` is invalid.

    Notes
    -----
    Uses the formulation from VanderWeele and Ding (2017).
    """
    # Minimal validation to avoid non-interpretable results.
    if not np.isfinite(coef):
        raise ValueError("coef must be a finite number")
    if (se is None) or (not np.isfinite(se)) or (se < 0):
        raise ValueError("se must be a finite number >= 0")

    def _evalue_from_ratio(rr):
        """Compute E-value from a ratio-like estimate (RR/OR/HR-like)."""
        if rr <= 0 or not np.isfinite(rr):
            return np.nan
        rr_ref = rr if rr >= 1 else 1.0 / rr
        if rr_ref <= 1:
            return 1.0
        return float(rr_ref + np.sqrt(rr_ref * (rr_ref - 1.0)))

    # Wald 95% CI: beta ± 1.96 * SE.
    z = 1.96

    def _safe_exp(x):
        return float(np.exp(np.clip(x, -MAX_LOG_EXP_INPUT, MAX_LOG_EXP_INPUT)))

    if is_binary:
        # Logit/Cox case: coefficient is on the log-ratio scale.
        rr_point = _safe_exp(coef)
        rr_ci_low = _safe_exp(coef - z * se)
        rr_ci_high = _safe_exp(coef + z * se)
    else:
        # Continuous case: SMD -> RR-like approximation.
        # (VanderWeele & Ding, 2017): RR ≈ exp(0.91 * SMD)
        rr_point = _safe_exp(0.91 * coef)
        rr_ci_low = _safe_exp(0.91 * (coef - z * se))
        rr_ci_high = _safe_exp(0.91 * (coef + z * se))

    e_value_point = _evalue_from_ratio(rr_point)

    # Conservative E-value: use the CI bound closest to the null (RR=1).
    if rr_ci_low <= 1.0 <= rr_ci_high:
        e_value_ci = 1.0
    else:
        if rr_point >= 1.0:
            nearest_bound = rr_ci_low
        else:
            nearest_bound = rr_ci_high
        e_value_ci = _evalue_from_ratio(nearest_bound)

    # Cap values to avoid absurd numbers from near-perfect or structural associations
    if e_value_point > 100.0:
        e_value_point = 100.0
    if e_value_ci > 100.0:
        e_value_ci = 100.0

    return {
        'rr_point': rr_point,
        'rr_ci_low': rr_ci_low,
        'rr_ci_high': rr_ci_high,
        'e_value_point': e_value_point,
        'e_value_ci': e_value_ci
    }

def fit_unpenalized_model(df_train, features, is_binary, target="OS_event"):
    """Fit inference-ready models on pre-selected features.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training subset used for inference.
    features : list[str]
        Features selected by stability selection.
    is_binary : bool
        Whether the target is binary.
    target : str, default="OS_event"
        Target column name.

    Returns
    -------
    dict
        Model metadata and a per-feature result table. For binary targets,
        coefficients come from a stable regularized fit and inferential
        statistics come from a GLM fit when available.

    Raises
    ------
    ValueError
        If ``target`` is missing or a binary target does not contain exactly
        two classes after cleaning.
    """
    # Input checks to keep the cross-fit pipeline robust.
    if target not in df_train.columns:
        raise ValueError(f"Target '{target}' is missing from the DataFrame")

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

    # Keep only required columns, then drop missing values.
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
        # For binary targets, force a 0/1 encoding for statsmodels.Logit.
        unique_vals = sorted(pd.unique(y))
        if len(unique_vals) != 2:
            raise ValueError(
                f"Binary target '{target}' must contain exactly 2 classes after cleaning; got: {unique_vals}"
            )
        if set(unique_vals) != {0, 1}:
            y = y.map({unique_vals[0]: 0, unique_vals[1]: 1})

    # Encode predictors with one-hot expansion; drop first to avoid dummy trap.
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

    # Add intercept before matrix diagnostics.
    X = add_constant(X, has_constant='add')

    # Check multicollinearity and extreme values.
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] > 0:
        # Drop zero-variance columns (true multicollinearity issue).
        zero_var_cols = [c for c in X_numeric.columns if c != 'const' and X_numeric[c].var() < 1e-10]
        if len(zero_var_cols) > 0:
            X = X.drop(columns=zero_var_cols)
            print(f"   Info: Zero-variance columns dropped (multicollinearity): {list(zero_var_cols)}")

        # Check condition number (diagnostic indicator, not always critical).
        try:
            cond_number = np.linalg.cond(X.values)
            # High condition number may occur with sparse dummy structures.
            if cond_number > 1e15:
                print(f"   Warning: Near-singular matrix detected (cond={cond_number:.2e})")
        except Exception as exc:
            log_warning(f"cond_number unavailable for {target}", exc)

    try:
        # Separate coefficient estimation from inferential diagnostics.

        if is_binary:
            # Regularized logistic fit for robust coefficients.
            clf = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                l1_ratio=1.0,
                C=LOGIT_STABILITY_C,
                max_iter=LOGIT_CV_MAX_ITER,
                tol=LOGIT_CV_TOL,
                class_weight='balanced',
                fit_intercept=False,
                random_state=RANDOM_SEED,
            )
            clf.fit(X.values, y.values)
            coef_model = SimpleNamespace(params=pd.Series(clf.coef_.ravel(), index=X.columns))

            # Optional GLM inference (SE/p-values/CI) on the same design matrix.
            if ENABLE_BINARY_MLE_INFERENCE:
                try:
                    with warnings.catch_warnings(record=True) as caught_glm:
                        warnings.simplefilter('always', category=RuntimeWarning)
                        inference_model = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=300)
                    if caught_glm:
                            print(f"   Warning: GLM inference emitted {len(caught_glm)} warning(s) for {target}")
                    conf_int = inference_model.conf_int()
                    has_inference = True
                except Exception as exc:
                    log_warning(f"MLE fit failed for {target}; p-values unavailable", exc)
                    inference_model = None
                    conf_int = None
                    has_inference = False
            else:
                inference_model = None
                conf_int = None
                has_inference = False
        else:
            # OLS provides both coefficients and inference directly.
            coef_model = OLS(y, X).fit()
            inference_model = coef_model
            conf_int = coef_model.conf_int()
            has_inference = True  # Inference is directly available with OLS.

        rows = []

        # Iterate over variables to extract coefficients and statistics.
        for param in coef_model.params.index:
            if param == 'const':
                continue

            beta = float(coef_model.params[param])

            # Guard against non-interpretable extreme coefficients.
            if not np.isfinite(beta) or abs(beta) > 50:
                print(f"   Warning: Extreme coefficient for {param}: {beta:.2f}")
                continue

            if has_inference and hasattr(inference_model, 'bse'):
                se = float(inference_model.bse[param])
                pval = float(inference_model.pvalues[param])
                ci_low = float(conf_int.loc[param, 0])
                ci_high = float(conf_int.loc[param, 1])

                # Keep the coefficient even if inferential terms are invalid.
                if not np.isfinite(se) or se <= 0:
                    print(f"   Info: Invalid SE for {param}: {se} (coef={beta:.3f} kept)")
                    se = np.nan
                    pval = np.nan
                    ci_low = np.nan
                    ci_high = np.nan
            else:
                # Inference unavailable on fallback path.
                se = np.nan
                pval = np.nan
                ci_low = np.nan
                ci_high = np.nan

            # Transform with exp(coef) for interpretable OR/HR-like values.
            if is_binary:
                effect_point = float(np.exp(np.clip(beta, -20, 20)))
                effect_ci_low = float(np.exp(np.clip(ci_low, -20, 20)))
                effect_ci_high = float(np.exp(np.clip(ci_high, -20, 20)))
            else:
                effect_point = beta
                effect_ci_low = ci_low
                effect_ci_high = ci_high

            # E-value: sensitivity to an unmeasured confounder.
            # Compute only when SE is available and valid.
            if np.isfinite(se) and se > 0:
                try:
                    e_val = compute_e_value(beta, se, is_binary=is_binary)
                except Exception as exc:
                    log_warning(f"compute_e_value failed for {target}/{param}", exc)
                    # Fallback if compute_e_value fails.
                    e_val = {
                        'rr_point': np.nan, 'rr_ci_low': np.nan, 'rr_ci_high': np.nan,
                        'e_value_point': np.nan, 'e_value_ci': np.nan
                    }
            else:
                # Invalid SE -> E-value unavailable.
                e_val = {
                    'rr_point': np.nan, 'rr_ci_low': np.nan, 'rr_ci_high': np.nan,
                    'e_value_point': np.nan, 'e_value_ci': np.nan
                }

            # One row per variable with all summary statistics.
            rows.append({
                'feature': param,
                'feature_original': map_feature_to_original(param, selected_features),
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
        log_exception(f"Model fit failed for {target} with {len(selected_features)} features", e)
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
    return {
        'target': target,
        'is_binary': is_binary,
        'n_obs': int(len(y)),
        'n_features_input': len(selected_features),
        'n_features_encoded': int(X.shape[1] - 1),
        'model': coef_model,
        'results_df': results_df
    }


def aggregate_results(results_list):
    """Aggregate fold-level inference into one per-feature summary.

    Parameters
    ----------
    results_list : list[dict]
        Outputs produced by :func:`fit_unpenalized_model` across folds.

    Returns
    -------
    pandas.DataFrame
        Aggregated table with Fisher-combined p-values and FDR-adjusted
        significance.
    """
    # Aggregate only folds that produced actual results.
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

    # Long table: one row = (feature, fold).
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
    group_col = 'feature_original' if 'feature_original' in stacked.columns else 'feature'
    for feature, grp in stacked.groupby(group_col, dropna=True):
        grp = grp.copy()
        
        # Extraire is_binary du groupe actuel (partagé par tous les folds du même target)
        if 'is_binary' in grp.columns and grp['is_binary'].notna().any():
            is_binary = bool(grp['is_binary'].dropna().iloc[0])
        else:
            is_binary = True
        
        pvals = grp['p_value'].dropna().to_numpy(dtype=float)

        if len(pvals) == 0:
            p_fisher = np.nan
            fisher_stat = np.nan
        else:
            # Fisher method: combine statistical evidence across folds.
            pvals = np.clip(pvals, 1e-300, 1.0)
            fisher_stat, p_fisher = combine_pvalues(pvals, method='fisher')
            fisher_stat = float(fisher_stat)
            p_fisher = float(p_fisher)

        coef_vals = grp['coef'].dropna().to_numpy(dtype=float)
        se_vals = grp['se'].dropna().to_numpy(dtype=float)
        
        n_folds = len(coef_vals)
        coef_mean = float(np.mean(coef_vals)) if n_folds > 0 else np.nan
        
        # Pooled SE for the mean of independent folds: SE_avg = sqrt(sum(SE_i^2)) / K
        if len(se_vals) > 0:
            pooled_se = np.sqrt(np.sum(se_vals**2)) / len(se_vals)
        else:
            pooled_se = np.nan

        # Recalculate consensus E-value and Effect size from aggregated parameters
        if np.isfinite(coef_mean) and np.isfinite(pooled_se) and pooled_se > 0:
            try:
                e_val_final = compute_e_value(coef_mean, pooled_se, is_binary=is_binary)
            except Exception:
                e_val_final = {'rr_point': np.nan, 'e_value_point': np.nan, 'e_value_ci': np.nan}
        else:
            # Fallback for point estimate only if SE is missing
            if is_binary and np.isfinite(coef_mean):
                rr_pt = float(np.exp(np.clip(coef_mean, -20, 20)))
            else:
                rr_pt = coef_mean
            e_val_final = {'rr_point': rr_pt, 'e_value_point': np.nan, 'e_value_ci': np.nan}

        aggregated_rows.append({
            'feature': feature,
            'feature_example_encoded': grp['feature'].dropna().iloc[0] if grp['feature'].notna().any() else feature,
            'target': grp['target'].dropna().iloc[0] if grp['target'].notna().any() else None,
            'is_binary': is_binary,
            'n_folds_present': int(grp['fold_id'].nunique()),
            'fisher_stat': fisher_stat,
            'p_value_fisher': p_fisher,
            'coef_mean': coef_mean,
            'pooled_se': pooled_se,
            'effect_point_mean': e_val_final['rr_point'],
            'e_value_point': e_val_final['e_value_point'],
            'e_value_ci': e_val_final['e_value_ci']
        })

    out = pd.DataFrame(aggregated_rows)
    if out.empty:
        return out

    valid_mask = out['p_value_fisher'].notna()
    out['p_value_fdr_bh'] = np.nan
    out['is_significant_fdr_5pct'] = False

    if valid_mask.any():
        # Control false discovery rate across all features.
        reject, pvals_corr, _, _ = multipletests(
            out.loc[valid_mask, 'p_value_fisher'].values,
            alpha=0.05,
            method='fdr_bh'
        )
        out.loc[valid_mask, 'p_value_fdr_bh'] = pvals_corr
        out.loc[valid_mask, 'is_significant_fdr_5pct'] = reject

    # Final output sorted by combined statistical signal.
    out = out.sort_values(['p_value_fisher', 'feature'], na_position='last').reset_index(drop=True)

    return out

def bootstrap_mediation_robust(df, cause, mediator, outcome, n_boot=1000, random_seed=42):
    """Estimate mediation effects using bootstrap resampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    cause : str
        Exposure variable.
    mediator : str
        Mediator variable.
    outcome : str
        Outcome variable.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict or None
        Mediation summary with indirect, direct, and total effects. Returns
        ``None`` if no valid bootstrap estimate is available.

    Raises
    ------
    ValueError
        If requested columns are missing or ``n_boot <= 0``.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")

    for col in [cause, mediator, outcome]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Keep useful subset and perform base cleaning.
    data = df[[cause, mediator, outcome]].copy()
    data = data.dropna()
    if data.empty:
        return None

    def _coerce_to_numeric(series):
        """Convert series to numeric; map binary categories to 0/1 when possible."""
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

    # Binary values must be encoded as 0/1 for Logit.
    if is_mediator_binary and set(pd.unique(data[mediator])) != {0, 1}:
        vals = sorted(pd.unique(data[mediator]))
        data[mediator] = data[mediator].map({vals[0]: 0.0, vals[1]: 1.0})

    if is_outcome_binary and set(pd.unique(data[outcome])) != {0, 1}:
        vals = sorted(pd.unique(data[outcome]))
        data[outcome] = data[outcome].map({vals[0]: 0.0, vals[1]: 1.0})

    if data[cause].nunique() < 2 or data[mediator].nunique() < 2 or data[outcome].nunique() < 2:
        return None

    # Detection of structural dependence (perfect separation)
    is_path_a_structural = False
    if is_mediator_binary:
        xtab_a = pd.crosstab(data[cause], data[mediator])
        if (xtab_a == 0).any().any():
            is_path_a_structural = True

    is_path_b_structural = False
    if is_outcome_binary:
        xtab_b = pd.crosstab(data[mediator], data[outcome])
        if (xtab_b == 0).any().any():
            is_path_b_structural = True

    rng = np.random.default_rng(random_seed)

    def _single_mediation_boot(boot_seed, data_input, cause_col, mediator_col, outcome_col, is_med_binary, is_out_binary):
        """Run one bootstrap iteration for mediation."""
        rng_boot = np.random.default_rng(boot_seed)
        boot_idx = rng_boot.integers(low=0, high=len(data_input), size=len(data_input))
        df_boot = data_input.iloc[boot_idx].copy()

        try:
            # Model a: mediator ~ cause
            X_a = add_constant(df_boot[[cause_col]], has_constant='add')
            y_a = df_boot[mediator_col]
            if is_med_binary:
                # Use regularization to reduce perfect separation issues.
                model_a = Logit(y_a, X_a).fit_regularized(
                    method='l1',
                    alpha=STATSMODELS_LOGIT_ALPHA,
                    disp=False,
                    maxiter=STATSMODELS_LOGIT_MAXITER,
                    trim_mode='size'
                )
            else:
                model_a = OLS(y_a, X_a).fit()

            a_coef = float(model_a.params[cause_col])
            if not np.isfinite(a_coef) or abs(a_coef) > 50:
                return {'success': False}

            # Model b/c': outcome ~ cause + mediator
            X_b = add_constant(df_boot[[cause_col, mediator_col]], has_constant='add')
            y_b = df_boot[outcome_col]
            if is_out_binary:
                model_b = Logit(y_b, X_b).fit_regularized(
                    method='l1',
                    alpha=STATSMODELS_LOGIT_ALPHA,
                    disp=False,
                    maxiter=STATSMODELS_LOGIT_MAXITER,
                    trim_mode='size'
                )
            else:
                model_b = OLS(y_b, X_b).fit()

            b_coef = float(model_b.params[mediator_col])
            c_prime = float(model_b.params[cause_col])

            if not np.isfinite(b_coef) or not np.isfinite(c_prime) or abs(b_coef) > 50 or abs(c_prime) > 50:
                return {'success': False}

            # Total model c: outcome ~ cause
            X_c = add_constant(df_boot[[cause_col]], has_constant='add')
            if is_out_binary:
                model_c = Logit(y_b, X_c).fit_regularized(
                    method='l1',
                    alpha=STATSMODELS_LOGIT_ALPHA,
                    disp=False,
                    maxiter=STATSMODELS_LOGIT_MAXITER,
                    trim_mode='size'
                )
            else:
                model_c = OLS(y_b, X_c).fit()
            c_total = float(model_c.params[cause_col])

            if not np.isfinite(c_total) or abs(c_total) > 50:
                return {'success': False}

            return {
                'a': a_coef,
                'b': b_coef,
                'indirect': a_coef * b_coef,
                'direct': c_prime,
                'total': c_total,
                'success': True
            }
        except Exception as exc:
            log_exception(
                f"Bootstrap mediation failed (seed={boot_seed}, path={cause_col}->{mediator_col}->{outcome_col})",
                exc,
            )
            return {'success': False}

    print(f"Bootstrap mediation {cause}->{mediator}->{outcome}: {n_boot} iterations (parallelized)")
    seeds = rng.integers(0, 100000, size=n_boot)

    # Parallel execution with joblib.
    results = Parallel(n_jobs=BOOTSTRAP_PARALLEL_N_JOBS, backend='threading')(
        delayed(_single_mediation_boot)(seed, data, cause, mediator, outcome, is_mediator_binary, is_outcome_binary)
        for seed in seeds
    )

    a_effects = []
    b_effects = []
    indirect_effects = []
    direct_effects = []
    total_effects = []
    n_failed = 0

    for res in results:
        if res.get('success', False):
            a_effects.append(res['a'])
            b_effects.append(res['b'])
            indirect_effects.append(res['indirect'])
            direct_effects.append(res['direct'])
            total_effects.append(res['total'])
        else:
            n_failed += 1

    if len(indirect_effects) == 0:
        return None

    def _summarize_bootstrap(samples, is_binary_target=True):
        arr = np.asarray(samples, dtype=float)
        mean_val = float(np.mean(arr))
        sd_val = float(np.std(arr))
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        prop_pos = float(np.mean(arr > 0))
        p_emp = float(2 * min(prop_pos, 1 - prop_pos))

        # E-value: treat bootstrap SD as SE for robustness check
        try:
            # We only compute E-value if SD is valid
            if sd_val > 0:
                e_val = compute_e_value(mean_val, sd_val, is_binary=is_binary_target)
            else:
                e_val = {'e_value_point': 1.0, 'e_value_ci': 1.0}
        except Exception:
            e_val = {'e_value_point': 1.0, 'e_value_ci': 1.0}

        return {
            'mean': mean_val,
            'sd': sd_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'prop_positive': prop_pos,
            'p_empirical': p_emp,
            'e_value_point': e_val.get('e_value_point', 1.0),
            'e_value_ci': e_val.get('e_value_ci', 1.0)
        }

    path_a_stats = _summarize_bootstrap(a_effects, is_mediator_binary)
    if is_path_a_structural:
        path_a_stats['e_value_point'] = 10.0
        path_a_stats['e_value_ci'] = 10.0
        path_a_stats['is_structural'] = True

    path_b_stats = _summarize_bootstrap(b_effects, is_outcome_binary)
    if is_path_b_structural:
        path_b_stats['e_value_point'] = 10.0
        path_b_stats['e_value_ci'] = 10.0
        path_b_stats['is_structural'] = True

    indirect_stats = _summarize_bootstrap(indirect_effects, is_outcome_binary)
    direct_stats = _summarize_bootstrap(direct_effects, is_outcome_binary)
    total_stats = _summarize_bootstrap(total_effects, is_outcome_binary)

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
        'path_a': path_a_stats,
        'path_b': path_b_stats,
        'indirect_effect': indirect_stats,
        'direct_effect': direct_stats,
        'total_effect': total_stats,
        'proportion_mediated_mean': prop_mediated,
        'n_boot_requested': int(n_boot),
        'n_boot_success': int(len(indirect_effects)),
        'n_boot_failed': int(n_failed)
    }

# -------------------------------------------------------------------
# MAIN FUNCTIONS
# -------------------------------------------------------------------
def run_stability_selection(df, candidates, target="OS_event", n_bootstrap=100, threshold=0.6, random_state=42):
    """Run bootstrap-based stability selection for one target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    candidates : list[str]
        Candidate predictors.
    target : str, default="OS_event"
        Target variable.
    n_bootstrap : int, default=100
        Number of bootstrap iterations.
    threshold : float, default=0.6
        Minimum selection frequency required to retain a predictor.
    random_state : int, default=42
        Seed for random sampling.

    Returns
    -------
    dict
        Selected features, per-feature stability scores, and number of
        successful iterations.
    """
    cols_to_use = candidates + [target]
    data = df[cols_to_use].copy()
    data = coerce_numeric_predictors(data, candidates)
    data = normalize_categorical_predictors(data, candidates)
    data = data.dropna()

    if len(data) < 50:
        print(f"   Too few samples for {target} (n={len(data)}). Skip.")
        return {'selected_features': [], 'stability_scores': {}, 'n_iterations': 0}

    X = data[candidates]
    y = data[target]

    # Detect target type to choose an appropriate selection model.
    is_binary_target = y.nunique() == 2

    cat_candidates = [c for c in categorical_features if c in candidates]
    num_candidates = [c for c in numeric_features if c in candidates]

    # Avoid silently ignoring variables not listed in static feature groups.
    remaining_candidates = [c for c in candidates if c not in set(cat_candidates + num_candidates)]
    for col in remaining_candidates:
        s_num = pd.to_numeric(data[col], errors='coerce')
        if s_num.notna().mean() >= 0.95:
            data[col] = s_num
            num_candidates.append(col)
        else:
            cat_candidates.append(col)

    # ElasticNetCV requires dense design; logistic path supports sparse OHE.
    cat_sparse = True if is_binary_target else False

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_candidates),
            ('cat', OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                min_frequency=OHE_MIN_FREQUENCY,
                max_categories=OHE_MAX_CATEGORIES,
                sparse_output=cat_sparse,
                drop='first'
            ),
             cat_candidates)
        ],
        verbose_feature_names_out=True
    )

    if is_binary_target:
        if USE_LOGIT_CV_IN_STABILITY:
            model = LogisticRegressionCV(
                cv=2,
                penalty='elasticnet',
                solver='saga',
                l1_ratios=LOGIT_CV_L1_RATIOS,
                Cs=LOGIT_CV_CS,
                scoring='roc_auc',
                max_iter=LOGIT_CV_MAX_ITER,
                tol=LOGIT_CV_TOL,
                class_weight='balanced',
                n_jobs=1,
                random_state=random_state
            )
        else:
            if BINARY_STABILITY_MODEL == "liblinear_l1":
                # Very stable for binary targets, with natural sparsity and no SAGA warnings.
                model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=LOGIT_STABILITY_C,
                    max_iter=LOGIT_CV_MAX_ITER,
                    tol=LOGIT_CV_TOL,
                    class_weight='balanced',
                    random_state=random_state
                )
            elif BINARY_STABILITY_MODEL == "lbfgs_l2":
                model = LogisticRegression(
                    penalty='l2',
                    solver='lbfgs',
                    C=LOGIT_STABILITY_C,
                    max_iter=LOGIT_CV_MAX_ITER,
                    tol=LOGIT_CV_TOL,
                    class_weight='balanced',
                    random_state=random_state
                )
            else:
                model = LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    C=LOGIT_STABILITY_C,
                    l1_ratio=LOGIT_STABILITY_L1_RATIO,
                    max_iter=LOGIT_CV_MAX_ITER,
                    tol=LOGIT_CV_TOL,
                    class_weight='balanced',
                    n_jobs=1,
                    random_state=random_state
                )
    else:
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=3,
            max_iter=5000,
            n_jobs=1,
            random_state=random_state
        )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    def _map_encoded_to_original(encoded_name, candidates_list):
        """Map transformed feature names back to source variables."""
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

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 100000, size=n_bootstrap)

    def _single_stability_iteration(seed, X_input, y_input, pipeline_obj, candidates_list):
        """Run one stability-selection iteration."""
        try:
            X_resampled, y_resampled = resample(
                X_input, y_input, replace=True, n_samples=len(X_input), random_state=seed
            )

            if y_resampled.nunique() < 2:
                return {'success': False, 'selected': set(), 'reason': 'single_class', 'n_conv_warn': 0, 'conv_msg': None}

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                pipeline_obj.fit(X_resampled, y_resampled)

            conv_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
            conv_msg = str(conv_warnings[0].message) if conv_warnings else None
            feature_names = pipeline_obj.named_steps['preprocessor'].get_feature_names_out()
            model_fitted = pipeline_obj.named_steps['model']
            coefs = np.asarray(model_fitted.coef_).ravel()

            selected_encoded = feature_names[np.abs(coefs) > 1e-6]
            selected_original = set()

            for enc in selected_encoded:
                orig = _map_encoded_to_original(enc, candidates_list)
                if orig is not None:
                    selected_original.add(orig)

            return {
                'success': True,
                'selected': selected_original,
                'reason': None,
                'n_conv_warn': len(conv_warnings),
                'conv_msg': conv_msg,
            }
        except Exception as exc:
            log_exception(f"Stability selection failed for {target} (seed={seed})", exc)
            return {'success': False, 'selected': set(), 'reason': 'exception', 'n_conv_warn': 0, 'conv_msg': None}

    model_name = "LogitCV(elasticnet)" if is_binary_target else "ElasticNetCV"
    if is_binary_target:
        if USE_LOGIT_CV_IN_STABILITY:
            model_name = "LogitCV(elasticnet)"
        else:
            model_name = f"Logit({BINARY_STABILITY_MODEL})"
    else:
        model_name = "ElasticNetCV"
    print(f"   Running Stability Selection for target: {target} with {n_bootstrap} bootstraps [{model_name}]...")

    # Parallel execution with joblib.
    results = Parallel(n_jobs=BOOTSTRAP_PARALLEL_N_JOBS, backend='threading')(
        delayed(_single_stability_iteration)(seed, X, y, clone(pipeline), candidates)
        for seed in seeds
    )

    selection_counts = {feature: 0 for feature in candidates}
    n_successful_runs = 0
    n_empty_selected = 0
    n_failed_runs = 0
    n_conv_warning_runs = 0
    n_conv_warning_total = 0
    conv_examples = {}

    for item in results:
        n_conv = int(item.get('n_conv_warn', 0)) if isinstance(item, dict) else 0
        if n_conv > 0:
            n_conv_warning_runs += 1
            n_conv_warning_total += n_conv
            msg = item.get('conv_msg')
            if msg:
                conv_examples[msg] = conv_examples.get(msg, 0) + 1

        if not isinstance(item, dict) or not item.get('success', False):
            n_failed_runs += 1
            continue

        n_successful_runs += 1
        selected_original = item.get('selected', set())
        if not selected_original:
            n_empty_selected += 1
            continue

        for feature in selected_original:
            selection_counts[feature] += 1

    if n_successful_runs == 0:
        print(f"   No successful iterations for {target}.")
        return {'selected_features': [], 'stability_scores': {}, 'n_iterations': 0}

    stability_scores = {feature: count / n_successful_runs for feature, count in selection_counts.items()}
    selected_features = [feature for feature, score in stability_scores.items() if score >= threshold]

    print(
        f"   Selected {len(selected_features)} features for {target} (threshold={threshold}): {selected_features}"
    )
    print(
            f"   Stability stats {target}: success={n_successful_runs}, empty={n_empty_selected}, failed={n_failed_runs}"
    )
    if n_conv_warning_total > 0:
        print(
            f"   Convergence warnings {target}: runs={n_conv_warning_runs}, total={n_conv_warning_total}"
        )
        for msg, count in sorted(conv_examples.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            print(f"      - x{count}: {msg}")
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
    """Run cross-fitted variable selection and inference.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset.
    target : str
        Target variable.
    candidates : list[str]
        Candidate predictors.
    n_splits : int, default=2
        Number of folds.
    n_bootstrap : int, default=DEFAULT_N_BOOTSTRAP
        Number of bootstrap iterations used in selection.
    stability_threshold : float, default=STABILITY_THRESHOLD
        Stability cut-off used to retain features.

    Returns
    -------
    dict
        Cross-fit metadata, fold details, and aggregated feature-level
        inference.

    Raises
    ------
    ValueError
        If ``target`` is missing or ``n_splits < 2``.
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is missing from the DataFrame")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

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

    # Shared cleaned dataset used by all cross-fitting steps.
    cols = available_candidates + [target]
    data = df[cols].copy()
    data = coerce_numeric_predictors(data, available_candidates)
    data = normalize_categorical_predictors(data, available_candidates)
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

    # Stratify binary outcomes to preserve class balance.
    if is_binary:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_indices = list(splitter.split(data, y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_indices = list(splitter.split(data))

    fold_results_for_aggregation = []
    fold_details = []

    def _process_fold(fold_idx, avail_cand, df_data, is_bin, split_inds, target_col, n_boot, stab_thresh):
        """Execute one fold pairing: selection fold -> inference fold."""
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

    print(f"Cross-fit {target}: {n_splits} folds (parallelized)")

    # Fold-level parallelization across independent fold pairs.
    results = Parallel(n_jobs=FOLD_PARALLEL_N_JOBS, backend='threading')(
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
    """Validate survival-related DAG links with a multivariable Cox model.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing survival columns.
    all_links : list[dict] or pandas.DataFrame
        Candidate links extracted from phase 1.
    base_levels_to_adjust : list[int], default=[0, 1]
        Causal levels used as adjustment covariates in the Cox model.

    Returns
    -------
    dict
        Validation status, fitted summaries, and agreement diagnostics.
    """
    if "OS_months" not in df.columns or "OS_event" not in df.columns:
        return {
            'status': 'skipped',
            'reason': "OS_months/OS_event columns are missing",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    if not isinstance(all_links, (list, pd.DataFrame)) or len(all_links) == 0:
        return {
            'status': 'skipped',
            'reason': "No links provided in all_links",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    links_df = pd.DataFrame(all_links) if isinstance(all_links, list) else all_links.copy()
    if links_df.empty:
        return {
            'status': 'skipped',
            'reason': "all_links is empty after conversion",
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
            'reason': "Unable to extract from/to columns from all_links",
            'cox_summary': pd.DataFrame(),
            'cox_significant': pd.DataFrame()
        }

    os_targets = {"OS", "OS_months", "OS_event"}
    dag_os_links = norm_links[norm_links['to'].isin(os_targets)].copy()
    dag_os_predictors = [p for p in dag_os_links['from'].dropna().unique().tolist() if p in df.columns]

    if not dag_os_predictors:
        return {
            'status': 'skipped',
            'reason': "No DAG-derived OS predictor is present in the DataFrame",
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

    print(f"Preparing Cox variables: {len(predictor_set)} candidate variables")
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
            'reason': f"Too few observations after Cox preprocessing (n={len(df_cox)})",
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
            'reason': f"Cox model convergence failed: {e}",
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
    """Run the complete causal pipeline (DML, Cox audit, mediation, export).

    Parameters
    ----------
    fast_mode : bool, default=False
        If ``True``, reduce bootstrap counts for quick debugging runs.

    Returns
    -------
    dict
        Output artifacts including exported links, Cox results, and mediation
        summaries.

    Raises
    ------
    FileNotFoundError
        If the input CSV file is missing.
    """
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print("=" * 80)
    print("PHASE 0 - INITIALIZATION")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    # Runtime parameters: optional fast mode for iteration/debug.
    n_bootstrap_dml = 30 if fast_mode else DEFAULT_N_BOOTSTRAP
    n_bootstrap_mediation = 300 if fast_mode else 1000
    n_splits_dml = 2

    if fast_mode:
        print("FAST mode enabled: reduced bootstraps for quicker runs")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Type coercion for known numeric columns.
    numeric_candidates = sorted(set(numeric_features + ["Age_at_ICI_start", "OS_months", "OS_event"]))
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Minimal cleanup of non-finite/extreme placeholder values.
    df = df.replace([np.inf, -np.inf], np.nan)
    all_links = []

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n" + "=" * 80)
    print("PHASE 1 - STRUCTURE DISCOVERY (DML)")
    print("=" * 80)

    all_targets_to_process = []
    causal_levels_sorted = sorted(CAUSAL_LEVELS.keys())
    target_levels = [lvl for lvl in causal_levels_sorted if lvl > 0]
    for target_level in target_levels:
        targets = [t for t in CAUSAL_LEVELS.get(target_level, []) if t in df.columns and t != "OS_months"]
        predictors = [
            v
            for lvl in causal_levels_sorted
            if lvl < target_level
            for v in CAUSAL_LEVELS.get(lvl, [])
            if v in df.columns
        ]
        predictors = [p for p in dict.fromkeys(predictors)]

        if not targets or not predictors:
            continue

        print(f"Level {target_level}: {len(targets)} target(s), {len(predictors)} candidate predictor(s)")
        for target in targets:
            all_targets_to_process.append((target, predictors, target_level))

    def _process_target_inference(target_info):
        """Run phase-1 inference for one target definition."""
        target, predictors, level = target_info
        try:
            print(f"  -> Processing {target} (level {level})")
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
            print(f"    Inference failed for {target}: {exc}")
            return (target, predictors, None)

    # Parallelization across independent targets.
    print(f"\nParallelizing {len(all_targets_to_process)} DML inferences...")
    inference_results = Parallel(n_jobs=TARGET_PARALLEL_N_JOBS, backend='threading')(
        delayed(_process_target_inference)(target_info)
        for target_info in all_targets_to_process
    )

    for target, predictors, inference_output in inference_results:
        if inference_output is None:
            continue

        agg_df = inference_output.get('aggregated_results', pd.DataFrame())
        if agg_df is None or agg_df.empty:
            continue

        # Approximate per-link stability from fold-level selected features.
        fold_details = inference_output.get('fold_details', [])
        selected_lists = [fd.get('selected_features', []) for fd in fold_details if isinstance(fd, dict)]
        n_fold = max(len(selected_lists), 1)
        stability_map = {}
        for selected in selected_lists:
            for feat in selected:
                stability_map[feat] = stability_map.get(feat, 0) + 1
        stability_map = {k: v / n_fold for k, v in stability_map.items()}

        sig_df = agg_df[agg_df['p_value_fdr_bh'] < 0.05].copy() if 'p_value_fdr_bh' in agg_df.columns else pd.DataFrame()

        print(f"    {target}: {len(sig_df)} significant variables at FDR<0.05")
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
                # Effect size on RR-like scale (from aggregated results)
                'effect_rr': float(row.get('effect_point_mean', np.nan)),
                # Pooled standard error across folds (if available)
                'pooled_se': float(row.get('pooled_se', np.nan)),
                'e_value': float(row.get('e_value_point', np.nan)),
                'e_value_ci': float(row.get('e_value_ci', np.nan)),
                'stability': float(stability_map.get(src_original, np.nan)),
                'method': 'cross_fitted_dml'
            })

    print(f"\nCausal links detected before Cox: {len(all_links)}")

    print("\n" + "=" * 80)
    print("PHASE 2 - TEMPORAL VALIDATION (COX)")
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

        # Update graph: keep survival links only if validated by Cox.
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
        print(f"Cox skipped: {cox_result.get('reason', 'unknown reason')}")

    print(f"Causal links after Cox audit: {len(all_links)}")

    print("\n" + "=" * 80)
    print("PHASE 3 - BOOTSTRAP MEDIATION & E-VALUE RESCUE")
    print("=" * 80)

    mediation_result = []
    cause_col = 'Vaccine100' if 'Vaccine100' in df.columns else None
    outcome_col = 'OS_event' if 'OS_event' in df.columns else None

    # Prioritize clinically plausible mediators present in the data.
    mediator_priority = [
        'Steroid_win_1_month_of_Vaccine',
        'Steroid_win_1_month_of_ICI_start',
        'Concurrent_Chemo',
        'BRAF',
        'ECOG',
        'CNS_disease',
    ]

    if cause_col and outcome_col:
        available_mediators = [m for m in mediator_priority if m in df.columns and m not in {cause_col, outcome_col}]
        # Keep only mediators with non-degenerate variation.
        available_mediators = [m for m in available_mediators if pd.to_numeric(df[m], errors='coerce').dropna().nunique() >= 2]

        if available_mediators:
            mediators_to_run = available_mediators[:3]
            print(f"Mediators tested: {mediators_to_run}")

            for i, mediator_col in enumerate(mediators_to_run):
                med_res = bootstrap_mediation_robust(
                    df=df,
                    cause=cause_col,
                    mediator=mediator_col,
                    outcome=outcome_col,
                    n_boot=n_bootstrap_mediation,
                    random_seed=RANDOM_SEED + i
                )

                if med_res is not None:
                    mediation_result.append(med_res)
                    p_indirect = med_res['indirect_effect']['p_empirical']
                    ci_low = med_res['indirect_effect']['ci_low']
                    ci_high = med_res['indirect_effect']['ci_high']
                    print(
                        f"Mediation {cause_col} -> {mediator_col} -> {outcome_col}: "
                        f"p_indirect={p_indirect:.4f}, IC95%=[{ci_low:.4f}, {ci_high:.4f}]"
                    )

                    # E-value Rescue Logic: add paths missed by DML but robust in mediation
                    RESCUE_THRESHOLD = 1.25
                    existing_links = {(l['from'], l['to']) for l in all_links}

                    # 1. Rescue Path A: cause -> mediator
                    if med_res['path_a']['e_value_ci'] > RESCUE_THRESHOLD or med_res['path_a'].get('is_structural', False):
                        if (cause_col, mediator_col) not in existing_links:
                            is_struct = med_res['path_a'].get('is_structural', False)
                            method_str = 'structural_dependence' if is_struct else 'mediation_rescue_e_value'
                            rescue_msg = f"Path {cause_col} -> {mediator_col} added ({'Structural' if is_struct else f'E-value CI={med_res['path_a']['e_value_ci']:.2f}'})"
                            print(f"      [RESCUE] {rescue_msg}")
                            all_links.append({
                                'from': cause_col,
                                'to': mediator_col,
                                'coef': med_res['path_a']['mean'],
                                'p_value': 0.0 if is_struct else med_res['path_a']['p_empirical'],
                                'e_value': med_res['path_a']['e_value_point'],
                                'e_value_ci': med_res['path_a']['e_value_ci'],
                                'method': method_str
                            })
                            existing_links.add((cause_col, mediator_col))

                    # 2. Rescue Path B: mediator -> outcome
                    if med_res['path_b']['e_value_ci'] > RESCUE_THRESHOLD or med_res['path_b'].get('is_structural', False):
                        if (mediator_col, outcome_col) not in existing_links:
                            is_struct = med_res['path_b'].get('is_structural', False)
                            method_str = 'structural_dependence' if is_struct else 'mediation_rescue_e_value'
                            rescue_msg = f"Path {mediator_col} -> {outcome_col} added ({'Structural' if is_struct else f'E-value CI={med_res['path_b']['e_value_ci']:.2f}'})"
                            print(f"      [RESCUE] {rescue_msg}")
                            all_links.append({
                                'from': mediator_col,
                                'to': outcome_col,
                                'coef': med_res['path_b']['mean'],
                                'p_value': 0.0 if is_struct else med_res['path_b']['p_empirical'],
                                'e_value': med_res['path_b']['e_value_point'],
                                'e_value_ci': med_res['path_b']['e_value_ci'],
                                'method': method_str
                            })
                            existing_links.add((mediator_col, outcome_col))

                else:
                    print(f"Inconclusive mediation for mediator {mediator_col} (no valid bootstrap)")
        else:
            print("Mediation not run: no available mediator with sufficient variance")
    else:
        print("Mediation not run (missing cause/outcome)")

    # Printing causal model (now including rescued links)
    if all_links:
        print("\nE-values du modèle final (incluant repêchages par médiation)")
        
        final_df = pd.DataFrame(all_links)
        # Standardize target names for survival
        final_df['to'] = final_df['to'].replace(['OS', 'OS_months', 'OS_event'], 'Overall_Survival')
        
        cols_to_show = ['from', 'to', 'coef', 'p_value', 'e_value', 'e_value_ci', 'method']
        available_cols = [c for c in cols_to_show if c in final_df.columns]
        display_df = final_df[available_cols].copy()
        
        col_names = {
            'from': 'Source',
            'to': 'Cible',
            'coef': 'Effet',
            'p_value': 'p-value',
            'e_value': 'E-value',
            'e_value_ci': 'E-value_CI',
            'method': 'Méthode'
        }
        display_df = display_df.rename(columns=col_names)
        sort_keys = [k for k in ['Cible', 'p-value'] if k in display_df.columns]
        if sort_keys:
            display_df = display_df.sort_values(sort_keys)
        print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".4f"))

    print("\n" + "=" * 80)
    print("PHASE 4 - EXPORT")
    print("=" * 80)

    export_path = SCRIPT_DIR / "../data/reinforced_causal_dag_structured.csv"
    links_df = pd.DataFrame(all_links)
    links_df.to_csv(export_path, index=False, encoding='utf-8-sig')

    print(f"Export CSV: {export_path}")
    print(f"Exported links: {len(links_df)}")
    print("Causal analysis completed successfully.")

    return {
        'links_df': links_df,
        'cox_result': cox_result,
        'mediation_result': mediation_result,
        'export_path': str(export_path)
    }


if __name__ == "__main__":
    main()
