"""
Descriptive Statistics and Table 1 Generation
---------------------------------------------
This script generates the baseline characteristics table (Table 1) for the
consolidated clinical dataset. It strictly filters the dataset to include only
the 14 core variables that survived the ETL intersection (common_only=True)
and listwise deletion, ensuring perfect alignment with the downstream DAG
and Cox survival models.

The output is formatted as a LaTeX table, ready to be copy-pasted into the
final research article.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tableone import TableOne
import warnings
import numpy as np

# Suppress minor pandas warnings from tableone for cleaner output
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
# Path to the dataset that has ALREADY been cleaned by preparation_covid_data.py
DATA_PATH = SCRIPT_DIR / "../data/combined_covid_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "../data/figures"

# Ensure output directory exists for saving plots
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The exact 14 variables identified in the naive model and causal levels
CORE_VARIABLES = [
    # Demographics (Level 0)
    "Age_at_ICI_start", 
    "Gender", 
    "Ethnicity",
    
    # Baseline Disease Characteristics (Level 1)
    "ECOG", 
    "CNS_disease", 
    "Previous_history_of_malignancy_at_ICI_start",
    
    # Treatments and Interventions (Level 2)
    "Vaccine100", 
    "Immunotherapy_Agent", 
    "Concurrent_Chemo", 
    "BRAF", 
    "Steroid_win_1_month_of_Vaccine", 
    "Steroid_win_1_month_of_ICI_start",
    
    # Outcomes (Level 3)
    "OS_months", 
    "OS_event"
]

# Identify categorical variables to compute counts and percentages.
# Age and OS_months are considered continuous (mean/SD).
CATEGORICAL_VARS = [
    "Gender", 
    "Ethnicity", 
    "ECOG", 
    "CNS_disease", 
    "Previous_history_of_malignancy_at_ICI_start",
    "Vaccine100", 
    "Immunotherapy_Agent", 
    "Concurrent_Chemo", 
    "BRAF", 
    "Steroid_win_1_month_of_Vaccine", 
    "Steroid_win_1_month_of_ICI_start",
    "OS_event"
]

CONTINUOUS_VARS = ["Age_at_ICI_start", "OS_months"]


def preprocess_continuous_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans continuous variables so they can be processed by pandas math functions.
    Specifically handles strings like '>89' in age columns.
    """
    df_clean = df.copy()
    
    # Fix the '>89' string issue in the age column by replacing it with 90
    if 'Age_at_ICI_start' in df_clean.columns:
        # Replace the string '>89' with '90', then convert the entire column to float
        df_clean['Age_at_ICI_start'] = df_clean['Age_at_ICI_start'].astype(str).str.replace('>89', '90')
        df_clean['Age_at_ICI_start'] = pd.to_numeric(df_clean['Age_at_ICI_start'], errors='coerce')
        
    if 'OS_months' in df_clean.columns:
        df_clean['OS_months'] = pd.to_numeric(df_clean['OS_months'], errors='coerce')
        
    return df_clean


def generate_school_report_stats(df: pd.DataFrame) -> None:
    """
    Computes standard Pandas descriptive statistics for the academic report.
    Includes measures of central tendency, dispersion, and frequencies.
    """
    print("\n" + "="*60)
    print(" DESCRIPTIVE STATISTICS (FOR ACADEMIC REPORT)")
    print("="*60)

    print("\n--- 1. Continuous Variables Summary ---")
    # describe() provides count, mean, std (sqrt of variance), min, 25%, 50% (median), 75%, max
    continuous_summary = df[CONTINUOUS_VARS].describe()
    
    # Adding empirical variance explicitly for statistical completeness
    continuous_summary.loc['variance'] = df[CONTINUOUS_VARS].var()
    print(continuous_summary.round(2))

    print("\n--- 2. Categorical Variables Frequencies ---")
    # Calculate absolute counts and normalized percentages for key features
    for col in ["Vaccine100", "ECOG", "OS_event"]:
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100
        stats_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
        print(f"\nDistribution of {col}:")
        print(stats_df.round(2))


def generate_visualizations(df: pd.DataFrame) -> None:
    """
    Generates standard statistical plots (Boxplots, Bar plots) using 
    Matplotlib and Pandas, and saves them as PNG files.
    """
    print("\n" + "="*60)
    print(" GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Boxplot: Overall Survival distribution by Vaccination Status
    plt.figure(figsize=(8, 6))
    
    # Separate data for Matplotlib
    os_unvax = df[df['Vaccine100'] == 0]['OS_months'].dropna()
    os_vax = df[df['Vaccine100'] == 1]['OS_months'].dropna()
    
    plt.boxplot([os_unvax, os_vax], labels=['Unvaccinated (0)', 'Vaccinated (1)'], patch_artist=True)
    plt.title("Overall Survival (Months) Distribution by Vaccination Status")
    plt.ylabel("Overall Survival (Months)")
    
    out_path_os = OUTPUT_DIR / "boxplot_OS_by_vaccine.png"
    plt.savefig(out_path_os, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path_os}")

    # 2. Boxplot: Age distribution by Vaccination Status
    plt.figure(figsize=(8, 6))
    
    age_unvax = df[df['Vaccine100'] == 0]['Age_at_ICI_start'].dropna()
    age_vax = df[df['Vaccine100'] == 1]['Age_at_ICI_start'].dropna()
    
    plt.boxplot([age_unvax, age_vax], labels=['Unvaccinated (0)', 'Vaccinated (1)'], patch_artist=True)
    plt.title("Age Distribution by Vaccination Status")
    plt.ylabel("Age at ICI Start")
    
    out_path_age = OUTPUT_DIR / "boxplot_Age_by_vaccine.png"
    plt.savefig(out_path_age, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path_age}")

    # 3. Barplot: Performance Status (ECOG)
    plt.figure(figsize=(8, 6))
    
    # Use Pandas to prepare the crosstab, then plot via Matplotlib
    ecog_crosstab = pd.crosstab(df['ECOG'], df['Vaccine100'])
    ecog_crosstab.plot(kind='bar', figsize=(8, 6), width=0.8)
    
    plt.title("ECOG Performance Status by Vaccination Group")
    plt.xlabel("ECOG Score (Lower is better)")
    plt.ylabel("Patient Count")
    plt.legend(title="Vaccine100", labels=['Unvaccinated (0)', 'Vaccinated (1)'])
    plt.xticks(rotation=0)
    
    out_path_ecog = OUTPUT_DIR / "barplot_ECOG_by_vaccine.png"
    plt.savefig(out_path_ecog, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path_ecog}")


def generate_latex_table_one(df: pd.DataFrame) -> None:
    """
    Generates the scientific 'Table 1' using the tableone library.
    This outputs ready-to-paste LaTeX code for the research article.
    """
    print("\n" + "="*60)
    print(" LATEX TABLE 1 (FOR SCIENTIFIC ARTICLE)")
    print("="*60)

    try:
        table1 = TableOne(
            df, 
            columns=CORE_VARIABLES, 
            categorical=CATEGORICAL_VARS, 
            groupby='Vaccine100', 
            pval=True,
            missing=False # Listwise deletion handled prior
        )
        print(table1.tabulate(tablefmt="latex"))
        print("\n------------------------------------------------")
    except Exception as e:
        print(f"Failed to generate Table 1: {e}")


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Please run the ETL script first.")
        return

    print("INFO: Loading cleaned dataset...")
    # Load data and filter to core variables
    df = pd.read_csv(DATA_PATH)
    
    missing_cols = [col for col in CORE_VARIABLES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CRITICAL ERROR: Missing core variables in the dataset: {missing_cols}")
        
    df_filtered = df[CORE_VARIABLES].copy()
    
    # Preprocess strings in continuous columns before dropping NAs
    df_filtered = preprocess_continuous_vars(df_filtered)
    
    # Enforce listwise deletion (Complete Case Analysis)
    initial_shape = df_filtered.shape
    df_clean = df_filtered.dropna()
    final_shape = df_clean.shape
    print(f"INFO: Applied listwise deletion. Rows retained: {final_shape[0]} (out of {initial_shape[0]}).")

    # Execute the three reporting phases
    generate_school_report_stats(df_clean)
    generate_visualizations(df_clean)
    generate_latex_table_one(df_clean)


if __name__ == "__main__":
    main()