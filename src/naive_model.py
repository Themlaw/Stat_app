"""
Baseline Causal Structure Learning
----------------------------------
This script implements a naive baseline for causal discovery using a standard
score-based approach (Hill Climbing with Bayesian Information Criterion). 
It serves as a benchmark to demonstrate the limitations (e.g., underfitting, 
sparsity) of unpenalized structural learning on high-dimensional clinical data.
"""

import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import bnlearn as bn

# Suppress non-critical warnings for cleaner execution output
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. GLOBAL CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------
# Ensure strict reproducibility across executions
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../data/combined_covid_data.csv"
OUTPUT_IMG_PATH = SCRIPT_DIR / "../data/dag_naive_baseline.png"

# Clinical features aligned with the downstream Double Machine Learning pipeline
TARGET_FEATURES = [
    "Gender", "Ethnicity", "Age_at_ICI_start",
    "Simplified_Stage", "ECOG", "CNS_disease", 
    "Previous_history_of_malignancy_at_ICI_start",
    "Vaccine100", "Steroid_win_1_month_of_Vaccine", 
    "Concurrent_Chemo", "BRAF", "Immunotherapy_Agent", 
    "Steroid_win_1_month_of_ICI_start",
    "OS_months", "OS_event"
]

# NetworkX Visualization Parameters
NODE_SIZE = 2000
NODE_COLOR = "#B2EBF2"
EDGE_COLOR = "#006064"


def load_and_preprocess_data(filepath: Path, target_cols: list) -> pd.DataFrame:
    """
    Loads the clinical dataset and enforces strict data quality required 
    for score-based Bayesian Network learning.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please run the ETL pipeline first.")

    print("INFO: Loading dataset...")
    df = pd.read_csv(filepath)

    # Filter strictly on predefined structural variables
    available_cols = [col for col in target_cols if col in df.columns]
    print(f"INFO: Retained {len(available_cols)} features for structural learning.")

    # Hill Climbing requires complete cases; enforce listwise deletion
    df_clean = df[available_cols].dropna().copy()
    print(f"INFO: Dataset shape after listwise deletion: {df_clean.shape}")

    # Standardize object types to strings to prevent bnlearn parsing errors
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)

    return df_clean


def visualize_and_save_dag(adj_matrix: pd.DataFrame, output_path: Path) -> None:
    """
    Generates and saves a standardized visual representation of the Directed Acyclic Graph (DAG)
    using NetworkX.
    """
    print("INFO: Generating DAG visualization...")
    graph = nx.DiGraph(adj_matrix)

    # Use spring layout with a fixed seed for reproducible topological rendering
    pos = nx.spring_layout(graph, k=1.5, seed=RANDOM_SEED)

    plt.figure(figsize=(14, 10))

    # Render Nodes
    nx.draw_networkx_nodes(
        graph, pos, 
        node_size=NODE_SIZE, 
        node_color=NODE_COLOR, 
        edgecolors=EDGE_COLOR, 
        linewidths=2
    )

    # Render Labels
    nx.draw_networkx_labels(
        graph, pos, 
        font_size=9, 
        font_weight='bold', 
        font_family='sans-serif'
    )

    # Render Directed Edges
    # min_target_margin ensures arrowheads do not overlap with the node boundaries
    nx.draw_networkx_edges(
        graph, pos, 
        node_size=NODE_SIZE,  
        edge_color=EDGE_COLOR, 
        width=2.0, 
        arrowsize=25, 
        arrowstyle='-|>',        
        connectionstyle="arc3,rad=0.1", 
        min_target_margin=15,    
        alpha=0.8
    )

    n_edges = adj_matrix.sum().sum()
    plt.title(f"Baseline: Naive DAG (Hill Climbing + BIC)\nEdges Detected: {n_edges}", 
              fontsize=16, fontweight='bold')
    plt.axis('off') 
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"INFO: Visualization successfully saved to {output_path}")


def analyze_treatment_effect_path(adj_matrix: pd.DataFrame, treatment: str, outcome: str) -> None:
    """
    Evaluates the presence and directionality of the specific causal edge 
    between the treatment intervention and the clinical outcome.
    """
    print("\n--- Structural Analysis Report ---")
    
    if treatment not in adj_matrix.columns or outcome not in adj_matrix.columns:
        print(f"WARN: Target variables ({treatment}, {outcome}) are isolated or missing from the graph.")
        return

    if adj_matrix.loc[treatment, outcome]:
        print(f"RESULT: Direct causal edge detected: {treatment} -> {outcome}")
    elif adj_matrix.loc[outcome, treatment]:
        print(f"RESULT: STRUCTURAL ERROR - Reverse causality detected: {outcome} -> {treatment}")
    else:
        print(f"RESULT: UNDERFITTING - No direct topological link found between {treatment} and {outcome}.")


def main():
    print("=====================================================")
    print(" Naive Score-Based Causal Discovery Pipeline")
    print("=====================================================\n")

    # 1. Data Preparation
    df_clean = load_and_preprocess_data(DATA_PATH, TARGET_FEATURES)

    # 2. Structural Learning (Hill Climbing + BIC)
    print("\nINFO: Initiating Hill Climbing search with BIC penalization...")
    model = bn.structure_learning.fit(df_clean, methodtype='hc', scoretype='bic')
    
    adj_matrix = model['adjmat']
    total_edges = adj_matrix.sum().sum()
    print(f"INFO: Structural learning complete. Total edges discovered: {total_edges}")

    # 3. Export Visualization
    visualize_and_save_dag(adj_matrix, OUTPUT_IMG_PATH)

    # 4. Clinical Hypothesis Check
    analyze_treatment_effect_path(adj_matrix, treatment='Vaccine100', outcome='OS_event')

    print("\nINFO: Pipeline execution terminated gracefully.")


if __name__ == "__main__":
    main()