import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# 1. Base parameters
# -------------------------------------------------------------------
# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../ressources/covid_data.xlsx"  # path provided by the local environment

SHEETS_TO_LOAD = [
    "1a Raw Data",
    "1b Raw Data",
    "1c Raw Data",
    "1d Raw Data",
]

OUTPUT_PATH = SCRIPT_DIR / "../data/combined_covid_data.csv"

# Missing-data handling options:
# "keep": keep all rows and columns (default)
# "drop_rows": drop rows with at least one missing value
# "drop_cols": drop columns with at least one missing value
MISSING_DATA_STRATEGY = "drop_rows"  # Options: "keep", "drop_rows", "drop_cols"

# Threshold for drop_cols: remove columns with more than X% missing values
MISSING_THRESHOLD = 0  # 0.5 = 50% missing values


# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------

def get_common_columns(excel_path: Path, sheets: list[str]) -> list[str]:
    """Identify columns shared by all selected sheets.

    Parameters
    ----------
    excel_path : pathlib.Path
        Path to the Excel workbook.
    sheets : list[str]
        Sheet names to inspect.

    Returns
    -------
    list[str]
        Sorted list of columns present in every available sheet.
    """
    xls = pd.ExcelFile(excel_path)
    sheets_columns = []
    
    for sheet in sheets:
        if sheet not in xls.sheet_names:
            print(f"Warning: sheet '{sheet}' is missing and will be skipped.")
            continue
        
        df = pd.read_excel(excel_path, sheet_name=sheet, nrows=0)  # Load only headers.
        sheets_columns.append(set(df.columns))
    
    if not sheets_columns:
        raise ValueError("No valid sheet was found.")

    # Intersect all column sets.
    common_columns = set.intersection(*sheets_columns)
    
    print(f"Found {len(common_columns)} shared columns across {len(sheets)} configured sheets")

    return sorted(common_columns)


def standardize_columns(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Standardize key column names across source sheets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe from one sheet.
    sheet_name : str
        Source sheet name.

    Returns
    -------
    pandas.DataFrame
        Standardized dataframe with a ``source_sheet`` provenance column.
    """
    df = df.copy()
    df["source_sheet"] = sheet_name

    # Map variant source column names to a standard schema.
    rename_map = {
        # OS (overall survival)
        "OS_from_relevant_ICI_start,_months": "OS_months",
        "OS_from_relevant_ICI_start_months": "OS_months",
        "OS_Months": "OS_months",

        # OS event code
        "OS_Code_(1=death, 0=censored)": "OS_event",
        "OS_Code_1=death": "OS_event",
        "OS_Code": "OS_event",
        "OS_Code_": "OS_event",

        # vaccine group
        "Group_(1=vaccine)": "Vaccine100",
        "Group_1=vaccine": "Vaccine100",
    }

    df = df.rename(columns=rename_map)

    # Ensure mandatory columns exist even if missing in a sheet.
    if "OS_months" not in df.columns:
        df["OS_months"] = np.nan
    if "OS_event" not in df.columns:
        df["OS_event"] = np.nan
    if "Vaccine100" not in df.columns:
        df["Vaccine100"] = np.nan

    # Cast Vaccine100 to binary integer when values are available.
    if df["Vaccine100"].notna().any():
        df["Vaccine100"] = df["Vaccine100"].astype("float").round().astype("Int64")

    return df


def handle_missing_data(df: pd.DataFrame, strategy: str = "keep", threshold: float = 0.5) -> pd.DataFrame:
    """Handle missing values using the configured strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to process.
    strategy : str, default="keep"
        One of ``keep``, ``drop_rows``, or ``drop_cols``.
    threshold : float, default=0.5
        Missing-value ratio threshold used with ``drop_cols``.

    Returns
    -------
    pandas.DataFrame
        Processed dataframe.
    """
    initial_shape = df.shape
    
    if strategy == "keep":
        print(f"Strategy 'keep': retaining all data ({initial_shape[0]} rows, {initial_shape[1]} columns)")
        return df
    
    elif strategy == "drop_rows":
        df_clean = df.dropna()
        print(f"Strategy 'drop_rows': {initial_shape[0]} -> {df_clean.shape[0]} rows ({initial_shape[0] - df_clean.shape[0]} removed)")
        return df_clean
    
    elif strategy == "drop_cols":
        # Compute the missing-value ratio per column.
        missing_pct = df.isnull().sum() / len(df)
        cols_to_keep = missing_pct[missing_pct <= threshold].index.tolist()
        cols_dropped = [col for col in df.columns if col not in cols_to_keep]
        
        df_clean = df[cols_to_keep]
        print(f"Strategy 'drop_cols' (threshold={threshold*100}%): {initial_shape[1]} -> {df_clean.shape[1]} columns ({len(cols_dropped)} removed)")
        if cols_dropped:
            print(f"   Removed columns: {', '.join(cols_dropped[:10])}" + ("..." if len(cols_dropped) > 10 else ""))
        return df_clean
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Options: 'keep', 'drop_rows', 'drop_cols'")


def load_and_combine_sheets(excel_path: Path, sheets: list[str], common_only: bool = True) -> pd.DataFrame:
    """Load, standardize, and combine selected sheets.

    Parameters
    ----------
    excel_path : pathlib.Path
        Path to the Excel workbook.
    sheets : list[str]
        Sheet names to load.
    common_only : bool, default=True
        If True, keep only columns shared by all loaded sheets.

    Returns
    -------
    pandas.DataFrame
        Combined dataframe with harmonized columns.
    """
    xls = pd.ExcelFile(excel_path)
    
    all_dfs = []

    for sheet in sheets:
        if sheet not in xls.sheet_names:
            print(f"Warning: sheet '{sheet}' is missing and will be skipped.")
            continue

        df = pd.read_excel(excel_path, sheet_name=sheet)
        
        # First, standardize key column names.
        df = standardize_columns(df, sheet)
        
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No sheet was loaded; check sheet names.")

    # Then compute shared columns (after standardization).
    if common_only:
        sheets_columns = [set(df.columns) for df in all_dfs]
        common_columns = set.intersection(*sheets_columns)
        print(f"Found {len(common_columns)} shared columns after standardization")

        # Filter each dataframe to shared columns.
        all_dfs = [df[sorted(common_columns)] for df in all_dfs]

    # Concatenate dataframes (column union handled by pandas).
    combined = pd.concat(all_dfs, axis=0, ignore_index=True)

    # Optionally remove exact duplicate rows.
    combined = combined.drop_duplicates()

    return combined


# -------------------------------------------------------------------
# 3. Usage
# -------------------------------------------------------------------
combined_df = load_and_combine_sheets(DATA_PATH, SHEETS_TO_LOAD, common_only=True)

print(f"\n{'='*60}")
print(f"Before missing-data handling: {combined_df.shape}")
print(f"{'='*60}")

# Apply missing-data strategy.
combined_df = handle_missing_data(combined_df, strategy=MISSING_DATA_STRATEGY, threshold=MISSING_THRESHOLD)

print(f"\n{'='*60}")
print(f"Final shape: {combined_df.shape}")
print(f"{'='*60}")

# Display missing-value statistics.
missing_stats = combined_df.isnull().sum()
if missing_stats.sum() > 0:
    print(f"\nMissing values by column (top 10):")
    print(missing_stats[missing_stats > 0].sort_values(ascending=False).head(10))
else:
    print(f"\nNo missing values in the final dataset")

print(f"\nData preview:")
print(combined_df[["source_sheet", "OS_months", "OS_event", "Vaccine100"]].head())

combined_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"\nData exported to {OUTPUT_PATH}")
