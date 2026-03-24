import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Paramètres
# -------------------------------------------------------------------
DATA_PATH = Path("covid_data.xlsx")

SHEETS_TO_LOAD = [
    "1a Raw Data",
    "1b Raw Data",
    "1c Raw Data",
    "1d Raw Data",
    "1e Raw Data",
]

# -------------------------------------------------------------------
# Chargement et analyse des colonnes communes
# -------------------------------------------------------------------
xls = pd.ExcelFile(DATA_PATH)

# Dictionnaire pour stocker les colonnes de chaque feuille
sheets_columns = {}

for sheet in SHEETS_TO_LOAD:
    if sheet not in xls.sheet_names:
        print(f"⚠️ Feuille {sheet} absente du fichier")
        continue
    
    df = pd.read_excel(DATA_PATH, sheet_name=sheet)
    sheets_columns[sheet] = set(df.columns)
    print(f"\n{sheet}: {len(df.columns)} colonnes")

# Colonnes communes à TOUTES les feuilles
common_columns = set.intersection(*sheets_columns.values())

print(f"\n{'='*60}")
print(f"Colonnes communes aux {len(SHEETS_TO_LOAD)} feuilles ({len(common_columns)} colonnes):")
print(f"{'='*60}")
for col in sorted(common_columns):
    print(f"  - {col}")

# Colonnes présentes dans au moins une feuille
all_columns = set.union(*sheets_columns.values())

print(f"\n{'='*60}")
print(f"Total de colonnes uniques: {len(all_columns)}")
print(f"{'='*60}")

# Détail par feuille (colonnes manquantes)
print(f"\n{'='*60}")
print("Colonnes manquantes par feuille:")
print(f"{'='*60}")
for sheet, cols in sheets_columns.items():
    missing = all_columns - cols
    if missing:
        print(f"\n{sheet} - manque {len(missing)} colonnes:")
        for col in sorted(missing)[:10]:  # Limité aux 10 premières
            print(f"  - {col}")
        if len(missing) > 10:
            print(f"  ... et {len(missing) - 10} autres")