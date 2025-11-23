# COVID-19 Vaccination & Overall Survival - Analyse Causale

Projet d'analyse causale explorant l'impact de la vaccination COVID-19 sur la survie globale (Overall Survival) chez les patients atteints de cancer traités par immunothérapie.

## Installation et Configuration

### 1. Créer l'environnement virtuel

** PREMIÈRE ÉTAPE OBLIGATOIRE** : Avant toute autre opération, créez l'environnement virtuel avec toutes les dépendances nécessaires :

```powershell
python -m pip install uv
python covid_training/src/MAKE_env.py
```

### 2. Activer l'environnement

Cela dépend de la plateforme sur laquelle vous êtes

## 📊 Exécution du Pipeline d'Analyse

#### Étape 1 : Préparation des données

```powershell
& "covid_training/src/.DAG_env/Scripts/python.exe" "covid_training/src/preparation_covid_data.py"
```

**Ce que fait ce script :**
- Charge les 4 feuilles Excel de données brutes (`1a`, `1b`, `1c`, `1d Raw Data`)
- Identifie les colonnes communes entre les feuilles
- Combine les données en un seul DataFrame
- Applique une stratégie de gestion des valeurs manquantes (par défaut : suppression des lignes incomplètes)
- Exporte le résultat dans `combined_covid_data.csv`

**Paramètres modifiables :**
- `MISSING_DATA_STRATEGY` : `"drop_rows"`, `"drop_cols"`, ou `"keep"`
- `MISSING_THRESHOLD` : seuil pour supprimer les colonnes avec trop de valeurs manquantes

**Sortie :** `ressources/combined_covid_data.csv`

---

#### Étape 2 : Analyse causale et construction du DAG

```powershell
& "covid_training/src/.DAG_env/Scripts/python.exe" "covid_training/src/analyze_covid_data.py"
```

**Ce que fait ce script :**
- Charge les données combinées
- Définit une **structure causale hiérarchique** à 5 niveaux :
  - **Niveau 0** : Variables démographiques (Gender, Ethnicity, Age)
  - **Niveau 1** : Caractéristiques baseline (Stage, ECOG, CNS disease)
  - **Niveau 2** : Traitements (Vaccination, Stéroïdes, Chimiothérapie)
  - **Niveau 3** : Outcomes intermédiaires (PFS - Progression-Free Survival)
  - **Niveau 4** : Outcome final (OS - Overall Survival)
  
- **Modélise** chaque variable en fonction des niveaux précédents :
  - Régression logistique pour les variables binaires
  - Régression linéaire pour les variables continues
  
- **Détecte les effets médiés** : identifie les chemins `X → M → Y` où M médie l'effet de X sur Y

- **Valide avec un modèle de Cox** : analyse de survie finale pour OS

**Sorties :**
- `ressources/causal_dag_structured.csv` : table de tous les liens causaux détectés avec leurs coefficients
- Affichage console : top prédicteurs, médiations détectées, variables significatives du modèle Cox

---

#### Étape 3 : Visualisation du DAG

```powershell
& "covid_training/src/.DAG_env/Scripts/python.exe" "covid_training/src/make_dag.py"
```

**Ce que fait ce script :**
- Charge les liens causaux du fichier CSV
- **Nettoie** le graphe : regroupe les variables one-hot encodées vers leurs versions originales
- **Filtre** : conserve uniquement les chemins menant à OS (Overall Survival)
- **Organise** le layout hiérarchiquement par niveaux causaux
- **Visualise** le DAG   
- **Analyse** les chemins causaux :
  - Chemins de `Vaccine100` → `OS`
  - Variables avec effet direct sur OS
  - Effets médiés

**Sorties :**
- `ressources/causal_dag_simple.png` : visualisation graphique du DAG
- `ressources/causal_dag_stats.csv` : statistiques du graphe (nombre de nœuds, arêtes, densité, etc.)
- `ressources/vaccine_to_os_paths.csv` : tous les chemins causaux de la vaccination vers OS

---

## 📈 Résultats et Interprétation

### Fichiers générés

1. **combined_covid_data.csv** : Données nettoyées et prêtes pour l'analyse
2. **causal_dag_structured.csv** : Relations causales avec coefficients et types d'effets
3. **causal_dag_simple.png** : Graphe visuel du réseau causal
4. **causal_dag_stats.csv** : Métriques du graphe (densité, profondeur, etc.)
5. **vaccine_to_os_paths.csv** : Chemins causaux détaillés vaccination → survie

### Interprétation du DAG

Le DAG permet de :
- **Identifier les confondeurs** : variables qui influencent à la fois le traitement et l'outcome
- **Distinguer effets directs et indirects** : impact direct de la vaccination vs. effets médiés par d'autres variables
- **Détecter les mécanismes causaux** : chemins biologiques/cliniques expliquant l'effet observé
- **Ajuster les analyses** : savoir quelles variables contrôler dans les modèles

## 🔧 Personnalisation

### Modifier les niveaux causaux

Dans `analyze_covid_data.py`, ajustez le dictionnaire `CAUSAL_LEVELS` pour changer la hiérarchie :

```python
CAUSAL_LEVELS = {
    0: ["Gender", "Ethnicity", "Age_at_ICI_start"],
    1: ["Simplified_Stage", "ECOG"],
    # ... etc.
}
```

### Changer les feuilles Excel chargées

Dans `preparation_covid_data.py`, modifiez :

```python
SHEETS_TO_LOAD = [
    "1a Raw Data",
    "1b Raw Data",
    # Ajoutez ou retirez des feuilles
]
```

### Ajuster les seuils de détection

Dans `analyze_covid_data.py` :
- Ligne 144 : `top_features = coefs.head(10)` → modifier le nombre de prédicteurs retenus
- Ligne 248 : `if mediation_ratio < 0.7:` → ajuster le seuil de médiation


## 🐛 Troubleshooting

### Erreur de chemin de fichier
Les scripts utilisent des chemins relatifs basés sur `__file__`. Assurez-vous d'exécuter les scripts avec les chemins complets ou depuis le répertoire racine.

### Environnement virtuel non activé
N'utilisez **pas** `python` directement, mais toujours l'interpréteur de l'environnement virtuel :
```powershell
& "covid_training/src/.DAG_env/Scripts/python.exe" <script>
```

### Données manquantes
Vérifiez que `covid_data.xlsx` est bien présent dans `covid_training/ressources/`

