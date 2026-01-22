import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
LINKS_PATH = SCRIPT_DIR / "../data/causal_dag_structured.csv"

# Chemins de sortie
OUTPUT_FULL = SCRIPT_DIR / "../data/causal_dag_COMPLETE_flat.png"       # Non filtré
OUTPUT_SMART = SCRIPT_DIR / "../data/causal_dag_SMART_flat.png"         # Filtré (Smart)

# Paramètres du filtre "Smart"
BASE_THRESHOLD = 0.10      
JUMP_PENALTY   = 0.15      

# Paramètres visuels
HORIZONTAL_SPACING = 6.0
VERTICAL_SPACING = 3.0
LABEL_FONT_SIZE = 9

# Épaisseur des arêtes (sera pondérée par les coefficients)
EDGE_WIDTH_MIN = 1.5
EDGE_WIDTH_MAX = 6.0
EDGE_ALPHA = 0.8

# Styles
EDGE_COLOR = "#2E86AB"
NODE_FACE_COLORS = ['#f4a261','#2a9d8f','#264653','#e9c46a','#e76f51'] 
RED_HIGHLIGHT = "#D62828"

# -------------------------------------------------------------------
# 1. Chargement & Nettoyage
# -------------------------------------------------------------------
if not LINKS_PATH.exists(): raise FileNotFoundError(f"Missing: {LINKS_PATH}")
links_df = pd.read_csv(LINKS_PATH, encoding='utf-8-sig')

# Mapping noms
all_names = set(links_df['from'].astype(str)).union(set(links_df['to'].astype(str)))
candidate_bases = {n for n in all_names if '_' not in n}

def map_to_orig(name, candidates):
    if pd.isna(name): return name
    if name in candidates: return name
    if "_" in name and name.split("_")[0] in candidates: return name.split("_")[0]
    return name

links_df['from'] = links_df['from'].apply(lambda x: map_to_orig(x, candidate_bases))
links_df['to']   = links_df['to'].apply(lambda x: map_to_orig(x, candidate_bases))

# Agrégation (Max absolu)
links_clean = links_df.groupby(['from', 'to'], as_index=False).agg({
    'coef': lambda x: x.iloc[np.argmax(np.abs(x))],
    'level_from': 'first', 'level_to': 'first', 'abs_coef': 'max'
})

# -------------------------------------------------------------------
# 2. NORMALISATION ET PRÉPARATION
# -------------------------------------------------------------------
# Normalisation relative par cible
links_clean['relative_importance'] = links_clean.groupby('to')['abs_coef'].transform(lambda x: x / x.max())

# Calcul des sauts et seuils
links_clean['level_from'] = links_clean['level_from'].fillna(0).astype(int)
links_clean['level_to']   = links_clean['level_to'].fillna(4).astype(int)
links_clean['jump_size']  = (links_clean['level_to'] - links_clean['level_from'])
links_clean['required_score'] = BASE_THRESHOLD + (JUMP_PENALTY * (links_clean['jump_size'] - 1).clip(lower=0))

# -------------------------------------------------------------------
# 3. CRÉATION DES DEUX GRAPHES (Full et Filtered)
# -------------------------------------------------------------------

# Graphe COMPLET (Sans filtre)
G_full = nx.DiGraph()
for _, r in links_clean.iterrows():
    G_full.add_edge(
        r['from'],
        r['to'],
        weight=r['relative_importance'],
        coef=r['coef'],
        abs_coef=abs(r['coef'])
    )

# Graphe FILTRÉ (Smart Filter)
links_filtered = links_clean[links_clean['relative_importance'] >= links_clean['required_score']].copy()
G_smart = nx.DiGraph()
for _, r in links_filtered.iterrows():
    G_smart.add_edge(
        r['from'],
        r['to'],
        weight=r['relative_importance'],
        coef=r['coef'],
        abs_coef=abs(r['coef'])
    )

print(f"📊 Liens totaux (G_full) : {G_full.number_of_edges()}")
print(f"📊 Liens filtrés (G_smart): {G_smart.number_of_edges()}")

# -------------------------------------------------------------------
# 4. CALCUL DU LAYOUT (Global pour cohérence)
# -------------------------------------------------------------------
# On calcule les niveaux et positions sur les données COMPLÈTES pour que 
# les nœuds soient au même endroit sur les deux images.
node_levels = {}
for _, row in links_clean.iterrows():
    node_levels[row['from']] = int(row['level_from'])
    node_levels[row['to']]   = int(row['level_to'])

# Sécurité
all_nodes_set = set(G_full.nodes()) | set(G_smart.nodes())
for n in all_nodes_set:
    if n not in node_levels: node_levels[n] = 0

unique_levels = sorted(set(node_levels.values()))
level_map = {lv: i for i, lv in enumerate(unique_levels)}
for n in node_levels: node_levels[n] = level_map[node_levels[n]]
penultimate_level = max(node_levels.values()) - 1 if node_levels else -1

# Positions basées sur G_full pour avoir tous les noeuds
level_to_nodes = {}
for n in G_full.nodes():
    level_to_nodes.setdefault(node_levels.get(n, 0), []).append(n)

pos = {}
for lv, nodes in level_to_nodes.items():
    # Tri stable
    nodes_sorted = sorted(nodes, key=lambda x: (-G_full.degree(x) if x in G_full else 0, x))
    count = len(nodes_sorted)
    for i, node in enumerate(nodes_sorted):
        pos[node] = (lv * HORIZONTAL_SPACING, (i - (count-1)/2) * VERTICAL_SPACING)

# -------------------------------------------------------------------
# 5. FONCTION DE DESSIN GÉNÉRIQUE
# -------------------------------------------------------------------
def draw_dag(graph, title, output_path, show=False):
    plt.figure(figsize=(20, 12))
    ax = plt.gca()
    plt.title(title, fontsize=18, weight='bold', pad=20)

    node_artists = {}
    
    # A. DESSINER LES NOEUDS
    for node in graph.nodes():
        if node not in pos: continue
        lvl = node_levels.get(node, 0)
        
        # Logique Couleur
        if lvl == penultimate_level:
            color = RED_HIGHLIGHT
            txt_col = 'white'
        else:
            color = NODE_FACE_COLORS[lvl % len(NODE_FACE_COLORS)]
            txt_col = 'black'
        
        t = ax.text(pos[node][0], pos[node][1], str(node),
                    fontsize=LABEL_FONT_SIZE + 3, fontweight='bold', color=txt_col,
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.65", facecolor=color, edgecolor='black', linewidth=1.2),
                    zorder=10)
        node_artists[node] = t

    # B. DESSINER LES ARÊTES (ÉPAISSEUR PONDÉRÉE)
    edge_strengths = [abs(data.get('coef', data.get('weight', 0))) for _, _, data in graph.edges(data=True)]
    max_strength = max(edge_strengths) if edge_strengths else 0

    for u, v in graph.edges():
        if u not in pos or v not in pos: continue

        data = graph.get_edge_data(u, v, default={})
        strength = abs(data.get('coef', data.get('weight', 0)))
        normalized = (strength / max_strength) if max_strength else 0
        linewidth = EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * normalized
        
        ax.annotate("", 
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=0.8,head_width=0.5",
                        color=EDGE_COLOR, 
                        alpha=EDGE_ALPHA,
                        linewidth=linewidth,
                        connectionstyle="arc3,rad=0.1",
                        patchB=node_artists[v], 
                        patchA=node_artists[u]
                    ),
                    zorder=5)

    # Cadrage
    visible_nodes = [n for n in graph.nodes() if n in pos]
    if visible_nodes:
        xs = [pos[n][0] for n in visible_nodes]
        ys = [pos[n][1] for n in visible_nodes]
        ax.set_xlim(min(xs)-2, max(xs)+2)
        ax.set_ylim(min(ys)-2, max(ys)+2)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Image sauvegardée : {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

# -------------------------------------------------------------------
# 6. EXÉCUTION
# -------------------------------------------------------------------

# 1. Sauvegarder le graphe COMPLET (sans afficher)
draw_dag(G_full, 
         "DAG Causal (Complet - Non Filtré)", 
         OUTPUT_FULL, 
         show=False)

# 2. Sauvegarder et Afficher le graphe FILTRÉ
draw_dag(G_smart, 
         f"DAG Causal (Smart Filter - Importance Rel > {BASE_THRESHOLD})", 
         OUTPUT_SMART, 
         show=True)