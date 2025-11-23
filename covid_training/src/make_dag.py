import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# -------------------------------------------------------------------
# 1. Chargement des liens causaux structurés
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
LINKS_PATH = SCRIPT_DIR / "../ressources/causal_dag_structured.csv"

if not LINKS_PATH.exists():
    raise FileNotFoundError(f"Le fichier {LINKS_PATH} n'existe pas. Exécutez d'abord analyze_covid_data_advanced.py")

links_df = pd.read_csv(LINKS_PATH, encoding='utf-8-sig')
print(f"✅ {len(links_df)} liens chargés depuis {LINKS_PATH}\n")

# -------------------------------------------------------------------
# 2. Nettoyage : supprimer les variables one-hot encodées
# -------------------------------------------------------------------

def map_to_original_variable(var_name):
    """Mappe une variable encodée vers sa variable originale"""
    base_vars = ["Gender", "Ethnicity", "Simplified_Stage", "ECOG", "BRAF", "CNS_disease"]
    for base in base_vars:
        if var_name.startswith(f"{base}_"):
            return base
    return var_name

# Mapper toutes les variables encodées vers leurs versions originales
links_df["from_original"] = links_df["from"].apply(map_to_original_variable)
links_df["to_original"] = links_df["to"].apply(map_to_original_variable)

# Grouper par (from_original, to_original) et garder le lien le plus fort
links_clean = links_df.groupby(["from_original", "to_original"]).agg({
    "coef": lambda x: x.iloc[np.argmax(np.abs(x))],  # Coef le plus fort
    "level_from": "first",
    "level_to": "first",
    "type": "first",
    "abs_coef": "max"
}).reset_index()

links_clean = links_clean.rename(columns={"from_original": "from", "to_original": "to"})

print(f"📊 Après nettoyage : {len(links_clean)} liens (avant : {len(links_df)})\n")

# -------------------------------------------------------------------
# 3. Construction du graphe orienté hiérarchique
# -------------------------------------------------------------------
G = nx.DiGraph()

# Ajouter les arêtes avec attributs
for _, row in links_clean.iterrows():
    source = row['from']
    target = row['to']
    
    G.add_edge(source, target, 
               type=row['type'],
               coef=row['coef'],
               level_from=row['level_from'],
               level_to=row['level_to'])

print(f"📊 Graphe construit:")
print(f"   - {G.number_of_nodes()} nœuds")
print(f"   - {G.number_of_edges()} arêtes\n")

# -------------------------------------------------------------------
# 4. Filtrage : garder uniquement les chemins vers OS
# -------------------------------------------------------------------

# Identifier tous les nœuds qui mènent à OS (ancêtres)
os_nodes = ["OS", "OS_months", "OS_event"]
ancestors = set()

for os_node in os_nodes:
    if os_node in G.nodes():
        ancestors.update(nx.ancestors(G, os_node))
        ancestors.add(os_node)

# Garder aussi les nœuds très connectés (hubs)
degree_dict = dict(G.degree())
hubs = [node for node, deg in degree_dict.items() if deg >= 3]

nodes_to_keep = list(ancestors.union(set(hubs)))

G_filtered = G.subgraph(nodes_to_keep).copy()

print(f"📊 Graphe filtré (chemins vers OS):")
print(f"   - {G_filtered.number_of_nodes()} nœuds")
print(f"   - {G_filtered.number_of_edges()} arêtes\n")

# -------------------------------------------------------------------
# 5. Layout hiérarchique par niveaux causaux
# -------------------------------------------------------------------

# Grouper les nœuds par niveau
level_positions = {}
for node in G_filtered.nodes():
    # Récupérer le niveau du nœud (niveau_to des arêtes entrantes)
    incoming_edges = list(G_filtered.in_edges(node, data=True))
    if incoming_edges:
        level = incoming_edges[0][2].get('level_to', 0)
    else:
        # Nœuds sans prédécesseurs = niveau 0
        level = 0
    
    if level not in level_positions:
        level_positions[level] = []
    level_positions[level].append(node)

# Créer des positions hiérarchiques
pos = {}
for level, nodes in sorted(level_positions.items()):
    n_nodes = len(nodes)
    for i, node in enumerate(nodes):
        x = level * 4  # Espacement horizontal par niveau
        y = (i - n_nodes/2) * 2.5  # Espacement vertical
        pos[node] = (x, y)

# Si certains nœuds n'ont pas de position, utiliser spring_layout
missing_nodes = set(G_filtered.nodes()) - set(pos.keys())
if missing_nodes:
    pos_spring = nx.spring_layout(G_filtered.subgraph(missing_nodes), k=2, seed=42)
    pos.update(pos_spring)

# -------------------------------------------------------------------
# 6. Visualisation du DAG hiérarchique SIMPLIFIÉ
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(28, 18))

# Couleur unique pour tous les nœuds (gris clair)
node_color = '#D3D3D3'

# Taille des nœuds proportionnelle au degré
degrees = dict(G_filtered.degree())
node_sizes = [500 + degrees[node] * 150 for node in G_filtered.nodes()]

# Dessiner les nœuds
nx.draw_networkx_nodes(G_filtered, pos,
                       node_color=node_color,
                       node_size=node_sizes,
                       alpha=0.9,
                       edgecolors='black',
                       linewidths=2,
                       ax=ax)

# Dessiner les labels avec meilleure visibilité
nx.draw_networkx_labels(G_filtered, pos,
                        font_size=11,
                        font_weight='bold',
                        font_family='sans-serif',
                        ax=ax)

# Dessiner toutes les arêtes de manière uniforme et visible
edge_color = '#2E86AB'  # Bleu foncé
edge_width = 2.5

# Dessiner les lignes AVEC flèches aux extrémités
nx.draw_networkx_edges(G_filtered, pos,
                       edge_color=edge_color,
                       width=edge_width,
                       arrows=True,
                       arrowstyle='->',
                       arrowsize=20,
                       alpha=0.7,
                       connectionstyle='arc3,rad=0.1',
                       node_size=node_sizes,
                       min_source_margin=15,
                       min_target_margin=15,
                       ax=ax)

# Ajouter des flèches bien visibles au milieu des arêtes
for u, v in G_filtered.edges():
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    
    # Point au milieu de l'arête
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # Direction de la flèche
    dx = x2 - x1
    dy = y2 - y1
    
    # Normaliser la direction
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        dx /= length
        dy /= length
    
    # Taille de la flèche
    arrow_length = 0.2
    
    # Dessiner la flèche au milieu
    ax.annotate('',
                xy=(mid_x + dx * arrow_length, mid_y + dy * arrow_length),
                xytext=(mid_x - dx * arrow_length, mid_y - dy * arrow_length),
                arrowprops=dict(arrowstyle='->',
                               color=edge_color,
                               lw=edge_width,
                               alpha=0.8,
                               shrinkA=0,
                               shrinkB=0))

ax.set_title("DAG Causal - COVID-19 Vaccination → Overall Survival",
             fontsize=20, fontweight='bold', pad=20)

ax.axis('off')
plt.tight_layout()

# Sauvegarde
output_path = SCRIPT_DIR / "../ressources/causal_dag_simple.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ DAG sauvegardé dans {output_path}")

plt.show()

# -------------------------------------------------------------------
# 7. Analyse des chemins causaux vers OS
# -------------------------------------------------------------------
print("\n" + "="*60)
print("ANALYSE DES CHEMINS CAUSAUX VERS OS")
print("="*60)

# Trouver tous les chemins vers OS
os_final = None
for node in ["OS", "OS_months", "OS_event"]:
    if node in G_filtered.nodes():
        os_final = node
        break

if os_final:
    # Chemins depuis Vaccine100
    if "Vaccine100" in G_filtered.nodes():
        try:
            paths = list(nx.all_simple_paths(G_filtered, "Vaccine100", os_final, cutoff=5))
            print(f"\n🎯 Chemins causaux de Vaccine100 → {os_final} ({len(paths)} chemins):\n")
            
            for i, path in enumerate(paths[:10], 1):
                print(f"  {i}. {' → '.join(path)}")
            
            if len(paths) > 10:
                print(f"  ... et {len(paths)-10} autres chemins")
        except nx.NetworkXNoPath:
            print(f"\n⚠️ Aucun chemin trouvé entre Vaccine100 et {os_final}")
    
    # Variables avec effet direct sur OS
    direct_causes = list(G_filtered.predecessors(os_final))
    print(f"\n📊 Variables avec effet DIRECT sur {os_final} ({len(direct_causes)}):")
    
    direct_effects = []
    for var in direct_causes:
        edge_data = G_filtered[var][os_final]
        direct_effects.append({
            "variable": var,
            "coef": edge_data.get('coef', 0),
            "type": edge_data.get('type', 'unknown')
        })
    
    direct_df = pd.DataFrame(direct_effects).sort_values('coef', key=abs, ascending=False)
    print(direct_df.to_string(index=False))
    
    # Variables avec effet médié
    print(f"\n🔗 Effets médiés détectés:")
    mediated_links = links_clean[links_clean['type'].str.contains('mediated_by', na=False)]
    if len(mediated_links) > 0:
        print(mediated_links[['from', 'to', 'type', 'coef']].to_string(index=False))
    else:
        print("  Aucune médiation détectée")

# -------------------------------------------------------------------
# 8. Export des statistiques
# -------------------------------------------------------------------

stats_df = pd.DataFrame([
    {"metric": "Nombre de nœuds", "value": G_filtered.number_of_nodes()},
    {"metric": "Nombre d'arêtes", "value": G_filtered.number_of_edges()},
    {"metric": "Densité", "value": f"{nx.density(G_filtered):.3f}"},
    {"metric": "Profondeur max (niveaux)", "value": max(level_positions.keys()) if level_positions else 0},
    {"metric": "Variables influençant OS", "value": len(direct_causes) if os_final else 0},
])

stats_df.to_csv(SCRIPT_DIR / "../ressources/causal_dag_stats.csv", index=False, encoding='utf-8-sig')
print(f"\n✅ Statistiques exportées dans ../ressources/causal_dag_stats.csv")

# Export de la table des chemins
if os_final and "Vaccine100" in G_filtered.nodes():
    try:
        paths = list(nx.all_simple_paths(G_filtered, "Vaccine100", os_final, cutoff=5))
        paths_df = pd.DataFrame([{"path": " → ".join(p), "length": len(p)-1} for p in paths])
        paths_df.to_csv(SCRIPT_DIR / "../ressources/vaccine_to_os_paths.csv", index=False, encoding='utf-8-sig')
        print(f"✅ Chemins Vaccine100→OS exportés dans ../ressources/vaccine_to_os_paths.csv")
    except:
        pass