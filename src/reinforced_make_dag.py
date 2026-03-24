import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).parent
LINKS_PATH = SCRIPT_DIR / "../data/reinforced_causal_dag_structured.csv"
OUTPUT_DAG_PNG = SCRIPT_DIR / "../data/reinforced_causal_dag_final.png"
OUTPUT_DAG_SVG = SCRIPT_DIR / "../data/reinforced_causal_dag_final.svg"
OUTPUT_PRUNED_CSV = SCRIPT_DIR / "../data/reinforced_causal_dag_pruned.csv"

# Visual tuning
HORIZONTAL_SPACING = 5.8
VERTICAL_SPACING = 2.25
LABEL_WRAP_WIDTH = 22
BASE_FONT_SIZE = 10
BASE_BOX_PADDING = 0.75

EDGE_WIDTH_MIN = 0.9
EDGE_WIDTH_MAX = 3.6
EDGE_ALPHA = 0.9

NODE_EDGE_COLOR = "#2b2b2b"
NODE_TEXT_COLOR = "#111111"

LEVEL_COLORS = {
    0: "#cfe8ff",  # demographie
    1: "#d8f3dc",  # baseline maladie
    2: "#fff3bf",  # traitements
    3: "#f8d7da",  # outcomes intermediaires
    4: "#f1f3f5",  # outcomes finaux
    5: "#e5dbff",  # survival auditee
}

CAUSAL_LEVELS = {
    0: ["Gender", "Ethnicity", "Age_at_ICI_start", "English_as_primary_language"],
    1: ["Simplified_Stage", "ECOG", "CNS_disease", "Previous_history_of_malignancy_at_ICI_start"],
    2: ["Vaccine100", "Steroid_win_1_month_of_Vaccine", "Concurrent_Chemo", "BRAF", "Immunotherapy_Agent", "Steroid_win_1_month_of_ICI_start"],
    3: ["PFS_", "PFS_Code"],
    4: ["OS_months", "OS_event"],
}


def build_level_map():
    level_map = {}
    for level, names in CAUSAL_LEVELS.items():
        for name in names:
            level_map[name] = level
    level_map["Overall_Survival"] = 5
    return level_map


def wrap_node_label(name):
    return textwrap.fill(str(name).replace("_", " "), width=LABEL_WRAP_WIDTH)


def _resolve_column(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Colonnes candidates introuvables: {candidates}")
    return None


def load_links(path):
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"from", "to", "coef"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {sorted(missing)}")

    df = df.copy()
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")

    p_col = _resolve_column(df, ["p_value_fdr_bh", "p_value"], required=True)
    s_col = _resolve_column(df, ["stability_score", "stability"], required=False)
    e_col = _resolve_column(df, ["e_value", "e_value_point", "e_value_point_mean"], required=False)

    df["p_stat"] = pd.to_numeric(df[p_col], errors="coerce")
    df["stability_stat"] = pd.to_numeric(df[s_col], errors="coerce") if s_col else np.nan
    df["e_value_stat"] = pd.to_numeric(df[e_col], errors="coerce") if e_col else np.nan

    df = df.dropna(subset=["from", "to", "coef", "p_stat"])
    print(f"Liens bruts chargees: {len(df)}")
    return df


def apply_causal_pruning(df, p_threshold=0.05, stability_threshold=0.65, use_evalue=True, evalue_threshold=1.2):
    out = df.copy()

    # 1) Significativité stricte
    out = out[out["p_stat"] <= p_threshold].copy()
    print(f"Apres filtre p<= {p_threshold:.3f}: {len(out)}")

    # 2) Stabilité minimale
    out = out[out["stability_stat"].fillna(0.0) >= stability_threshold].copy()
    print(f"Apres filtre stabilite>= {stability_threshold:.2f}: {len(out)}")

    # 3) Robustesse E-value
    if use_evalue:
        ok_mask = out["e_value_stat"].isna() | (out["e_value_stat"] >= evalue_threshold)
        out = out[ok_mask].copy()
        print(f"Apres filtre E-value>= {evalue_threshold:.2f} (NaN conserves): {len(out)}")

    # Dedoublonnage sur la relation causale: garder le lien le plus fort.
    out["abs_coef"] = out["coef"].abs()
    idx = out.groupby(["from", "to"], as_index=False)["abs_coef"].idxmax()["abs_coef"].values
    out = out.loc[idx].reset_index(drop=True)
    print(f"Apres dedoublonnage (from,to): {len(out)}")

    return out


def apply_chronological_orientation(df, level_map):
    out = df.copy()
    out["level_from"] = out["from"].map(level_map)
    out["level_to"] = out["to"].map(level_map)

    known_levels = out["level_from"].notna() & out["level_to"].notna()
    wrong_direction = known_levels & (out["level_from"] >= out["level_to"])

    removed = int(wrong_direction.sum())
    out = out[~wrong_direction].copy()
    print(f"Arretes supprimees par orientation chronologique: {removed}")
    print(f"Arretes apres orientation: {len(out)}")
    return out


def apply_weighted_transitive_reduction(df, tolerance=1.0):
    out = df.copy()
    g = nx.DiGraph()
    for _, r in out.iterrows():
        g.add_edge(r["from"], r["to"], abs_coef=float(abs(r["coef"])))

    to_remove = set()
    for a, c, data_ac in g.edges(data=True):
        direct = float(data_ac.get("abs_coef", 0.0))
        if direct <= 0:
            continue

        best_indirect = 0.0
        inter_nodes = set(g.successors(a)).intersection(set(g.predecessors(c)))
        for b in inter_nodes:
            if not g.has_edge(a, b) or not g.has_edge(b, c):
                continue
            indirect = float(g[a][b].get("abs_coef", 0.0)) * float(g[b][c].get("abs_coef", 0.0))
            best_indirect = max(best_indirect, indirect)

        if best_indirect > 0 and direct < (best_indirect * tolerance):
            to_remove.add((a, c))

    if to_remove:
        out = out[~out.apply(lambda r: (r["from"], r["to"]) in to_remove, axis=1)].copy()
    print(f"Arretes supprimees par reduction transitive ponderee: {len(to_remove)}")
    print(f"Arretes apres reduction transitive: {len(out)}")
    return out


def build_graph(links_df):
    g = nx.DiGraph()
    for _, row in links_df.iterrows():
        g.add_edge(
            str(row["from"]),
            str(row["to"]),
            coef=float(row["coef"]),
            abs_coef=float(abs(row["coef"])),
            p_value=float(row["p_stat"]),
            stability=float(row["stability_stat"]) if pd.notna(row["stability_stat"]) else np.nan,
            e_value=float(row["e_value_stat"]) if pd.notna(row["e_value_stat"]) else np.nan,
        )
    return g


def infer_node_levels(graph, level_map):
    levels = {n: level_map[n] for n in graph.nodes() if n in level_map}

    changed = True
    while changed:
        changed = False
        for node in graph.nodes():
            if node in levels:
                continue
            preds = [p for p in graph.predecessors(node) if p in levels]
            if preds:
                levels[node] = max(levels[p] for p in preds) + 1
                changed = True

    for node in graph.nodes():
        levels.setdefault(node, 0)

    return levels


def build_positions(graph, levels):
    level_to_nodes = {}
    for node in graph.nodes():
        level_to_nodes.setdefault(levels[node], []).append(node)

    pos = {}
    for level_idx, level in enumerate(sorted(level_to_nodes.keys())):
        nodes = sorted(level_to_nodes[level], key=lambda n: (-graph.degree(n), n))
        count = len(nodes)
        for i, node in enumerate(nodes):
            x = level_idx * HORIZONTAL_SPACING
            y = -(i - (count - 1) / 2.0) * VERTICAL_SPACING
            pos[node] = (x, y)
    return pos


def _edge_style_from_metrics(stability, stability_min, abs_coef, coef_min, coef_max):
    stab = float(stability) if np.isfinite(stability) else 1.0
    stab = min(max(stab, stability_min), 1.0)
    stab_ratio = 0.0 if (1.0 - stability_min) <= 1e-12 else (stab - stability_min) / (1.0 - stability_min)

    # Variation principale sur l'intensite de l'effet (|coef|), plus lisible quand la stabilite est quasi constante.
    if coef_max <= coef_min + 1e-12:
        coef_ratio = 0.5
    else:
        coef_ratio = (float(abs_coef) - coef_min) / (coef_max - coef_min)
        coef_ratio = min(max(coef_ratio, 0.0), 1.0)

    width_ratio = 0.75 * coef_ratio + 0.25 * stab_ratio
    width = EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * width_ratio
    gray = int(round(170 - 150 * width_ratio))  # gris clair -> noir selon intensite combinee
    color = f"#{gray:02x}{gray:02x}{gray:02x}"
    return width, color


def draw_semantic_dag(graph, levels, output_png, output_svg, stability_min=0.65, show=False):
    if graph.number_of_nodes() == 0:
        raise ValueError("Le graphe est vide, rien a dessiner.")

    pos = build_positions(graph, levels)
    unique_levels = sorted(set(levels.values()))
    fig_w = max(18, 4 + 2.6 * len(unique_levels))
    fig_h = max(11, 3 + 0.85 * graph.number_of_nodes())

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title("DAG final (pipeline pruning + orientation + transitive reduction)", fontsize=16, fontweight="bold", pad=14)

    centrality = nx.degree_centrality(graph)
    node_artists = {}

    for node in graph.nodes():
        x, y = pos[node]
        level = levels[node]
        c = centrality.get(node, 0.0)
        font_size = BASE_FONT_SIZE + (2 if c > 0.3 else 0)
        pad = BASE_BOX_PADDING + (0.2 if c > 0.3 else 0.0)

        t = ax.text(
            x,
            y,
            wrap_node_label(node),
            ha="center",
            va="center",
            fontsize=font_size,
            color=NODE_TEXT_COLOR,
            fontweight="bold",
            bbox=dict(
                boxstyle=f"round,pad={pad}",
                facecolor=LEVEL_COLORS.get(level, "#e9ecef"),
                edgecolor=NODE_EDGE_COLOR,
                linewidth=1.3,
            ),
            zorder=5,
        )
        node_artists[node] = t

    abs_vals = [float(d.get("abs_coef", 0.0)) for _, _, d in graph.edges(data=True)]
    coef_min = min(abs_vals) if abs_vals else 0.0
    coef_max = max(abs_vals) if abs_vals else 1.0

    for u, v, d in graph.edges(data=True):
        width, color = _edge_style_from_metrics(
            stability=d.get("stability", np.nan),
            stability_min=stability_min,
            abs_coef=d.get("abs_coef", 0.0),
            coef_min=coef_min,
            coef_max=coef_max,
        )
        ax.annotate(
            "",
            xy=pos[v],
            xycoords="data",
            xytext=pos[u],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.9,head_width=0.55",
                color=color,
                linewidth=width,
                alpha=EDGE_ALPHA,
                connectionstyle="arc3,rad=0.07",
                patchA=node_artists[u],
                patchB=node_artists[v],
            ),
            zorder=2,
        )

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    ax.set_xlim(min(xs) - 1.8, max(xs) + 1.8)
    ax.set_ylim(min(ys) - 2.2, max(ys) + 2.2)

    legend_lines = [
        "Layout: hierarchique par niveau causal",
        "Epaisseur arete: surtout |coef| (et stabilite)",
        "Couleur arete: intensite combinee",
        "  - fine claire: effet plus faible",
        "  - epaisse foncee: effet plus fort/stable",
        "Taille noeud: centralite de degre",
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor="#cccccc"),
    )

    ax.axis("off")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    print(f"Image PNG: {output_png}")
    print(f"Image SVG: {output_svg}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(show=False, p_threshold=0.05, stability_threshold=0.65, use_evalue=True, evalue_threshold=1.2, transitive_tolerance=1.0):
    level_map = build_level_map()
    links = load_links(LINKS_PATH)

    links = apply_causal_pruning(
        links,
        p_threshold=p_threshold,
        stability_threshold=stability_threshold,
        use_evalue=use_evalue,
        evalue_threshold=evalue_threshold,
    )
    links = apply_chronological_orientation(links, level_map)
    links = apply_weighted_transitive_reduction(links, tolerance=transitive_tolerance)

    links.to_csv(OUTPUT_PRUNED_CSV, index=False, encoding="utf-8-sig")
    print(f"CSV pruned: {OUTPUT_PRUNED_CSV}")

    graph = build_graph(links)
    print(f"Noeuds finaux: {graph.number_of_nodes()}")
    print(f"Aretes finales: {graph.number_of_edges()}")

    levels = infer_node_levels(graph, level_map)
    draw_semantic_dag(
        graph,
        levels,
        output_png=OUTPUT_DAG_PNG,
        output_svg=OUTPUT_DAG_SVG,
        stability_min=stability_threshold,
        show=show,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genere le DAG final avec pipeline d'elagage causal.")
    parser.add_argument("--show", action="store_true", help="Afficher la figure en plus de la sauvegarde.")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="Seuil p-value FDR pour conserver une arrete.")
    parser.add_argument("--stability-threshold", type=float, default=0.65, help="Seuil minimal de stabilite.")
    parser.add_argument("--evalue-threshold", type=float, default=1.2, help="Seuil minimal d'E-value.")
    parser.add_argument("--disable-evalue", action="store_true", help="Desactive le filtre E-value.")
    parser.add_argument("--transitive-tolerance", type=float, default=1.0, help="Tolérance pour suppression transitive ponderee.")
    args = parser.parse_args()

    main(
        show=args.show,
        p_threshold=args.p_threshold,
        stability_threshold=args.stability_threshold,
        use_evalue=not args.disable_evalue,
        evalue_threshold=args.evalue_threshold,
        transitive_tolerance=args.transitive_tolerance,
    )
