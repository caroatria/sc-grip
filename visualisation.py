import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def load_data(cluster_of_interest):
    df = pd.read_csv(f"predictions/predicted_interactions_cluster_{cluster_of_interest}.csv")
    print("number of edges (raw)", df.shape[0])

    # Only keep interactions labeled -0 (inhibition) or 1 (activation)
    df = df[df['predicted_label'] != -1]
    print("number of edges (filtered)", df.shape[0])
    return df

def plot_in_out_degree_dist(G):
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]

    # Plot histogram with transparency and separate colors
    plt.hist(in_degrees, bins=np.arange(0, max(in_degrees + out_degrees) + 2) - 0.5,
             alpha=0.6, color='blue', label='In-Degree')
    plt.hist(out_degrees, bins=np.arange(0, max(in_degrees + out_degrees) + 2) - 0.5,
             alpha=0.6, color='orange', label='Out-Degree')

    # Set x and y ticks (optional: customize based on actual data)
    plt.xticks(np.arange(0, max(in_degrees + out_degrees) + 1, 5))
    plt.yticks(np.arange(0, plt.gca().get_ylim()[1]+1, 5))

    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.title("In-Degree and Out-Degree Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/node_degree_distr_{cluster_of_interest}_{target_gene}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_activation_score_distribution(df, target_gene, cluster_of_interest):
    subset_df = df[df['edge'].str.contains(target_gene)]
    plt.figure(figsize=(8,6))
    subset_df['activation_score'].hist(bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Activation Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.title(f"Activation score distribution cluster {cluster_of_interest}: {target_gene}")
    plt.savefig(f"plots/act_score_{cluster_of_interest}_{target_gene}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Activation score distribution fig saved.")

def build_graph(df, threshold, focus_gene=None):
    full_G = nx.DiGraph()
    for _, row in df.iterrows():
        source, target = row['edge'].split("->")
        score = row['activation_score']
        full_G.add_edge(source, target, weight=score, label=f"{score:.2f}")

    if focus_gene:
        # 1. Get all edges involving the focus gene
        focus_edges = [(u, v) for u, v in full_G.edges if focus_gene in (u, v)]

        # 2. Build a subgraph with only those nodes
        focus_nodes = set([u for u, v in focus_edges] + [v for u, v in focus_edges])
        subG = full_G.subgraph(focus_nodes).copy()

        # 3. Apply threshold on out-degree
        hub_nodes = [n for n in subG.nodes if subG.out_degree(n) >= threshold]
        filtered_edges = [(u, v, subG[u][v]) for u, v in subG.edges if u in hub_nodes]

    else:
        # Full graph mode
        hub_nodes = [n for n in full_G.nodes if full_G.out_degree(n) >= threshold]
        filtered_edges = [(u, v, full_G[u][v]) for u, v in full_G.edges if u in hub_nodes]

    # Final graph
    G = nx.DiGraph()
    G.add_edges_from([(u, v, d) for u, v, d in filtered_edges])
    return G


def map_genes(G):
    gene_names = {node: node.split(".")[-1] for node in G.nodes}
    subgraph_gene_names = {node: gene_names[node] for node in G.nodes if node in gene_names}

    # Load ortholog translation table
    translation_df = pd.read_excel("/Users/work/Desktop/translation_tables/sd_to_mm/sd_to_mm.xlsx")
    translation_dict = dict(zip(translation_df['Sponge gene'], translation_df['Gene name']))

    # Handle missing values
    for k, v in translation_dict.items():
        if pd.isna(v):
            translation_dict[k] = k

    node_ortholog_dict = {}
    for node, gene_name in subgraph_gene_names.items():
        node_ortholog_dict[node] = translation_dict.get(node, gene_name)
    return node_ortholog_dict

def plot_graph(G, node_ortholog_dict, cluster_of_interest, target_gene=None):
    if len(G.nodes) == 0:
        print("No nodes to plot.")
        return

    plt.figure(figsize=(16, 12))
    pos = nx.kamada_kawai_layout(G)  # spring layout generally better for larger graphs

    source_nodes = {u for u, v in G.out_edges()}
    sink_nodes = set(G.nodes) - source_nodes
    edges = G.edges()

    weights = [G[u][v]['weight'] for u, v in edges]
    min_w, max_w = min(weights), max(weights)
    norm_weights = [(w - min_w) / (max_w - min_w) if max_w > min_w else 0.5 for w in weights]

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=source_nodes,
        node_size=1500,
        node_color="lightblue"
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=sink_nodes,
        node_size=300,
        node_color="lightgrey"
    )
    nx.draw_networkx_labels(G, pos, labels=node_ortholog_dict, font_size=10)

    nx.draw_networkx_edges(
        G, pos, edgelist=edges,
        edge_color=norm_weights, edge_cmap=plt.cm.coolwarm, width=2
    )

    edge_labels = {(u, v): f"{G[u][v]['weight']:.5f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    if target_gene:
        title = f"Gene interaction cluster {cluster_of_interest}: {target_gene}"
        filename = f"plots/interaction_graph_{cluster_of_interest}_{target_gene}.png"
    else:
        title = f"Gene interaction hubs in cluster {cluster_of_interest}"
        filename = f"plots/interaction_graph_{cluster_of_interest}_full.png"

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved graph to {filename}")

# === RUNNING SECTION ===

cluster_of_interest = 7
threshold = 1
target_gene = None #None #"SUB2.g5379"  # Set to None if you want full hub graph
n_rows = 50
df = load_data(cluster_of_interest)

if target_gene:
    subset_df = df[df['edge'].str.contains(target_gene)]
    subset_df = subset_df.head(n_rows)
    print("number of edges (for target gene)", subset_df.shape[0])
    #plot_activation_score_distribution(subset_df, target_gene, cluster_of_interest)
    G = build_graph(subset_df, threshold=threshold, focus_gene=target_gene)
else:
    df = df.head(n_rows)
    G = build_graph(df, threshold=threshold, focus_gene=None)

# plot_in_out_degree_dist(G)
node_ortholog_dict = map_genes(G)
plot_graph(G, node_ortholog_dict, cluster_of_interest, target_gene=target_gene)
