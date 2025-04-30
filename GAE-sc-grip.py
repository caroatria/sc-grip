import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader, Batch
import pandas as pd
import numpy as np
import scanpy as sc


class GAEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAEModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.decoder = torch.nn.Bilinear(out_channels, out_channels, 1)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return torch.sigmoid(self.decoder(src, dst)).squeeze()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)


def load_cell_graphs(expression_file, edge_file, selected_cells=None):
    expr_df = pd.read_csv(expression_file, index_col=0)
    edge_df = pd.read_csv(edge_file, index_col=0)

    if selected_cells is not None:
        expr_df = expr_df.loc[selected_cells]

    adj = torch.tensor(edge_df.values, dtype=torch.float)
    edge_index = (adj > 0).nonzero(as_tuple=False).T  # Fixed graph

    data_list = []
    for i in range(expr_df.shape[0]):
        expr = torch.tensor(expr_df.iloc[i].values.reshape(-1, 1), dtype=torch.float)
        data = Data(x=expr, edge_index=edge_index)
        data_list.append(data)

    return data_list, edge_index


def train(model, dataset, num_epochs=30):
    optimizer = torch.optim.Adam(model.parameters(),lr= 0.00001) # lr=0.0001
    criterion = torch.nn.BCELoss()
    loader = DataLoader(dataset, batch_size=1, shuffle=True) 

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            preds = model(data.x, data.edge_index)
            # Define interaction label: 1 if similar, 0 if different expression
            tf_expr = data.x[data.edge_index[0]] ##inhibiting relationship
            tgt_expr = data.x[data.edge_index[1]]  ##ac
            cos_sim = F.cosine_similarity(tf_expr, tgt_expr)
            labels = (cos_sim > 0.5).float()

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataset):.4f}")


def get_consensus_edge_labels(model, dataset, low_thresh=0.4, high_thresh=0.6):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=1):
            pred = model(data.x, data.edge_index)
            all_preds.append(pred)

    stacked = torch.stack(all_preds)  # [n_cells x n_edges]
    avg_preds = stacked.mean(dim=0)

    # Three classes: 0 = inhibition, 1 = activation, -1 = uncertain
    ternary_labels = torch.full_like(avg_preds, -1)  # Initialize as uncertain
    ternary_labels[avg_preds <= low_thresh] = 0      # Inhibition
    ternary_labels[avg_preds >= high_thresh] = 1     # Activation

    return ternary_labels, avg_preds

if __name__ == "__main__":
    # Load AnnData and subset cluster
    adata = sc.read_h5ad("/Users/work/Desktop/subdom.h5ad")
    expression_file = '/Users/work/Desktop/expression_matrix.csv'
    edge_file = '/Users/work/Desktop/suberites_presence_absence.csv'

    cluster_key = 'clusters'
    target_cluster = '7'
    print("analysing cluster...", target_cluster)
    selected = adata[adata.obs[cluster_key] == target_cluster]
    selected_indices = selected.obs_names.to_list()

    print(f"Selected {len(selected_indices)} cells from cluster {target_cluster}")

    # Load per-cell graphs
    dataset, edge_index = load_cell_graphs(expression_file, edge_file, selected_cells=selected_indices)
    print(f"Loaded {len(dataset)} graphs (1 per cell)")

    in_channels = dataset[0].x.shape[1]
    model = GAEModel(in_channels=in_channels, out_channels=64)

    # Train on the full dataset (no split, because it's not supervised)
    train(model, dataset, num_epochs=15)

    # Predict final interaction labels
    edge_labels, edge_probs = get_consensus_edge_labels(model, dataset)

    # Save results
    edge_names = pd.read_csv(edge_file, index_col=0).index.to_list()
    tf_targets = pd.read_csv(edge_file, index_col=0).columns.to_list()
    edge_df = pd.read_csv(edge_file, index_col=0)

    edge_map = (edge_df.values > 0).nonzero()
    edge_names_flat = [f"{edge_names[i]}->{tf_targets[j]}" for i, j in zip(*edge_map)]

    pd.DataFrame({
    'edge': edge_names_flat,
    'activation_score': edge_probs.cpu().numpy(),
    'predicted_label': edge_labels.cpu().numpy().astype(int)
    }).to_csv(f'predictions/predicted_interactions_cluster_{target_cluster}.csv', index=False)


    # torch.save(model.state_dict(), 'model_cluster_{}.pth'.format(target_cluster))
    print("Done. Predictions saved.")
