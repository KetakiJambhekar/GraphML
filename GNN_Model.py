import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear

from Load_Graph import pyg_data


class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = Linear(output_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return self.fc(x)


# Initialize features if not provided
if not hasattr(pyg_data, 'x'):
    pyg_data.x = torch.eye(pyg_data.num_nodes)  # Identity matrix as initial features

model = GCNModel(input_dim=pyg_data.num_features, hidden_dim=16, output_dim=2)
print(model)

# Example training loop (for simplicity, we'll skip training process here)
