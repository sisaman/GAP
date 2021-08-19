import torch
import torch.nn.functional as F
from torch.nn import Dropout, SELU
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv


class GNN(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv1 = None
        self.conv2 = None
        self.dropout = Dropout(p=dropout)
        self.activation = SELU(inplace=True)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(g, x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class GCN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)


class GAT(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        heads = 4
        self.conv1 = GATConv(input_dim, hidden_dim, num_heads=heads)
        self.conv2 = GATConv(heads * hidden_dim, output_dim, num_heads=1)

    def forward(self, g, x):
        x = self.conv1(g, x).flatten(1)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(g, x).mean(1)
        return x


class GraphSAGE(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_dim, output_dim, aggregator_type='mean')


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 model: dict(help='base GNN model', choices=['gcn', 'sage', 'gat']) = 'gcn',
                 hidden_dim: dict(help='dimension of the hidden layers') = 16,
                 dropout: dict(help='dropout rate (between zero and one)') = 0.0,
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.gnn = {'gcn': GCN, 'sage': GraphSAGE, 'gat': GAT}[model](
            input_dim=input_dim,
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, g):
        x = g.ndata['feat']
        h = self.gnn(g, x)
        y = F.log_softmax(h, dim=1)
        return y

    def training_step(self, g):
        train_mask = g.ndata['train_mask']
        y_true = g.ndata['label'][train_mask]
        y_pred = self(g)[train_mask]
        
        loss = F.nll_loss(input=y_pred, target=y_true)
        acc = self.accuracy(input=y_pred, target=y_true)
        
        metrics = {'train/loss': loss.item(), 'train/acc': acc}
        return loss, metrics

    def validation_step(self, g):
        y_pred = self(g)
        y_true = g.ndata['label']

        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

        metrics = {
            'val/loss': F.nll_loss(input=y_pred[val_mask], target=y_true[val_mask]).item(),
            'val/acc': self.accuracy(input=y_pred[val_mask], target=y_true[val_mask]),
            'test/acc': self.accuracy(input=y_pred[test_mask], target=y_true[test_mask]),
        }

        return metrics

    @staticmethod
    def accuracy(input, target):
        input = input.argmax(dim=1) if len(input.size()) > 1 else input
        target = target.argmax(dim=1) if len(target.size()) > 1 else target
        return (input == target).float().mean().item()

    def reset_parameters(self):
        self.gnn.reset_parameters()
