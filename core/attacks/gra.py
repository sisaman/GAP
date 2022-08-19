from typing import Annotated
import torch
import torch.nn.functional as F
from torchmetrics.functional import auroc
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from torch_geometric.transforms import RandomLinkSplit
from class_resolver.contrib.torch import activation_resolver
from core.attacks.base import AttackBase
from core.modules.base import Metrics
from core.console import console
from core.methods.node.base import NodeClassification
from core.models.mlp import MLP
from core.trainer.progress import TrainerProgress


class GraphReconstructionAttack (AttackBase):
    def __init__(self, 
                 num_attack_samples:    Annotated[int, dict(help='number of attack samples')] = 10000,
                 use_scores:            Annotated[bool, dict(help='use scores')] = False,
                 use_features:          Annotated[bool, dict(help='use features')] = False,
                 use_labels:            Annotated[bool, dict(help='use labels')] = False,
                 supervised:            Annotated[bool, dict(help='supervised attack')] = False,
                 hidden_dim:      Annotated[int,   dict(help='dimension of the hidden layers')] = 16,
                 num_layers:      Annotated[int,   dict(help='number of MLP layers')] = 2,
                 activation:      Annotated[str,   dict(help='type of activation function', choices=['relu', 'selu', 'tanh'])] = 'selu',
                 dropout:         Annotated[float, dict(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  dict(help='if true, then model uses batch normalization')] = True,
                 epochs:         Annotated[int,   dict(help='number of epochs for training')] = 100,
                 learning_rate:  Annotated[float, dict(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:   Annotated[float, dict(help='weight decay', option='--wd')] = 0.0,
                 ):
        assert use_scores or use_features or use_labels
        self.num_attack_samples = num_attack_samples
        self.use_scores = use_scores
        self.use_features = use_features
        self.use_labels = use_labels
        self.supervised = supervised
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        mlp_encoder = MLP(
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_fn=activation_resolver.make(activation),
            batch_norm=batch_norm,
        )

        self.model = GAE(mlp_encoder)

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        device = method.device
        data = data.to(device)

        if self.use_scores:
            self.target_metrics = method.fit(Data(**data.to_dict()), prefix='target/')
            data.scores = method.predict().to(device)

        feature_list = []

        if self.use_scores:
            feature_list.append(data.scores)

        if self.use_features:
            feature_list.append(data.x)

        if self.use_labels:
            num_classes = data.y.max().item() + 1
            labels = F.one_hot(data.y, num_classes).float()
            feature_list.append(labels)


        data = Data(x=torch.cat(feature_list, dim=1), adj_t=data.adj_t).to(device)
        train_data, val_data, test_data = RandomLinkSplit(
            num_val=int(self.num_attack_samples * .2),
            num_test=data.num_edges//2-self.num_attack_samples,
            is_undirected=True,
            split_labels=True,
            key='edge',
            add_negative_train_samples=True,
        )(data)

        console.info(f'train data: {train_data.pos_edge_index.size(1)} positive edges, {train_data.neg_edge_index.size(1)} negative edges')
        console.info(f'val data: {val_data.pos_edge_index.size(1)} positive edges, {val_data.neg_edge_index.size(1)} negative edges')
        console.info(f'test data: {test_data.pos_edge_index.size(1)} positive edges, {test_data.neg_edge_index.size(1)} negative edges')

        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        auc = self.fit(train_data, val_data, test_data)
        
        return {
            'attack/test/auc': auc,
        }

    def fit(self, train_data: Data, val_data: Data, test_data: Data) -> float:
        progress = TrainerProgress(
            num_epochs=self.epochs, 
            num_train_steps=1, 
            num_val_steps=1, 
            num_test_steps=1,
        )

        best_val_auc = final_test_auc = 0

        with progress:
            for epoch in range(1, self.epochs + 1):
                loss = self.train(data=train_data)
                val_auc = self.test(val_data)
                test_auc = self.test(test_data)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    final_test_auc = test_auc

                metrics = {
                    'train/loss': loss,
                    'val/auc': val_auc,
                    'test/auc': test_auc,
                }
                progress.update(task='epoch', metrics=metrics, advance=1)

        return final_test_auc

    def train(self, data: Data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        z: torch.Tensor = self.model.encode(data.x)
        loss = self.model.recon_loss(z, pos_edge_index=data.pos_edge_index, neg_edge_index=data.neg_edge_index)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, data: Data) -> tuple[float, float]:
        self.model.eval()
        z = self.model.encode(data.x)
        # return self.model.test(z, data.pos_edge_index, data.neg_edge_index)     # FIXME: use torchmetrics for faster evaluation on GPU
        y = torch.cat([data.pos_edge, data.neg_edge], dim=0).long()

        pos_pred = self.model.decoder(z, data.pos_edge_index, sigmoid=True)
        neg_pred = self.model.decoder(z, data.neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        return auroc(preds=pred, target=y).item() * 100
