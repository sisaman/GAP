from typing import Annotated

import torch
import torch.nn.functional as F
from core.args.utils import remove_prefix
from core.attacks.base import AttackBase
from core.console import console
from core.methods.link.mlp.mlp import MLPLinkPredictionMethod
from core.methods.node.base import NodeClassification
from core.modules.base import Metrics
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit


class GraphReconstructionAttack (AttackBase):
    def __init__(self, 
                 num_attack_samples:    Annotated[int, dict(help='number of attack samples')] = 10000,
                 use_scores:            Annotated[bool, dict(help='use scores')] = True,
                 use_features:          Annotated[bool, dict(help='use features')] = False,
                 use_labels:            Annotated[bool, dict(help='use labels')] = False,
                 supervised:            Annotated[bool, dict(help='supervised attack')] = False,
                 **attack_args:         Annotated[dict, dict(help='attack method kwargs', bases=[MLPLinkPredictionMethod], prefixes=['attack_'])]
                 ):
        assert use_scores or use_features or use_labels
        self.num_attack_samples = num_attack_samples
        self.use_scores = use_scores
        self.use_features = use_features
        self.use_labels = use_labels
        self.supervised = supervised

        self.attack_method = MLPLinkPredictionMethod(**remove_prefix(attack_args, 'attack_'))

    def reset(self):
        self.attack_method.reset_parameters()

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

        self.attack_method.reset_parameters()
        train_metrics = self.attack_method.fit(train_data=train_data, val_data=val_data, prefix='attack/')
        test_metrics = self.attack_method.test(data=test_data, prefix='attack/')
        return {**train_metrics, **test_metrics}
