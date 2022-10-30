from typing import Annotated
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from core.args.utils import ArgInfo
from core.attacks.base import ModelBasedAttack
from core.modules.base import Metrics
from core.methods.node.base import NodeClassification


class LinkStealingAttack (ModelBasedAttack):
    def __init__(self, 
                 num_attack_samples:    Annotated[int,  ArgInfo(help='number of attack samples')] = 10000,
                 use_scores:            Annotated[bool, ArgInfo(help='use scores')] = False,
                 use_features:          Annotated[bool, ArgInfo(help='use features')] = False,
                 use_labels:            Annotated[bool, ArgInfo(help='use labels')] = False,
                 **kwargs:              Annotated[dict, ArgInfo(help='extra options passed to base class', bases=[ModelBasedAttack])]
                 ):
        super().__init__(**kwargs)
        assert use_scores or use_features or use_labels
        self.num_attack_samples = num_attack_samples
        self.use_scores = use_scores
        self.use_features = use_features
        self.use_labels = use_labels

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        attack_metrics = super().execute(method, data)
        return {
            **(self.target_metrics if self.use_scores else {}),
            **attack_metrics,
        }

    def prepare_attack_dataset(self, method: NodeClassification, data: Data) -> Data:
        device = method.device
        data = data.to(device)

        if self.use_scores:
            self.target_metrics = method.fit(Data(**data.to_dict()), prefix='target/')
            data.scores = method.predict().to(device)

        # convert adj_t to edge_index
        edge_index_directed = torch.cat(data.adj_t.t().coo()[:-1]).view(2,-1)

        # exclude reverse edges
        mask = edge_index_directed[0] < edge_index_directed[1]
        edge_index_undirected = edge_index_directed[:,mask]

        # calculate sample size
        total_pos = edge_index_undirected.size(1)
        total_neg = data.num_nodes * (data.num_nodes - 1) // 2 - total_pos
        num_half = min(total_neg, total_pos, self.num_attack_samples // 2)

        # randomly sample num_half positive (existing) edges and num_half negative (non-existing) edges
        perm = torch.randperm(total_pos, device=device)[:num_half]
        pos_idx = edge_index_undirected[:, perm]
        neg_idx = negative_sampling(edge_index_directed, num_nodes=data.num_nodes, num_neg_samples=num_half, method='sparse', force_undirected=True)
        attack_edge_index = torch.cat([pos_idx, neg_idx], dim=1)

        # encode attack edges
        x = self.encode(data, attack_edge_index)
        y = torch.cat([
            torch.ones(num_half, dtype=torch.long, device=device),
            torch.zeros(num_half, dtype=torch.long, device=device),
        ])

        # shuffle data
        num_total = 2 * num_half
        perm = torch.randperm(num_total, device=device)
        x, y = x[perm], y[perm]

        # train test split
        num_train = int(num_total * 0.5)
        num_test = int(num_total * 0.4)
        
        train_mask = y.new_zeros(num_total, dtype=torch.bool)
        train_mask[:num_train] = True

        val_mask = y.new_zeros(num_total, dtype=torch.bool)
        val_mask[num_train:-num_test] = True

        test_mask = x.new_zeros(num_total, dtype=torch.bool)
        test_mask[-num_test:] = True

        # create data object
        attack_data = Data(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        return attack_data

    def encode(self, data: Data, idx: Tensor) -> Tensor:
        feature_list = []

        if self.use_scores:
            f_u = data.scores[idx[0]]
            f_v = data.scores[idx[1]]
            e_u = Categorical(probs=f_u).entropy().view(-1, 1)
            e_v = Categorical(probs=f_v).entropy().view(-1, 1)
            feature_list += [
                self.distance(f_u, f_v),
                self.pairwise(f_u, f_v),
                self.pairwise(e_u, e_v),
            ]

        if self.use_features:
            F_u = data.x[idx[0]]
            F_v = data.x[idx[1]]
            feature_list += [
                self.distance(F_u, F_v),
                self.pairwise(F_u, F_v),
            ]

        if self.use_labels:
            num_classes = data.y.max().item() + 1
            labels = F.one_hot(data.y, num_classes).float()
            y_u = labels[idx[0]]
            y_v = labels[idx[1]]
            feature_list += [
                self.distance(y_u, y_v),
                self.pairwise(y_u, y_v),
            ]

        x = torch.cat(feature_list, dim=1)
        x = (x - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0])
        return x

    @staticmethod
    def distance(u: Tensor, v: Tensor) -> Tensor:
        eps = 1e-8
        u_centered = u - u.mean(dim=0)
        v_centered = v - v.mean(dim=0)

        dist_list = [
            cosine_similarity(u, v),                                                    # cosine similarity
            torch.norm(u - v, p=2, dim=1),                                              # euclidean distance
            (u_centered * v_centered).sum(dim=1) / (
                torch.norm(u_centered, p=2, dim=1) * torch.norm(v_centered, p=2, dim=1) + eps
            ),                                                                          # correlation
            torch.norm(u - v, p=torch.inf, dim=1),                                      # chebyshev distance
            torch.norm(u - v, p=1, dim=1) / (torch.norm(u + v, p=1, dim=1) + eps),      # Bray-Curtis distance
            torch.norm(u - v, p=1, dim=1),                                              # manhattan distance
            (torch.abs(u - v) / (torch.abs(u + v) + eps)).sum(dim=1),                   # canberra distance
            torch.norm(u - v, p=2, dim=1) ** 2,                                         # squared euclidean distance
        ]

        return torch.stack(dist_list, dim=1)

    @staticmethod
    def pairwise(u: Tensor, v: Tensor) -> Tensor:
        op_list = [
            (u + v) / 2,        # average
            u * v,              # hadamard
            torch.abs(u - v),   # weighted l1
            (u - v) ** 2,       # weighted l2
        ]

        return torch.cat(op_list, dim=1)
