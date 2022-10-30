from typing import Annotated
import numpy as np
import torch
from torch_geometric.data import Data
from core.args.utils import ArgInfo
from core.attacks.base import AttackBase
from core.modules.base import Metrics
from core import console
from core.methods.node.base import NodeClassification
from sklearn.metrics import roc_curve, roc_auc_score


class LikelihoodRatioAttack(AttackBase):
    def __init__(self, num_shadow: Annotated[int, ArgInfo(help='number of shadow models')] = 16):
        self.num_shadow = num_shadow

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        device = method.device
        data = data.to(device)
        S = self.num_shadow
        N = data.num_nodes
        indices = torch.multinomial(input=torch.ones(N, S), num_samples=S//2, replacement=False).to(device)
        mask = torch.zeros(N, S, dtype=bool, device=device).scatter_(1, indices, True)
        score_list = []

        for i in range(S):
            test_mask = mask[:, i]
            train_mask, val_mask = self.generate_train_val_masks(test_mask, train_ratio=1)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            console.info(f'training shadow model {i+1}/{S}')
            method.reset_parameters()
            method.fit(Data(**data.to_dict()), prefix=f'shadow_{i}/')
            probs = method.predict()
            py = probs.gather(1, data.y.view(-1, 1)).squeeze_()
            phi = torch.logit(py, eps=1e-7)
            score_list.append(phi)

        logits = torch.stack(score_list, dim=1)
        mean_in = logits[~mask].view(N, S//2).mean(dim=1, keepdim=True)
        mean_out = logits[mask].view(N, S//2).mean(dim=1, keepdim=True)
        std_in = logits[~mask].view(N, S//2).std(dim=1, keepdim=True)
        std_out = logits[mask].view(N, S//2).std(dim=1, keepdim=True)
        dist_in = torch.distributions.Normal(mean_in, std_in.clamp_min_(1e-9))
        dist_out = torch.distributions.Normal(mean_out, std_out.clamp_min_(1e-9))
        logp_in: torch.Tensor = dist_in.log_prob(logits)
        logp_out: torch.Tensor = dist_out.log_prob(logits)
        ratio = (logp_in - logp_out).exp_().clamp_max_(1e20)
        preds = ratio.T.reshape(-1).cpu()
        target = (~mask).T.reshape(-1).long().cpu()
        auc = roc_auc_score(y_score=preds, y_true=target) * 100
        fpr, tpr, _ = roc_curve(y_score=preds, y_true=target)
        tpr_at_low_fpr = tpr[np.where(fpr<=.01)[0][-1]] * 100
        return {
            'attack/test/auc': auc,
            'attack/test/tpr@0.01fpr': tpr_at_low_fpr,
        }
        
    def generate_train_val_masks(self, test_mask: torch.Tensor, train_ratio: float) -> tuple[torch.Tensor]:
        train_val_mask = ~test_mask
        train_val_indices = torch.nonzero(train_val_mask).view(-1)
        num_train_val = train_val_indices.size(0)
        num_train = int(num_train_val * train_ratio)
        perm = torch.randperm(num_train_val, device=train_val_indices.device)
        train_val_indices = train_val_indices[perm]
        val_mask = train_val_mask.clone()
        val_mask[train_val_indices[:num_train]] = False
        train_mask = train_val_mask
        train_mask[train_val_indices[num_train:]] = False
        return train_mask, val_mask
