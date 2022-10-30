from abc import ABC, abstractmethod
from typing import Annotated
import torch
from torch_geometric.data import Data
from torchmetrics.functional import auroc, roc
from core.args.utils import ArgInfo, remove_prefix
from core import console
from core.modules.base import Metrics
from core.methods.node.base import NodeClassification
from core.methods.node import MLP


class AttackBase(ABC):
    @abstractmethod
    def execute(self, method: NodeClassification, data: Data) -> Metrics: pass
    @abstractmethod
    def reset(self): pass


class ModelBasedAttack(AttackBase):
    def __init__(self, **kwargs:  Annotated[dict, ArgInfo(help='attack method kwargs', bases=[MLP], prefixes=['attack_'])]):
        self.attack_model = MLP(
            num_classes=2, # either member or non-member
            **remove_prefix(kwargs, 'attack_')
        )

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        # construct attack dataset
        attack_data = self.prepare_attack_dataset(method, data)
        console.debug(f'attack dataset: {attack_data.train_mask.sum()} train nodes, {attack_data.val_mask.sum()} val nodes, {attack_data.test_mask.sum()} test nodes')

        # train attack model and get attack accuracy
        console.info('step 3: training attack model')
        self.attack_model.reset_parameters()
        attack_metrics = self.attack_model.fit(attack_data, prefix='attack/')
        
        # compute extra attack metrics
        preds = self.attack_model.predict()[attack_data.test_mask, 1]
        target = attack_data.y[attack_data.test_mask]
        attack_metrics['attack/test/auc'] = auroc(preds=preds, target=target).item() * 100
        fpr, tpr, _ = roc(preds=preds, target=target)
        attack_metrics['attack/test/tpr@0.01fpr'] = tpr[torch.where(fpr<=.01)[0][-1]].item() * 100

        return attack_metrics

    def reset(self):
        self.attack_model.reset_parameters()

    @abstractmethod
    def prepare_attack_dataset(self, method: NodeClassification, data: Data) -> Data: pass
