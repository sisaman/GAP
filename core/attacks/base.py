from abc import ABC, abstractmethod
from typing import Annotated
from torch_geometric.data import Data
from torchmetrics.functional import auroc
from core.args.utils import remove_prefix
from core.console import console
from core.classifiers.base import Metrics
from core.methods.base import NodeClassification
from core.methods.mlp import MLP


class AttackBase(ABC):
    @abstractmethod
    def execute(self, method: NodeClassification, data: Data) -> Metrics: pass


class ModelBasedAttack(MLP, AttackBase):
    def __init__(self, **kwargs:  Annotated[dict, dict(help='attack method kwargs', bases=[MLP], prefixes=['attack_'], exclude=['device', 'use_amp'])]):
        super().__init__(
            num_classes=2,  # either member or non-member
            **remove_prefix(kwargs, prefix='attack_'),
        )

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        # construct attack dataset
        attack_data = self.prepare_attack_dataset(method, data)
        console.debug(f'attack dataset: {attack_data.train_mask.sum()} train nodes, {attack_data.val_mask.sum()} val nodes, {attack_data.test_mask.sum()} test nodes')

        # train attack model and get attack accuracy
        console.info('step 3: training attack model')
        attack_metrics = self.fit(attack_data, prefix='attack/')
        
        # compute extra attack metrics
        preds = self.predict()[attack_data.test_mask, 1]
        target = attack_data.y[attack_data.test_mask]
        attack_metrics['attack/test/auc'] = auroc(preds=preds, target=target).item() * 100

        return attack_metrics

    @abstractmethod
    def prepare_attack_dataset(self, method: NodeClassification, data: Data) -> Data: pass
