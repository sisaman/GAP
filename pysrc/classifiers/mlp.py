from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from pysrc.classifiers.base import ClassifierBase, Metrics, Stage
from pysrc.models import MLP


class MLPClassifier(MLP, ClassifierBase):
    def __init__(self, 
                 num_classes: int,
                 hidden_dim: int = 16,  
                 num_layers: int = 2, 
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 ):

        super().__init__(
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
        )

    def step(self, batch: tuple[Tensor, Tensor], stage: Stage) -> tuple[Optional[Tensor], Metrics]:
        x, y = batch
        preds = F.log_softmax(self(x), dim=-1)
        acc = preds.argmax(dim=1).eq(y).float().mean() * 100
        metrics = {f'{stage}/acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=y)
            metrics[f'{stage}/loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        h = self(x)
        return torch.softmax(h, dim=-1)
