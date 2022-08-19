from typing import Callable, Optional
import torch
from torch import Tensor
from core.models import MLP
from torch_geometric.data import Data
from torch_geometric.nn import GAE, InnerProductDecoder
from torchmetrics.functional import auroc
from core.modules.base import TrainableModule, Stage, Metrics


class MLPLinkPredictor(TrainableModule):
    def __init__(self, *,
                 hidden_dim: int = 16,  
                 num_layers: int = 2, 
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 ):

        super().__init__()

        encoder = MLP(
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=True,
        )

        self.model = GAE(encoder=encoder, decoder=InnerProductDecoder())

    def forward(self, x: Tensor) -> Tensor:
        return self.model.encode(x)

    def step(self, data: Data, stage: Stage) -> tuple[Optional[Tensor], Metrics]:
        z = self(data.x)
        y = torch.cat([data.pos_edge, data.neg_edge], dim=0).long()
        pos_pred = self.model.decoder(z, data.pos_edge_index, sigmoid=True)
        neg_pred = self.model.decoder(z, data.neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        auc = auroc(preds=pred, target=y).item() * 100
        metrics = {'auc': auc}

        loss = None
        if stage != 'test':
            loss = self.model.recon_loss(z, pos_edge_index=data.pos_edge_index, neg_edge_index=data.neg_edge_index)
            metrics['loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        z = self(data.x)
        pos_pred = self.model.decoder(z, data.pos_edge_index, sigmoid=True)
        neg_pred = self.model.decoder(z, data.neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        return pred

    def reset_parameters(self):
        return self.model.reset_parameters()