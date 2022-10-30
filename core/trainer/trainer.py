import os
import uuid
import torch
from torch.types import Number
from torch.optim import Optimizer
from typing import Annotated, Iterable, Literal, Optional
from core.args.utils import ArgInfo
from core.loggers import Logger
from torchmetrics import MeanMetric
from core.trainer.progress import TrainerProgress
from core.modules.base import Metrics, Stage, TrainableModule


class Trainer:
    def __init__(self,
                 patience:      int = 0,
                 monitor:       str = 'val/acc',
                 monitor_mode:  Literal['min', 'max'] = 'max',
                 val_interval:  Annotated[int, ArgInfo(help='interval of validation')] = 1,
                 ):

        assert monitor_mode in ['min', 'max']

        self.patience = patience
        self.val_interval = val_interval
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        
        # trainer internal state
        self.model: TrainableModule = None
        self.metrics: dict[str, MeanMetric] = {}

    def reset(self):
        self.model = None
        self.metrics = {}

    def update_metrics(self, metric_name: str, metric_value: object, weight: int = 1) -> None:
        # if this is a new metric, add it to self.metrics
        device = metric_value.device if torch.is_tensor(metric_value) else 'cpu'
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MeanMetric(compute_on_step=False).to(device)

        # update the metric
        self.metrics[metric_name].update(metric_value, weight=weight)

    def aggregate_metrics(self, stage: Stage='train') -> Metrics:
        metrics = {}

        for metric_name, metric_value in self.metrics.items():
            if stage in metric_name.split('/'):
                value = metric_value.compute()
                metric_value.reset()
                if torch.is_tensor(value):
                    value = value.item()
                metrics[metric_name] = value

        return metrics

    def is_better(self, current_metric: Number, previous_metric: Number) -> bool:
        if self.monitor_mode == 'max':
            return current_metric > previous_metric
        elif self.monitor_mode == 'min':
            return current_metric < previous_metric
        else:
            raise ValueError(f'Unknown metric mode: {self.monitor_mode}')

    def fit(self, 
            model: TrainableModule, 
            epochs: int,
            optimizer: Optimizer, 
            train_dataloader: Iterable, 
            val_dataloader: Optional[Iterable]=None, 
            test_dataloader: Optional[Iterable]=None, 
            checkpoint: bool=False,
            prefix: str = ''
            ) -> Metrics:

        self.model = model
        self.optimizer = optimizer
        monitor_key = f'{prefix}{self.monitor}'

        if checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = os.path.join('checkpoints', f'{uuid.uuid1()}.pt')
            torch.save(self.model.state_dict(), checkpoint_path)

        if val_dataloader is None:
            val_dataloader = []

        if test_dataloader is None:
            test_dataloader = []

        self.progress = TrainerProgress(
            num_epochs=epochs, 
            num_train_steps=len(train_dataloader), 
            num_val_steps=len(val_dataloader), 
            num_test_steps=len(test_dataloader),
        )
        
        with self.progress:
            best_metrics = None
            num_epochs_without_improvement = 0
            
            for epoch in range(1, epochs + 1):
                metrics = {f'{prefix}epoch': epoch}

                # train loop
                train_metrics = self.loop(train_dataloader, stage='train', prefix=prefix)
                metrics.update(train_metrics)
                    
                # validation loop
                if val_dataloader and self.val_interval and epoch % self.val_interval == 0:

                    val_metrics = self.loop(val_dataloader, stage='val', prefix=prefix)
                    metrics.update(val_metrics)

                    if best_metrics is None or self.is_better(metrics[monitor_key], best_metrics[monitor_key]):
                        best_metrics = metrics
                        num_epochs_without_improvement = 0

                        if checkpoint:
                            torch.save(self.model.state_dict(), checkpoint_path)
                    else:
                        num_epochs_without_improvement += 1
                        if num_epochs_without_improvement >= self.patience > 0:
                            break

                # test loop
                if test_dataloader:
                    test_metrics = self.loop(test_dataloader, stage='test', prefix=prefix)
                    metrics.update(test_metrics)

                # log and update progress
                Logger.get_instance().log(metrics)
                self.progress.update(task='epoch', metrics=metrics, advance=1)

        if best_metrics is None:
            best_metrics = metrics
        else:
            # load best model if checkpointing is enabled
            if checkpoint:
                self.model.load_state_dict(torch.load(checkpoint_path))

        # log and return best metrics
        Logger.get_instance().log_summary(best_metrics)
        return best_metrics

    def test(self, dataloader: Iterable, prefix: str = '') -> Metrics:
        metrics = self.loop(dataloader, stage='test', prefix=prefix)
        return metrics

    def loop(self, dataloader: Iterable, stage: Stage, prefix: str) -> Metrics:
        self.model.train(stage == 'train')
        self.progress.update(stage, visible=len(dataloader) > 1)

        for batch in dataloader:
            metrics = self.step(batch, stage, prefix)
            for item in metrics:
                self.update_metrics(item, metrics[item], weight=len(batch))
            self.progress.update(stage, advance=1)

        self.progress.reset(stage, visible=False)
        return self.aggregate_metrics(stage)

    def step(self, batch, stage: Stage, prefix: str) -> Metrics:
        if stage == 'train':
            self.optimizer.zero_grad(set_to_none=True)

        grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(stage == 'train')
        loss, metrics = self.model.step(batch, stage=stage)
        torch.set_grad_enabled(grad_state)
        
        if stage == 'train' and loss is not None:
            loss.backward()
            self.optimizer.step()

        return {f'{prefix}{stage}/{key}': value for key, value in metrics.items()}
