import os
import uuid
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from typing import Iterable, Literal, Optional
from pysrc.loggers import Logger
from torchmetrics import MeanMetric
from pysrc.privacy.algorithms import NoisySGD
from pysrc.trainer.progress import TrainerProgress
from pysrc.trainer.typing import TrainerStage, Metrics


class Trainer:
    def __init__(self,
                 epochs:        int = 100,
                 patience:      int = 0,
                 val_interval:  int = 1,
                 use_amp:       bool = False,
                 monitor:       str = 'val/acc',
                 monitor_mode:  Literal['min', 'max'] = 'max',
                 device:        Literal['cpu', 'cuda'] = 'cuda',
                 noisy_sgd:     Optional[NoisySGD] = None,
                 ):

        self.epochs = epochs
        self.patience = patience
        self.val_interval = val_interval
        self.use_amp = use_amp
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.device = device
        self.noisy_sgd = noisy_sgd
        
        self.logger = Logger.get_instance()
        
        self.metrics = {
            'train/loss': MeanMetric(compute_on_step=False).to(device),
            'train/acc': MeanMetric(compute_on_step=False).to(device),
            'val/loss': MeanMetric(compute_on_step=False).to(device),
            'val/acc': MeanMetric(compute_on_step=False).to(device),
            'test/acc': MeanMetric(compute_on_step=False).to(device),
        }

    def reset(self):
        self.model: Module = None
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_metrics: dict[str, object] = None
        self.checkpoint_path: str = None

        for metric in self.metrics.values():
            metric.reset()

    def aggregate_metrics(self, stage: TrainerStage='train') -> Metrics:
        metrics = {}

        for metric_name, metric_value in self.metrics.items():
            if metric_name.startswith(stage):
                value = metric_value.compute()
                metric_value.reset()
                if torch.is_tensor(value):
                    value = value.item()
                metrics[metric_name] = value

        return metrics

    def performs_better(self, metrics: Metrics) -> bool:
        if self.best_metrics is None:
            return True
        elif self.monitor_mode == 'max':
            return metrics[self.monitor] > self.best_metrics[self.monitor]
        elif self.monitor_mode == 'min':
            return metrics[self.monitor] < self.best_metrics[self.monitor]
        else:
            raise ValueError(f'Unknown metric mode: {self.monitor_mode}')

    def load_best_model(self) -> Module:
        if self.checkpoint_path:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            return self.model
        else:
            raise Exception('No checkpoint found')

    def fit(self, 
            model: Module, 
            optimizer: Optimizer, 
            train_dataloader: Iterable, 
            val_dataloader: Optional[Iterable]=None, 
            test_dataloader: Optional[Iterable]=None, 
            checkpoint: bool=False
            ) -> Metrics:

        self.reset()
        self.model = model.to(self.device)
        self.model.train()
        self.optimizer = optimizer

        if self.noisy_sgd:
            self.model, self.optimizer, train_dataloader = self.noisy_sgd(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_dataloader,
            )

        if checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            self.checkpoint_path = os.path.join('checkpoints', f'{uuid.uuid1()}.pt')

        if val_dataloader is None:
            val_dataloader = []

        if test_dataloader is None:
            test_dataloader = []

        self.progress = TrainerProgress(
            num_epochs=self.epochs, 
            num_train_steps=len(train_dataloader), 
            num_val_steps=len(val_dataloader), 
            num_test_steps=len(test_dataloader),
        )
        
        with self.progress:
            num_epochs_without_improvement = 0
            
            for epoch in range(1, self.epochs + 1):
                metrics = {'epoch': epoch}

                # train loop
                train_metrics = self.loop(train_dataloader, stage='train')
                metrics.update(train_metrics)

                # validation loop
                if val_dataloader:
                    val_metrics = self.loop(val_dataloader, stage='val')
                    metrics.update(val_metrics)

                # test loop
                if test_dataloader:
                    test_metrics = self.loop(test_dataloader, stage='test')
                    metrics.update(test_metrics)
                    
                # update best metrics
                if val_dataloader and self.val_interval:
                    if epoch % self.val_interval == 0:
                        if self.performs_better(metrics):
                            self.best_metrics = metrics
                            num_epochs_without_improvement = 0

                            if checkpoint:
                                torch.save(self.model.state_dict(), self.checkpoint_path)
                        else:
                            num_epochs_without_improvement += 1
                            if num_epochs_without_improvement >= self.patience > 0:
                                break
                else:
                    self.best_metrics = metrics

                # log and update progress
                if self.logger: self.logger.log(metrics)
                self.progress.update('epoch', metrics=metrics, advance=1)
        
        if self.logger: self.logger.log_summary(self.best_metrics)
        return self.best_metrics

    def loop(self, dataloader: Iterable, stage: TrainerStage) -> Metrics:
        self.model.train(stage == 'train')
        self.progress.update(stage, visible=len(dataloader) > 1)

        for batch in dataloader:
            metrics = self.step(batch, stage)
            for item in metrics:
                self.metrics[item].update(metrics[item], weight=len(batch))
            self.progress.update(stage, advance=1)

        self.progress.reset(stage, visible=False)
        return self.aggregate_metrics(stage)

    def step(self, batch, stage: TrainerStage) -> Metrics:
        if stage == 'train':
            self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(stage == 'train')
            loss, metrics = self.model.step(batch, stage=stage)
            torch.set_grad_enabled(prev)
        
        if stage == 'train' and loss is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return metrics