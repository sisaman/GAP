from typing import Text
from rich.console import Group
from rich.padding import Padding
from rich.table import Column, Table
from console import console
import os, uuid
import torch
from args import support_args
from loggers import Logger
from torchmetrics import MeanMetric
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn


@support_args
class Trainer:
    def __init__(self,
                 epochs:        dict(help='number of training epochs') = 100,
                 patience:      dict(help='early-stopping patience window size') = 0,
                 val_interval:  dict(help='number of epochs to wait for validation', type=int) = 1,
                 use_amp:       dict(help='use automatic mixed precision training') = False,
                 monitor:       dict(help='metric to monitor') = 'val/acc',
                 monitor_mode:  dict(help='monitor mode', choices=['min', 'max']) = 'max',
                 device = 'cuda',
                 dp_mechanism = None,
                 ):

        self.epochs = epochs
        self.patience = patience
        self.val_interval = val_interval
        self.use_amp = use_amp
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.device = device
        self.dp_mechanism = dp_mechanism
        self.logger = Logger.get_instance()
        
        self.metrics = {
            'train/loss': MeanMetric(compute_on_step=False).to(device),
            'train/acc': MeanMetric(compute_on_step=False).to(device),
            'val/loss': MeanMetric(compute_on_step=False).to(device),
            'val/acc': MeanMetric(compute_on_step=False).to(device),
            'test/acc': MeanMetric(compute_on_step=False).to(device),
        }

    def reset(self):
        self.model = None
        self.best_metrics = None
        self.checkpoint_path = None

        for metric in self.metrics.values():
            metric.reset()

    def aggregate_metrics(self, stage='train'):
        metrics = {}

        for metric_name, metric_value in self.metrics.items():
            if metric_name.startswith(stage):
                value = metric_value.compute()
                metric_value.reset()
                if torch.is_tensor(value):
                    value = value.item()
                metrics[metric_name] = value

        return metrics

    def performs_better(self, metrics):
        if self.best_metrics is None:
            return True
        elif self.monitor_mode == 'max':
            return metrics[self.monitor] > self.best_metrics[self.monitor]
        elif self.monitor_mode == 'min':
            return metrics[self.monitor] < self.best_metrics[self.monitor]
        else:
            raise ValueError(f'Unknown metric mode: {self.monitor_mode}')

    def load_best_model(self):
        if self.checkpoint_path:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            return self.model
        else:
            raise Exception('No checkpoint found')

    def fit(self, model, optimizer, train_dataloader, val_dataloader=None, test_dataloader=None, checkpoint=False):
        self.reset()
        self.model = model.to(self.device)
        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if self.dp_mechanism:
            self.model, optimizer, train_dataloader = self.dp_mechanism(
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
                self.train_loop(train_dataloader, optimizer, scaler)
                metrics.update(self.aggregate_metrics(stage='train'))

                # validation loop
                if val_dataloader:
                    self.validation_loop(val_dataloader, 'val')
                    metrics.update(self.aggregate_metrics(stage='val'))

                # test loop
                if test_dataloader:
                    self.validation_loop(test_dataloader, 'test')
                    metrics.update(self.aggregate_metrics(stage='test'))
                    
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

    def train_loop(self, dataloader, optimizer, scaler):
        self.model.train()
        self.progress.update('train', visible=len(dataloader) > 1)

        for batch in dataloader:
            metrics = self.train_step(batch, optimizer, scaler)
            for item in metrics:
                self.metrics[item].update(metrics[item], weight=len(batch))
            self.progress.update('train', advance=1)

        self.progress.reset('train', visible=False)

    def validation_loop(self, dataloader, stage):
        self.model.eval()
        self.progress.update(stage, visible=len(dataloader) > 1)

        for batch in dataloader:
            metrics = self.validation_step(batch, stage)
            for item in metrics:
                self.metrics[item].update(metrics[item], weight=len(batch))
            self.progress.update(stage, advance=1)

        self.progress.reset(stage, visible=False)

    def train_step(self, batch, optimizer, scaler):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss, metrics = self.model.step(batch, stage='train')
        
        if loss is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return metrics

    @torch.no_grad()
    def validation_step(self, batch, stage='val'):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            _, metrics = self.model.step(batch, stage=stage)
        return metrics


class TrainerProgress(Progress):
    def __init__(self, num_epochs, num_train_steps, num_val_steps, num_test_steps):

        progress_bar = [
            SpinnerColumn(),
            "{task.description}",
            "{task.completed:>3}/{task.total}",
            "{task.fields[unit]}",
            BarColumn(),
            "{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            # "{task.fields[metrics]}"
        ]

        super().__init__(*progress_bar, console=console)

        self.trainer_tasks = {
            'epoch': self.add_task(total=num_epochs, metrics='', unit='epochs', description='overal progress'),
            'train': self.add_task(total=num_train_steps, metrics='', unit='steps', description='training', visible=False),
            'val':   self.add_task(total=num_val_steps, metrics='', unit='steps', description='validation', visible=False),
            'test':  self.add_task(total=num_test_steps, metrics='', unit='steps', description='testing', visible=False),
        }

        self.max_rows = 0

    def update(self, task, **kwargs):
        if 'metrics' in kwargs:
            kwargs['metrics'] = self.render_metrics(kwargs['metrics'])

        super().update(self.trainer_tasks[task], **kwargs)

    def reset(self, task, **kwargs):
        super().reset(self.trainer_tasks[task], **kwargs)

    def render_metrics(self, metrics):
        out = []
        for split in ['train', 'val', 'test']:
            metric_str = ' '.join(f'{k}: {v:.3f}' for k, v in metrics.items() if k.startswith(split))
            out.append(metric_str)
        
        return '  '.join(out)

    def make_tasks_table(self, tasks):
        """Get a table to render the Progress display.

        Args:
            tasks (Iterable[Task]): An iterable of Task instances, one per row of the table.

        Returns:
            Table: A table instance.
        """
        table_columns = (
            (
                Column(no_wrap=True)
                if isinstance(_column, str)
                else _column.get_table_column().copy()
            )
            for _column in self.columns
        )

        table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)

        if tasks:
            epoch_task = tasks[0]
            metrics = epoch_task.fields['metrics']

            for task in tasks:
                if task.visible:
                    table.add_row(
                        *(
                            (
                                column.format(task=task)
                                if isinstance(column, str)
                                else column(task)
                            )
                            for column in self.columns
                        )
                    )

            self.max_rows = max(self.max_rows, table.row_count)
            pad_top = 0 if epoch_task.finished else self.max_rows - table.row_count
            group = Group(table, Padding(Text(metrics), pad=(pad_top,0,0,2)))
            return Padding(group, pad=(0,0,1,18))

        else:
            return table