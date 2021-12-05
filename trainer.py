from console import console
import os, uuid
import torch
from args import support_args
from loggers import Logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn


def render_metrics(metrics):
    out = []
    for split in ['train', 'val', 'test']:
        metric_str = ' '.join(f'{k}: {v:.3f}' for k, v in metrics.items() if k.startswith(split))
        out.append(metric_str)
    
    return '  '.join(out)
    

@support_args
class Trainer:
    def __init__(self,
                 epochs:        dict(help='number of training epochs') = 100,
                 patience:      dict(help='early-stopping patience window size') = 0,
                 val_interval:  dict(help='number of epochs to wait for validation', type=int) = 1,
                 use_amp:       dict(help='use automatic mixed precision training') = False,
                 monitor:       dict(help='metric to monitor') = 'val/acc',
                 monitor_mode:   dict(help='monitor mode', choices=['min', 'max']) = 'max',
                 device = 'cuda',
                 ):

        self.epochs = epochs
        self.patience = patience
        self.val_interval = val_interval
        self.use_amp = use_amp
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.device = device
        self.logger = Logger.get_instance()
        self.reset()

    def reset(self):
        self.model = None
        self.best_metrics = None
        self.checkpoint_path = None

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

    def fit(self, model, train_dataloader, val_dataloader, test_dataloader=None, checkpoint=False, description=''):
        # data = data.to(self.device)
        self.model = model.to(self.device)
        optimizer = self.model.configure_optimizers()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            self.checkpoint_path = os.path.join('checkpoints', f'{uuid.uuid1()}.pt')

        num_epochs_without_improvement = 0

        progress_bar = [
            TextColumn('                 '),
            SpinnerColumn(),
            "{task.description}",
            "{task.completed:>3}/{task.total}",
            "{task.fields[unit]}",
            BarColumn(),
            "{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "{task.fields[metrics]}",
        ]

        ### TODO: move progress bar to main.py ###

        with Progress(*progress_bar, console=console) as progress:
            task_training = progress.add_task(description, total=self.epochs, metrics='', unit='epochs')

            for epoch in range(1, self.epochs + 1):
                metrics = {'epoch': epoch}

                for batch in train_dataloader:
                    batch = batch.to(self.device)
                    self._train(batch, optimizer, scaler)

                metrics.update(**self.model.get_metrics(stage='train'))
                    
                if self.val_interval:
                    if epoch % self.val_interval == 0:

                        for batch in val_dataloader:
                            batch = batch.to(self.device)
                            self._validate(batch, stage='val')
                        
                        metrics.update(**self.model.get_metrics(stage='val'))

                        if test_dataloader is not None:
                            for batch in test_dataloader:
                                batch = batch.to(self.device)
                                self._validate(batch, stage='test')

                            metrics.update(**self.model.get_metrics(stage='test'))

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

                self.logger.log(metrics)
                progress.update(task_training, metrics=render_metrics(metrics), advance=1)
        
        self.logger.log_summary(self.best_metrics)
        return self.best_metrics

    def _train(self, batch, optimizer, scaler):
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss = self.model.step(batch, stage='train')
        
        if loss is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    @torch.no_grad()
    def _validate(self, batch, stage='val'):
        self.model.eval()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self.model.step(batch, stage=stage)