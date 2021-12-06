from console import console
import os, uuid
import torch
from args import support_args
from loggers import Logger
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn
    

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

    def fit(self, model, train_dataloader, val_dataloader, test_dataloader=[], checkpoint=False):
        # data = data.to(self.device)
        self.model = model.to(self.device)
        optimizer = self.model.configure_optimizers()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            self.checkpoint_path = os.path.join('checkpoints', f'{uuid.uuid1()}.pt')

        num_epochs_without_improvement = 0
        num_train_steps = len(train_dataloader)
        num_val_steps = len(val_dataloader)
        num_test_steps = len(test_dataloader)

        progress = TrainerProgress(
            num_epochs=self.epochs, 
            num_train_steps=num_train_steps, 
            num_val_steps=num_val_steps, 
            num_test_steps=num_test_steps
        )
        
        progress.start()
        
        for epoch in range(1, self.epochs + 1):
            metrics = {'epoch': epoch}

            progress.update('train', visible=num_train_steps > 1)
            for batch in train_dataloader:
                batch = batch.to(self.device)
                self._train(batch, optimizer, scaler)
                progress.update('train', advance=1)

            progress.reset('train', visible=False)
            metrics.update(**self.model.aggregate_metrics(stage='train'))
                
            if self.val_interval:
                if epoch % self.val_interval == 0:

                    progress.update('val', visible=num_val_steps > 1)
                    for batch in val_dataloader:
                        batch = batch.to(self.device)
                        self._validate(batch, stage='val')
                        progress.update('val', advance=1)
                    
                    progress.reset('val', visible=False)
                    metrics.update(**self.model.aggregate_metrics(stage='val'))

                    if num_test_steps > 0:

                        progress.update('test', visible=num_test_steps > 1)
                        for batch in test_dataloader:
                            batch = batch.to(self.device)
                            self._validate(batch, stage='test')
                            progress.update('test', advance=1)

                        progress.reset('test', visible=False)
                        metrics.update(**self.model.aggregate_metrics(stage='test'))

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

            if self.logger: self.logger.log(metrics)
            progress.update('epoch', metrics=metrics, advance=1)

        progress.stop()
        
        if self.logger: self.logger.log_summary(self.best_metrics)
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


class TrainerProgress(Progress):
    def __init__(self, num_epochs, num_train_steps, num_val_steps, num_test_steps):

        progress_bar = [
            TextColumn('                 '),
            SpinnerColumn(),
            "{task.description}",
            "{task.completed:>3}/{task.total}",
            "{task.fields[unit]}",
            BarColumn(),
            "{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "{task.fields[metrics]}"
        ]

        super().__init__(*progress_bar, console=console)

        self.trainer_tasks = {
            'epoch': self.add_task(total=num_epochs, metrics='', unit='epochs', description='overal progress'),
            'train': self.add_task(total=num_train_steps, metrics='', unit='steps', description='training       ', visible=False),
            'val':   self.add_task(total=num_val_steps, metrics='', unit='steps', description='validation     ', visible=False),
            'test':  self.add_task(total=num_test_steps, metrics='', unit='steps', description='testing        ', visible=False),
        }

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