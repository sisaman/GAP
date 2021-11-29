from console import console
import torch
from torch.optim import SGD, Adam
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
                 optimizer:     dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
                 learning_rate: dict(help='learning rate', option='--lr') = 0.01,
                 weight_decay:  dict(help='weight decay (L2 penalty)') = 0.0,
                 epochs:        dict(help='number of training epochs') = 100,
                 patience:      dict(help='early-stopping patience window size') = 0,
                 val_interval:  dict(help='number of epochs to wait for validation', type=int) = 1,
                 use_amp:       dict(help='use automatic mixed precision training') = False,
                 device = 'cuda',
                 ):

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.patience = patience
        self.val_interval = val_interval
        self.use_amp = use_amp
        self.device = device
        
        self.logger = Logger.get_instance()
        self.reset()

    def reset(self):
        self.model = None
        self.best_metrics = None

    def configure_optimizer(self):
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

    def performs_better(self, metrics):
        if self.best_metrics is None:
            return True
        else:
            return metrics['val/acc'] > self.best_metrics['val/acc']

    def fit(self, model, data, description=''):
        data = data.to(self.device)
        self.model = model.to(self.device)
        optimizer = self.configure_optimizer()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        num_epochs_without_improvement = 0
        self.best_metrics = None

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
                train_metrics = self._train(data, optimizer, scaler, epoch)
                metrics.update(**train_metrics)
                val_metrics = self._validation(data)
                metrics.update(**val_metrics)

                if self.val_interval:
                    if epoch % self.val_interval == 0:

                        if self.performs_better(metrics):
                            self.best_metrics = metrics
                            num_epochs_without_improvement = 0
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

    def _train(self, data, optimizer, scaler, epoch):
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss, metrics = self.model.training_step(data, epoch)
        
        if loss is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return metrics

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model.validation_step(data)
        return out
