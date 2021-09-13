import sys
import torch
from torch.optim import SGD, Adam
from tqdm.auto import tqdm
from args import support_args
from loggers import Logger


@support_args
class Trainer:
    def __init__(self,
                 privacy_accountant,
                 optimizer: dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
                 learning_rate: dict(help='learning rate') = 0.01,
                 weight_decay: dict(help='weight decay (L2 penalty)') = 0.0,
                 max_epochs: dict(help='maximum number of training epochs') = 200,
                 patience: dict(help='early-stopping patience window size') = 0,
                 val_interval: dict(help='number of epochs to wait for validation') = 1,
                 ):

        self.privacy_accountant = privacy_accountant
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_interval = val_interval
        self.logger = Logger.get_instance()
        self.model = None
        self.best_metrics = None

    def configure_optimizer(self):
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def performs_better(self, metrics):
        if self.best_metrics is None:
            return True
        else:
            return metrics['val/loss'] < self.best_metrics['val/loss']

    def fit(self, model, data):
        self.model = model.to(data.x.device)
        optimizer = self.configure_optimizer()

        num_epochs_without_improvement = 0
        self.best_metrics = None

        epoch_progbar = tqdm(range(self.max_epochs), desc='Epoch: ', file=sys.stdout)
        for epoch in epoch_progbar:
            metrics = {'epoch': epoch}
            train_metrics = self._train(data, optimizer)
            metrics.update(**train_metrics)

            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self._validation(data)
                metrics.update(**val_metrics)

                if self.performs_better(metrics):
                    self.best_metrics = metrics
                    num_epochs_without_improvement = 0
                else:
                    num_epochs_without_improvement += 1
                    if num_epochs_without_improvement >= self.patience > 0:
                        break

            metrics['eps'] = self.privacy_accountant()
            self.logger.log(metrics)

            # display metrics on progress bar
            epoch_progbar.set_postfix({
                key: val for key, val in metrics.items() #if key.split('/')[0] in ['train', 'val', 'test']
            })

        self.logger.log_summary(self.best_metrics)
        return self.best_metrics

    def _train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        loss, metrics = self.model.training_step(data)
        if loss is not None:
            loss.backward()
            optimizer.step()
        return metrics

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        return self.model.validation_step(data)
