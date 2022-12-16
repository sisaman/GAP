from typing import Iterable
from rich.console import Group
from rich.padding import Padding
from rich.text import Text
from rich.table import Column, Table
from core.modules.base import Metrics
from core import console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, Task
from rich.highlighter import ReprHighlighter


class TrainerProgress(Progress):
    def __init__(self, 
                 num_epochs: int, 
                 num_train_steps: int, 
                 num_val_steps: int, 
                 num_test_steps: int,
                 ):

        progress_bar = [
            SpinnerColumn(),
            "{task.description}",
            "[cyan]{task.completed:>3}[/cyan]/[cyan]{task.total}[/cyan]",
            "{task.fields[unit]}",
            BarColumn(),
            "[cyan]{task.percentage:>3.0f}[/cyan]%",
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

    def update(self, task: Task, **kwargs):
        if 'metrics' in kwargs:
            kwargs['metrics'] = self.render_metrics(kwargs['metrics'])

        super().update(self.trainer_tasks[task], **kwargs)

    def reset(self, task: Task, **kwargs):
        super().reset(self.trainer_tasks[task], **kwargs)

    def render_metrics(self, metrics: Metrics) -> str:
        out = []
        for split in ['train', 'val', 'test']:
            metric_str = ' '.join(f'{k}: {v:.3f}' for k, v in metrics.items() if f'{split}/' in k)
            out.append(metric_str)
        
        return '  '.join(out)

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
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

        highlighter = ReprHighlighter()
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
            group = Group(table, Padding(highlighter(metrics), pad=(pad_top,0,0,2)))
            return Padding(group, pad=(0,0,1,21))

        else:
            return table