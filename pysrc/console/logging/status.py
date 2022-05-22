import rich
from rich.logging import RichHandler
from rich.spinner import Spinner
from rich.table import Table
from rich.status import Status
from time import time
import logging


# with console.status(''): pass
class LogStatus(Status):
    def __init__(self,
        status,
        console,
        level=logging.INFO,
        speed: float = 1.0,
        refresh_per_second: float = 12.5,
    ):
        super().__init__(status, 
            console=console, 
            spinner='simpleDots', 
            speed=speed, 
            refresh_per_second=refresh_per_second
        )
        
        self.status = status
        self.level = level
        spinner = Spinner('simpleDots', style='status.spinner', speed=speed)
        record = logging.LogRecord(name=None, level=level, pathname=None, lineno=None, msg=None, args=None, exc_info=None)
        handler = RichHandler(console=console)
        table = Table.grid()
        table.add_row(self.status, spinner)
        
        self._spinner = rich.logging.LogRender(show_level=True, time_format='[%X]')(
            console=console, 
            level=handler.get_level_text(record),
            renderables=[table]
        )
        self._live = rich.live.Live(
            self.renderable,
            console=console,
            refresh_per_second=refresh_per_second,
            transient=True,
        )
        
    def __enter__(self):
        self._start_time = time()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self._end_time = time()
        self.console.log(f'{self.status}...done in {self._end_time - self._start_time:.2f} s', level=self.level)
