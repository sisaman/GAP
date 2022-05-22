import warnings
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from pysrc.console.logging import log
from pysrc.console.logging import LogStatus


def setup():
    console = Console(tab_size=4)
    error_console = Console(stderr=True, tab_size=4)
    install(console=error_console, width=error_console.width)
    warnings.filterwarnings('ignore')
    logging.basicConfig(
        level='INFO', 
        format='%(message)s', 
        datefmt="[%X]", 
        handlers=[RichHandler(console=console, omit_repeated_times=True)]
    )

    console.log = lambda *args, **kwargs: log(*args, console=console, **kwargs)
    with console.status(''): pass
    console.status = lambda status, level=logging.INFO: LogStatus(status, console, level)
    return console


console = setup()
