import logging
import warnings
from rich.traceback import install
from rich.logging import RichHandler
from pysrc.console.console import Console


def setup() -> Console:
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
    return console


console = setup()