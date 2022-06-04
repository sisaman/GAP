import logging
import warnings
from rich.traceback import install
from rich.logging import RichHandler
from pysrc.console.console import Console


def setup(log_level: str='INFO', ignore_warnings: bool=True) -> Console:
    console = Console(tab_size=4)
    error_console = Console(stderr=True, tab_size=4)
    install(console=error_console, width=error_console.width)
    if ignore_warnings:
        warnings.simplefilter('ignore')
    logging.basicConfig(
        level=log_level,
        format='%(message)s', 
        datefmt="[%X]", 
        handlers=[RichHandler(
            console=console, 
            omit_repeated_times=True,
        )]
    )
    return console


console = setup(log_level='DEBUG', ignore_warnings=False)
