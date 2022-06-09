import logging
from rich.traceback import install
from rich.logging import RichHandler
from core.console.console import *

log_level = DEBUG

# define main and error consoles
console = Console(tab_size=4, log_level=log_level)
error_console = Console(stderr=True, tab_size=4, log_level=log_level)

# setup console for tracebacks
install(console=error_console, width=error_console.width)

# create logger
log_handler = RichHandler(
    console=console, 
    omit_repeated_times=True,
    log_time_format="[%X]"
)
logger = logging.getLogger('gap')
logger.setLevel(log_level)
logger.addHandler(log_handler)

# setup warnings
logging.getLogger("py.warnings").addHandler(log_handler)
logging.getLogger("py.warnings").propagate = False
logging.captureWarnings(True)
