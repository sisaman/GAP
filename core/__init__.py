import logging
import warnings
from rich.traceback import install
from rich.logging import RichHandler
from core.console.console import Console

globals = {
    'debug': False,
}

# define main and error consoles
console = Console(tab_size=4)
error_console = Console(stderr=True, tab_size=4)

# setup console for tracebacks
install(console=error_console, width=error_console.width)

# create logger
log_handler = RichHandler(
    console=console, 
    omit_repeated_times=True,
    log_time_format="[%X]"
)
logger = logging.getLogger('gap')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# setup warnings
logging.getLogger("py.warnings").addHandler(log_handler)
logging.getLogger("py.warnings").propagate = False
logging.captureWarnings(True)
warnings.simplefilter("ignore")
