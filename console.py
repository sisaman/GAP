from rich.console import Console
from rich.logging import RichHandler
from rich.padding import Padding
from rich.traceback import install
import warnings
import logging

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

def log(msg):
    logging.info('')
    console.print(Padding(msg, (0, 0, 0, 18)))

console.log = log
