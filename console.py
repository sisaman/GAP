from rich.console import Console
from rich.logging import RichHandler
from rich.padding import Padding
import warnings
import logging

console = Console(tab_size=4)
warnings.filterwarnings('ignore')
logging.basicConfig(
    level='INFO', 
    format='%(message)s', 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True, console=console, omit_repeated_times=True)]
)

def log(msg):
    logging.info('')
    console.print(Padding(msg, (0, 0, 0, 18)))

console.log = log
