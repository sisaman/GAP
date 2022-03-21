import os
import rich
from rich.console import Console, NewLine
from rich.logging import RichHandler
from rich.scope import render_scope
from rich.segment import Segment
from rich.spinner import Spinner
from rich.styled import Styled
from rich.table import Table
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


def log(
    *objects,
    level = logging.INFO,
    sep: str = " ",
    end: str = "\n",
    style = None,
    justify = None,
    emoji = None,
    markup = None,
    highlight = None,
    log_locals: bool = False,
    _stack_offset: int = 2,
) -> None:
    """Log rich content to the terminal.
    Args:
        objects (positional args): Objects to log to the terminal.
        sep (str, optional): String to write between print data. Defaults to " ".
        end (str, optional): String to write at end of print data. Defaults to "\\\\n".
        style (Union[str, Style], optional): A style to apply to output. Defaults to None.
        justify (str, optional): One of "left", "right", "center", or "full". Defaults to ``None``.
        overflow (str, optional): Overflow method: "crop", "fold", or "ellipsis". Defaults to None.
        emoji (Optional[bool], optional): Enable emoji code, or ``None`` to use console default. Defaults to None.
        markup (Optional[bool], optional): Enable markup, or ``None`` to use console default. Defaults to None.
        highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use console default. Defaults to None.
        log_locals (bool, optional): Boolean to enable logging of locals where ``log()``
            was called. Defaults to False.
        _stack_offset (int, optional): Offset of caller from end of call stack. Defaults to 1.
    """
    if not objects:
        objects = (NewLine(),)

    render_hooks = console._render_hooks[:]

    with console:
        renderables = console._collect_renderables(
            objects, sep, end, justify=justify, emoji=emoji, markup=markup, highlight=highlight
        )
        if style is not None:
            renderables = [Styled(renderable, style) for renderable in renderables]

        filename, line_no, locals = console._caller_frame_info(_stack_offset)
        link_path = None if filename.startswith("<") else os.path.abspath(filename)
        path = filename.rpartition(os.sep)[-1]
        if log_locals:
            locals_map = {key: value for key, value in locals.items() if not key.startswith("__")}
            renderables.append(render_scope(locals_map, title="[i]locals"))

        record = logging.LogRecord(name=None, level=level, pathname=None, lineno=None, msg=None, args=None, exc_info=None)
        handler = RichHandler(console=console)
        console._log_render.show_level = True

        renderables = [
            console._log_render(
                console,
                renderables,
                level=handler.get_level_text(record),
                log_time=console.get_datetime(),
                path=path,
                line_no=line_no,
                link_path=link_path,
            )
        ]
        for hook in render_hooks:
            renderables = hook.process_renderables(renderables)
        new_segments = []
        extend = new_segments.extend
        render = console.render
        render_options = console.options
        for renderable in renderables:
            extend(render(renderable, render_options))
        buffer_extend = console._buffer.extend
        for line in Segment.split_and_crop_lines(new_segments, console.width, pad=False):
            buffer_extend(line)

with console.status(''): pass
class LogStatus(rich.status.Status):
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
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.console.log(self.status+'...done', level=self.level)


console.log = log
console.status = lambda status, level=logging.INFO: LogStatus(status, console, level)