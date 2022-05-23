from typing import Callable, TypeVar
from pysrc.console import console
import math
import inspect
from rich.table import Table
from rich.highlighter import ReprHighlighter
from rich import box
from tabulate import tabulate
from argparse import ArgumentParser, ArgumentTypeError, Namespace

RT = TypeVar('RT')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def strip_unexpected_kwargs(callable: Callable, kwargs: dict) -> dict[str, object]:
    signature = inspect.signature(callable)
    parameters = signature.parameters

    # check if the function has kwargs
    for _, param in parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    kwargs = {arg: value for arg, value in kwargs.items() if arg in parameters}
    return kwargs


def init_from_args(callable: Callable[..., RT], args: Namespace, **kwargs) -> RT:
    args_dict = strip_unexpected_kwargs(callable, vars(args))
    args_dict.update(kwargs)
    return callable(**args_dict)


def create_arguments(callable: Callable, parser: ArgumentParser):
    parameters = inspect.signature(callable).parameters
    for param_name, param_obj in parameters.items():
        if param_obj.annotation is not inspect.Parameter.empty:
            arg_info = param_obj.annotation
            arg_info['default'] = param_obj.default
            arg_info['dest'] = param_name
            arg_info['type'] = arg_info.get('type', type(param_obj.default))

            if arg_info['type'] is bool:
                arg_info['type'] = str2bool
                arg_info['nargs'] = '?'
                arg_info['const'] = True

            if 'choices' in arg_info:
                choices = [str(c) for c in arg_info['choices']]
                arg_info['help'] = arg_info.get('help', '') + f" (choices: {', '.join(choices)})"
                arg_info['metavar'] = param_name.upper()

            options = {f'--{param_name}', f'--{param_name.replace("_", "-")}'}
            custom_options = arg_info.pop('option', [])
            custom_options = [custom_options] if isinstance(custom_options, str) else custom_options
            options.update(custom_options)
            options = sorted(sorted(list(options)), key=len)
            parser.add_argument(*options, **arg_info)


def print_args(args: Namespace, num_cols: int = 4):
    args = vars(args)
    num_args = len(args)
    num_rows = math.ceil(num_args / num_cols)
    col = 0
    data = {}
    keys = []
    vals = []

    for i, (key, val) in enumerate(args.items()):
        keys.append(f'{key}:')
        
        vals.append(val)
        if (i + 1) % num_rows == 0:
            data[col] = keys
            data[col+1] = vals
            keys = []
            vals = []
            col += 2

    data[col] = keys
    data[col+1] = vals

    highlighter = ReprHighlighter()
    message = tabulate(data, tablefmt='plain')
    table = Table(title='program arguments', show_header=False, box=box.HORIZONTALS)
    table.add_row(highlighter(message))

    console.print()
    console.log(table)
    console.print()
