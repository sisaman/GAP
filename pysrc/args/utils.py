from pysrc.console import console
import math
import inspect
from rich.table import Table
from rich.highlighter import ReprHighlighter
from rich import box
from tabulate import tabulate
from argparse import ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def argsetup(Cls):

    def strip_unexpected_kwargs(func, kwargs):
        signature = inspect.signature(func)
        parameters = signature.parameters

        # check if the function has kwargs
        for name, param in parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return kwargs

        kwargs = {arg: value for arg, value in kwargs.items() if arg in parameters}
        return kwargs

    def from_args(args, **kwargs) -> Cls:
        args = strip_unexpected_kwargs(Cls.__init__, vars(args))
        args.update(kwargs)
        return Cls(**args)

    def add_parameters_as_argument(parser):
        function = Cls.__init__
        parameters = inspect.signature(function).parameters
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

    Cls.add_args = add_parameters_as_argument
    Cls.from_args = from_args

    return Cls


def print_args(args, num_cols=4):
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
