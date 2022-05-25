from typing import Annotated, Callable, Literal, Union, get_args, get_origin
from pysrc.console import console
import math
import inspect
from rich.table import Table
from rich.highlighter import ReprHighlighter
from rich import box
from tabulate import tabulate
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pysrc.utils import RT


ArgType = Union[Namespace, dict[str, object]]

class ArgWithLiteral:
    def __init__(self, main_type, literals):
        self.main_type = main_type
        self.literals = literals

    def __call__(self, arg):
        try:
            return self.main_type(arg)
        except ArgumentTypeError:
            if arg in self.literals:
                return arg
            else:
                raise ArgumentTypeError(f'{arg} is not a valid literal')

def str2bool(v: Union[str, bool]) -> bool:
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


def invoke_by_args(callable: Callable[..., RT], args: ArgType) -> RT:
    args_dict = args if isinstance(args, dict) else vars(args)
    args_dict = strip_unexpected_kwargs(callable, args_dict)
    return callable(**args_dict)


def create_arguments(callable: Callable, parser: ArgumentParser):
    parameters = inspect.signature(callable).parameters
    for param_name, param_obj in parameters.items():
        annotation = param_obj.annotation
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)
            param_type = annotation[0]
            metadata: dict = annotation[1]
            metadata['type'] = param_type
            metadata['default'] = param_obj.default
            metadata['dest'] = param_name

            if param_type is bool:
                metadata['type'] = str2bool
                metadata['nargs'] = '?'
                metadata['const'] = True
            elif get_origin(param_type) is Union:
                sub_types = get_args(param_type)
                if len(sub_types) == 2 and get_origin(sub_types[0]) is Literal:
                    metadata['type'] = ArgWithLiteral(main_type=sub_types[1], literals=get_args(sub_types[0]))
                    metadata['metavar'] = f"{sub_types[1].__name__}" + '|{' +  ','.join(map(str, get_args(sub_types[0]))) + '}'

            if 'choices' not in metadata:
                try:
                    metadata['metavar'] = metadata.get('metavar', param_type.__name__)
                except: pass
            
            #     choices = [str(c) for c in metadata['choices']]
            #     metadata['help'] = metadata.get('help', '') + f" (choices: {', '.join(choices)})"

            options = {f'--{param_name}', f'--{param_name.replace("_", "-")}'}
            custom_options = metadata.pop('option', [])
            custom_options = [custom_options] if isinstance(custom_options, str) else custom_options
            options.update(custom_options)
            options = sorted(sorted(list(options)), key=len)
            parser.add_argument(*options, **metadata)


def print_args(args: ArgType, num_cols: int = 4):
    args = args if isinstance(args, dict) else vars(args)
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
