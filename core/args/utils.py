from typing import Annotated, Callable, Literal, Union, get_args, get_origin
from core import console
import inspect
from argparse import SUPPRESS, ArgumentParser, ArgumentTypeError, Namespace
from core.utils import dict2table


ArgType = Union[Namespace, dict[str, object]]
ArgInfo = dict[str, object]


class ArgWithLiteral:
    def __init__(self, main_type, literals):
        self.main_type = main_type
        self.literals = literals

    def __call__(self, arg):
        try:
            return self.main_type(arg)
        except ValueError:
            if arg in self.literals:
                return arg
            else:
                raise ArgumentTypeError(f'{arg} is not a valid literal')

def boolean(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def remove_prefix(kwargs: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in kwargs.items() if k.startswith(prefix)}


def strip_kwargs(callable: Callable, kwargs: dict, prefix: str='') -> dict[str, object]:
    signature = inspect.signature(callable)
    parameters = signature.parameters
    out_kwargs = {}
    for arg, value in kwargs.items():
        if arg.startswith(prefix) and arg[len(prefix):] in parameters:
            out_kwargs[arg] = value

    # check if the function has kwargs
    for _, param in parameters.items():
        annotation = param.annotation
        if get_origin(annotation) is Annotated:
            metadata = get_args(annotation)[1]
            bases = metadata.get('bases', [])
            prefixes = metadata.get('prefixes', [''] * len(bases))
            for base_callable, pr in zip(bases, prefixes):
                out_kwargs.update(strip_kwargs(base_callable, kwargs, prefix=prefix+pr))
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    return out_kwargs


def create_arguments(callable: Callable, parser: ArgumentParser, exclude: list = [], prefix: str = ''):
    arguments_added = [action.dest for action in parser._actions]
    parameters = inspect.signature(callable).parameters

    # iterate over the parameters
    for param_name, param_obj in parameters.items():
        arg_name = prefix + param_name

        # skip the parameters that are in the exclude list
        if param_name in exclude or arg_name in arguments_added:
            continue
        
        # get the annotation
        annot_obj = param_obj.annotation

        # only process annotated parameters
        if get_origin(annot_obj) is Annotated:

            # extract parameter type and metadata from annotation
            annotation = get_args(annot_obj)
            metadata: dict = annotation[1]
            param_type = metadata.get('type', annotation[0])

            # get the base callable arguments
            bases = metadata.get('bases', False)

            if bases:
                # if there are base callables, recursively add their args to the parser
                prefixes = metadata.get('prefixes', [''] * len(bases))
                for base_callable, pr in zip(bases, prefixes):
                    create_arguments(
                        callable=base_callable, 
                        parser=parser, 
                        exclude=metadata.get('exclude', []) + exclude,
                        prefix=prefix+pr
                    )
            else:
                # if there are no base callables, add the parameter to the parser
                metadata['type'] = param_type
                metadata['dest'] = arg_name

                # if the parameter has a default value, add it to the parser
                # otherwise, set the parameter as required
                if param_obj.default is not inspect.Parameter.empty:
                    metadata['default'] = param_obj.default
                else:
                    metadata['required'] = True
                    metadata['default'] = SUPPRESS
                    metadata['help'] += ' (required)'

                # tweak specific data types
                if param_type is bool:
                    # process boolean parameters
                    metadata['type'] = boolean
                    metadata['nargs'] = '?'
                    metadata['const'] = True
                elif get_origin(param_type) is Union:
                    # process union parameters
                    sub_types = get_args(param_type)
                    if len(sub_types) == 2 and get_origin(sub_types[0]) is Literal:
                        metadata['type'] = ArgWithLiteral(main_type=sub_types[1], literals=get_args(sub_types[0]))
                        metadata['metavar'] = f"<{sub_types[1].__name__}>" + '|{' +  ','.join(map(str, get_args(sub_types[0]))) + '}'

                # if metadata contains "choices", the parser uses that as meta variable
                # otherwise, if "metavar" is not provided, set the meta variable to  <parameter type>
                if 'choices' not in metadata:
                    try:
                        metadata['metavar'] = metadata.get('metavar', f'<{param_type.__name__}>')
                    except: pass

                # create options based on parameter name
                options = {f'--{arg_name}', f'--{arg_name.replace("_", "-")}'}

                # add custome options if provided
                custom_options = metadata.pop('option', [])
                custom_options = [custom_options] if isinstance(custom_options, str) else custom_options
                for option in custom_options:
                    idx = max([i for i,c in enumerate(option) if c == '-'])
                    option = option[:idx+1] + prefix + option[idx+1:]
                    options.add(option)

                # sort option names based on their length
                options = sorted(sorted(list(options)), key=len)

                # add the parameter to the parser
                parser.add_argument(*options, **metadata)


def print_args(args: ArgType, num_cols: int = 4):
    args = args if isinstance(args, dict) else vars(args)
    table = dict2table(args, num_cols=num_cols, title='program arguments')
    console.print()
    console.info(table)
    console.print()
