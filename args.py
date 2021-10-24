import enum
import inspect
from argparse import ArgumentTypeError, Action
from utils import colored_text, TermColors


class Enum(enum.Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class EnumAction(Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        _enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if _enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(_enum, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in _enum))

        super().__init__(**kwargs)

        self._enum = _enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        enum = self._enum(values)  # noqa
        setattr(namespace, self.dest, enum)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def support_args(Cls):

    def strip_unexpected_kwargs(func, kwargs):
        signature = inspect.signature(func)
        parameters = signature.parameters

        # check if the function has kwargs
        for name, param in parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return kwargs

        kwargs = {arg: value for arg, value in kwargs.items() if arg in parameters}
        return kwargs

    def from_args(args, **kwargs):
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
                elif isinstance(arg_info['type'], Enum):
                    arg_info['action'] = EnumAction

                if 'choices' in arg_info:
                    arg_info['help'] = arg_info.get('help', '') + f" (choices: {', '.join(arg_info['choices'])})"
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


def print_args(args):
    message = [f'{name}: {colored_text(str(value), TermColors.FG.cyan)}' for name, value in vars(args).items()]
    print(', '.join(message) + '\n')
