from typing import Annotated
from pysrc.loggers.base import LoggerBase
from pysrc.loggers.csv import CSVLogger
from pysrc.loggers.wandb import WandbLogger


class Logger:

    _instance: LoggerBase = None
    _options = {
        'logger': 'csv',
        'project': 'GAP-DEBUG',
        'output_dir': './output',
        'debug': False,
        'enabled': True,
        'config': {}
    }

    @classmethod
    def get_instance(cls) -> LoggerBase:
        return cls._instance

    def __new__(cls) -> LoggerBase:
        if cls._instance is None:
            raise RuntimeError('Logger is not initialized, call Logger.setup() first')
        return cls._instance

    @classmethod
    def setup(cls,
        logger:        Annotated[str,  dict(help='select logger type', choices=['wandb', 'csv'])] = 'csv',
        project:       Annotated[str,  dict(help="project name for logger")] = 'GAP-DEBUG',
        output_dir:    Annotated[str,  dict(help="directory to store the results", option='-o')] = './output',
        debug:         Annotated[bool, dict(help='enable debugger logging')] = False,
        enabled:       bool = True,
        config:        dict = {}
        ) -> LoggerBase:

        cls._options['logger'] = logger
        cls._options['project'] = project
        cls._options['output_dir'] = output_dir
        cls._options['debug'] = debug
        cls._options['enabled'] = enabled
        cls._options['config'] = config

        LoggerCls = WandbLogger if debug or logger == 'wandb' else CSVLogger
        cls._instance = LoggerCls(project=project, output_dir=output_dir, enabled=enabled or debug, config=config)
        return cls._instance
