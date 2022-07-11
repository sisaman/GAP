import core.globals
from typing import Annotated
from core.console import console
from core.loggers.base import LoggerBase
from core.loggers.csv import CSVLogger
from core.loggers.wandb import WandbLogger


class Logger:
    supported_loggers = {
        'csv': CSVLogger,
        'wandb': WandbLogger,
    }

    _instance: LoggerBase = None
    
    @classmethod
    def get_instance(cls) -> LoggerBase:
        return cls._instance

    def __new__(cls) -> LoggerBase:
        if cls._instance is None:
            raise RuntimeError('Logger is not initialized, call Logger.setup() first')
        return cls._instance

    @classmethod
    def setup(cls,
        logger:   Annotated[str,  dict(help='select logger type', choices=supported_loggers)] = 'csv',
        **kwargs: Annotated[dict, dict(help='additional kwargs for the underlying logger', bases=[LoggerBase])],
        ) -> LoggerBase:
        
        if core.globals.DEBUG_MODE:
            logger = 'wandb'
            kwargs['enabled'] = True
            kwargs['project'] += '-DEBUG'
            console.debug(f'debug mode: wandb logger is enabled for project {kwargs["project"]}')
        
        LoggerCls = cls.supported_loggers[logger]
        cls._instance = LoggerCls(**kwargs)
        return cls._instance
