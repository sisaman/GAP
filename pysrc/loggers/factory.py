from argparse import Namespace
from pysrc.args.utils import argsetup
from pysrc.loggers.csv import CSVLogger
from pysrc.loggers.wandb import WandbLogger


@argsetup
class Logger:
    instance = None

    @classmethod
    def get_instance(cls):
        return cls.instance

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.__init__(cls, *args, **kwargs)
        return cls.instance

    def __init__(self,
                 logger:        dict(help='select logger type', choices=['wandb', 'csv']) = 'csv',
                 project:       dict(help="project name for logger") = 'GAP-DEBUG',
                 output_dir:    dict(help="directory to store the results", option='-o') = './output',
                 debug:         dict(help='enable debugger logging') = False,
                 enabled=True,
                 config=Namespace()):
        LoggerCls = WandbLogger if debug or logger == 'wandb' else CSVLogger
        Logger.instance = LoggerCls(project=project, output_dir=output_dir, enabled=enabled, config=config)
