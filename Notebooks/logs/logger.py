import logging
from datetime import datetime
import asyncio
import os

LOG_DIR = r"C:\BII\viAi"


class Logger:
    """ Construct a Logger Instance
        TODO - A singleton pattern
    """

    _instance = None

    # def __init__(self, module, console=True):
    #     """
    #     Initialize a logger with the module __name__, __name__, write to
    #     a single log file
    #     @param module:
    #     @param console:
    #     """
    #
    #     self._module = module
    #     self._wrt_to_console = console

    def __new__(cls, module, console=True):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)

            cls._module = module
            cls._wrt_to_console = console

        return cls._instance

    def configure(self):
        """Configure the logger

        @return:
        """
        logger = logging.getLogger(self._module)
        logger.setLevel(logging.DEBUG)

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        log_file = 'viAi_{}.log'.format(datetime.now().strftime("%m-%d-%Y"))
        log = os.path.join(LOG_DIR, log_file)
        fh = logging.FileHandler(log)
        fh.setLevel(logging.DEBUG)

        # create console handler
        ch = None

        if self._wrt_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        if self._wrt_to_console:
            ch.setFormatter(formatter)

        # we don't want to add two of the same handlers on the
        # logger if some creates two loggers with the same name
        has_fh = False
        has_ch = False
        for hdl in logger.handlers:
            if isinstance(hdl, logging.FileHandler):
                has_fh = True
            if isinstance(hdl, logging.StreamHandler):
                has_ch = True

        # _add the handlers to the logger
        if not has_fh:
            logger.addHandler(fh)
        if self._wrt_to_console and not has_ch:
            logger.addHandler(ch)

        return logger

