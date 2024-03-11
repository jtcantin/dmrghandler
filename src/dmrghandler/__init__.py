# read version from installed package
import logging
from importlib.metadata import version

__version__ = version("dmrghandler")

logging.getLogger(__name__).addHandler(logging.NullHandler())
