import logging

log = logging.getLogger(__name__)


class LoggerWriter:
    """
    By xjcl From https://stackoverflow.com/a/66209331
    """

    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith("\n"):
            self.buf.append(msg.removesuffix("\n"))
            self.logfct("".join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass
