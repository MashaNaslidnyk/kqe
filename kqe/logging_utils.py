import logging

logging_levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET,
}


def init_logging(level="info", filename=None):
    # if someone tried to log something before basicConfig is called, Python creates a default handler that
    # goes to the console and will ignore further basicConfig calls. Remove the handler if there is one.
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    head = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging_levels[level], format=head, filename=filename)
    logging.captureWarnings(True)


info = logging.info
warn = logging.warning
error = logging.error
debug = logging.debug
warning = logging.warning
