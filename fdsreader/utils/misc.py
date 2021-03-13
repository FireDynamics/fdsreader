import logging
from functools import wraps
import sys
from fdsreader import settings


def log_error(module):
    def decorated(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if settings.DEBUG:
                    raise e
                else:
                    e = type(e)(f"Module {str(module)}: {str(e)}\nThe error can be safely ignored if not requiring the {str(module)} module. However, please consider to submit an issue on Github including the error message, the stack trace and your FDS input-file so we can reproduce the error and fix it as soon as possible!").with_traceback(sys.exc_info()[2])
                    logging.exception(e)
        return wrapped
    return decorated
