import logging
from functools import wraps

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
                    msg = (
                        f"Module {str(module)}: {str(e)}\n"
                        f"The error can be safely ignored if not requiring the {str(module)} module.\n"
                        f"Please consider submitting an issue on GitHub including the error message,\n"
                        f"the stack trace and your FDS input-file so we can reproduce and fix it."
                    )
                    if not settings.IGNORE_ERRORS:
                        logging.warning(msg, exc_info=True)

        return wrapped

    return decorated
