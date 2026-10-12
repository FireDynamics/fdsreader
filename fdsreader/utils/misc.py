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
                    if args and hasattr(args[0], "load_errors"):
                        # Drop the traceback before storing: it otherwise keeps the failed
                        # loader's whole frame (locals, self, open file handles) reachable via
                        # load_errors for as long as the Simulation object lives.
                        args[0].load_errors.append((str(module), e.with_traceback(None)))

        return wrapped

    return decorated
