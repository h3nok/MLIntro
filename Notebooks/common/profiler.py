import time
from functools import wraps

from logs import logger as L

PROF_DATA = {}

_logger = L.Logger(module=__name__, console=True).configure()


class Timer:

    @staticmethod
    def function_timer(fn):
        @wraps(fn)
        def with_profiling(*args, **kwargs):
            start_time = time.time()

            ret = fn(*args, **kwargs)

            elapsed_time = time.time() - start_time

            if fn.__name__ not in PROF_DATA:
                PROF_DATA[fn.__name__] = [0, []]
            PROF_DATA[fn.__name__][0] += 1
            PROF_DATA[fn.__name__][1].append(elapsed_time)

            return ret

        return with_profiling

    @staticmethod
    def log():
        for fname, data in PROF_DATA.items():
            max_time = max(data[1])
            avg_time = sum(data[1]) / len(data[1])
            _logger.info("Function %s called %d times. " % (fname, data[0]))
            _logger.info('Execution time max: %.3f sec , average: %.3f sec ' % (max_time, avg_time))

    @staticmethod
    def cleanup():
        global PROF_DATA
        PROF_DATA = {}
        del PROF_DATA


