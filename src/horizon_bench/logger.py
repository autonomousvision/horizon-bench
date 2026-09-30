import logging
import traceback
import sys
import random
import os
from horizon_bench.Config import LOGS_DIR


def initialize_logger(model_name, input_filename):
    logger = logging.getLogger()
    log_path = os.path.join(LOGS_DIR, model_name, input_filename.replace('.txt', '.json'))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, delay=True)
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)

    def write_log(exception, exc_traceback=None):
        tb = "".join(traceback.format_tb(exc_traceback)) if exc_traceback else traceback.format_exc()
        log_entry = {
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "model_name": model_name,
            "input_filename": input_filename,
            "stack_trace": tb,
            "timestamp": logging.Formatter().formatTime(logging.LogRecord(name="", level=logging.ERROR, pathname="", lineno=0, msg="", args=(), exc_info=None))
        }
        logger.error(log_entry)

    def exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        write_log(exc_value, exc_traceback)

    sys.excepthook = exception_hook

    logger.write_log = write_log
    return logger


def risky_operation_example():
    # Simulate an operation that may raise an exception
    if random.random() < 0.5:
        raise ValueError("An example error occurred.")
    else: 
        raise RuntimeError("Another example error occurred.")




if __name__ == "__main__":
    # usage example - no try/except needed, exceptions are logged automatically
    logger = initialize_logger("ExampleModel", '1.txt')
    risky_operation_example()