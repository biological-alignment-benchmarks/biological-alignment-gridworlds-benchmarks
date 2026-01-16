# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os
import sys
import time
import datetime
import warnings
import re

from progressbar import ProgressBar

if os.name == "nt":
    from semaphore_win_ctypes import (
        AcquireSemaphore,
        CreateSemaphore,
        OpenSemaphore,
        Semaphore,
    )
else:
    import posix_ipc

# this one is cross-platform
from filelock import FileLock


def disable_gym_warning():  # gym is actually not used by current code, the gym import probably comes from some dependency
    warnings.filterwarnings(
        "ignore", message=".*Gym has been unmaintained.*", category=UserWarning
    )


def wait_for_enter(message=None):
    if os.name == "nt":
        import msvcrt

        if message is not None:
            print(message)
        msvcrt.getch()  # Uses less CPU on Windows than input() function. This becomes perceptible when multiple console windows with Python are waiting for input. Note that the graph window will be frozen, but will still show graphs.
    else:
        if message is None:
            message = ""
        input(message)


# TODO: function to read CSV inside lock


def try_df_to_csv_write(df, filepath, **kwargs):
    while (
        True
    ):  # TODO: refactor this loop to a shared helper function. recording.py uses a same pattern
        try:
            with FileLock(
                str(filepath)
                + ".lock"  # filepath may be PosixPath, so need to convert to str
            ):  # NB! take the lock inside the loop, not outside, so that when we are waiting for user confirmation for retry, we do not block other processes during that wait
                df.to_csv(filepath, **kwargs)
            return
        except PermissionError:
            print(
                f"Cannot write to file {filepath} Is the file open by Excel or some other program?"
            )
            wait_for_enter("\nPress [enter] to retry.")


class RobustProgressBar(ProgressBar):
    def __init__(self, *args, initial_value=0, disable=False, granularity=1, **kwargs):
        self.disable = disable
        self.granularity = granularity
        self.initial_value = initial_value
        self.prev_value = initial_value
        super(RobustProgressBar, self).__init__(
            *args, initial_value=initial_value, **kwargs
        )

    def __enter__(self):
        if not self.disable:
            try:
                super(RobustProgressBar, self).__enter__()
            except Exception:  # TODO: catch only console write related exceptions
                pass
        return self

    def __exit__(self, type, value, traceback):
        if not self.disable:
            try:
                super(RobustProgressBar, self).__exit__(type, value, traceback)
            except Exception:  # TODO: catch only console write related exceptions
                pass
        return

    def update(self, value=None, *args, force=False, **kwargs):
        if not self.disable:
            try:
                if force or (
                    value is not None and value - self.prev_value >= self.granularity
                ):  # avoid too frequent console updates which would slow down the computation
                    if value is not None:
                        if self.prev_value == self.initial_value:
                            force = True  # without forcing, for some reason the progressbar shows "0" when the value is "1"
                        self.prev_value = value
                    super(RobustProgressBar, self).update(
                        value, *args, force=force, **kwargs
                    )
            except Exception:  # TODO: catch only console write related exceptions
                pass
        return

    # def _blackHoleMethod(*args, **kwargs):
    #    return

    # def __getattr__(self, attr):
    #    if not self.disable:
    #        return super(RobustProgressBar, self).__getattr__(attr)
    #    else:
    #        return self._blackHoleMethod


# / class RobustProgressBar(ProgressBar):


def get_now_str():
    now_str = datetime.datetime.strftime(datetime.datetime.now(), "%m.%d %H:%M:%S")
    return now_str


# / def get_now_str():


# https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
class Timer(object):
    def __init__(self, name=None, quiet=False):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        if not self.quiet and self.name:
            print(get_now_str() + " : " + self.name + "...")

        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        elapsed = time.time() - self.tstart

        if not self.quiet:
            if self.name:
                print(
                    get_now_str() + " : " + self.name + " totaltime: {}".format(elapsed)
                )
            else:
                print(get_now_str() + " : " + "totaltime: {}".format(elapsed))
        # / if not quiet:


# / class Timer(object):


# There does not seem to be a cross platform semaphore class available, so lets create one by combining platform specific semaphores.
# Note that there is a cross-platform lock class available in filelock package. This would be equivalent to special case of Semaphore with max_count=1.
class Semaphore(object):
    def __init__(self, name, max_count, *args, disable=False, **kwargs):
        self.name = name
        self.max_count = max_count
        self.disable = disable
        self.win_semaphore = None
        self.win_acquired_semaphore = None
        self.posix_semaphore = None
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        if not self.disable:
            if os.name == "nt":
                self.win_semaphore = CreateSemaphore(
                    self.name, *self.args, maximum_count=self.max_count, **self.kwargs
                )
                self.win_semaphore.__enter__()
                self.win_acquired_semaphore = AcquireSemaphore(self.win_semaphore)
                self.win_acquired_semaphore.__enter__()
            else:
                self.posix_semaphore = posix_ipc.Semaphore(
                    self.name,
                    *self.args,
                    flags=posix_ipc.O_CREAT,
                    initial_value=self.max_count,
                    **self.kwargs,
                )
                self.posix_semaphore.__enter__()

        return self

    def __exit__(self, type, value, traceback):
        if not self.disable:
            if os.name == "nt":
                self.win_acquired_semaphore.__exit__(type, value, traceback)
                self.win_acquired_semaphore = None
                self.win_semaphore.__exit__(type, value, traceback)
                self.win_semaphore = None
            else:
                self.posix_semaphore.__exit__(type, value, traceback)
                self.posix_semaphore = None
        return


# / class Semaphore(object):


def check_for_nan_errors(ex, cfg):
    if isinstance(ex, ValueError):
        msg = str(ex)
        if (
            cfg.hparams.model_params.soft_stop_training_on_nan_errors
            and "found invalid values" in msg  # f"but found invalid values:\n{value}"
        ):
            print("SB3 encountered NaNs")
            return True
        else:
            return False
    elif isinstance(ex, RuntimeError):
        msg = str(ex)
        if (
            cfg.hparams.model_params.soft_stop_training_on_nan_errors
            and cfg.hparams.model_params.early_detect_nans  # is torch.autograd.set_detect_anomaly(True) called?
            and " nan values" in msg  # "' returned nan values in its "
        ):
            print("Torch encountered NaNs")
            return True
        else:
            return False
    elif isinstance(ex, FloatingPointError):
        msg = str(ex)
        if (
            cfg.hparams.model_params.soft_stop_training_on_nan_errors
            and cfg.hparams.model_params.early_detect_nans  # is np.seterr(divide="raise", over="raise", invalid="raise", under="ignore") called?
            and "encountered in" in msg  # "%s encountered in %s"
        ):
            print("Numpy encountered NaNs")
            return True
        else:
            return False
    else:
        return False


# / def check_for_nan_errors(ex, cfg):


# adds timestamps to all print statements, even if they do not use logger object
class LogTimestamper(object):
    def __init__(self, terminal):
        self.terminal = terminal
        pid = os.getpid()
        self.pid_str = " : " + str(pid).rjust(7) + " :"
        self.first_message_printed = False

        self.newline_re = re.compile(
            r"(\r\n|\n\r|\n)"
        )  # NB! () is needed to capture the separators as individual groups
        self.timestamp_re = re.compile(
            r"[0-9]{4}\.[01]?[0-9]\.[0-3]?[0-9] [012]?[0-9]:[0-5]?[0-9]:[0-5]?[0-9]"
        )

    def get_now_str(self):
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y.%m.%d %H:%M:%S")

    def write(self, message_in):
        if isinstance(message_in, bytes):
            message_decoded = message_in.decode(
                "utf-8", "ignore"
            )  # NB! message_in might be byte array not a string
        else:
            message_decoded = message_in

        now = self.get_now_str()
        newline_prefix = (
            " " + now + self.pid_str
        )  # NB! add space in front of the timestamp to mitigate the first character appearing at the end of last progressbar update line when print calls are interleaved with progress bar updates

        lines = re.split(self.newline_re, message_decoded)

        messages_with_timestamp = []

        if not self.first_message_printed:
            self.first_message_printed = True
            messages_with_timestamp.append(newline_prefix)

        found_subprocess_timestamps = False
        for index, message in enumerate(lines):
            if message in ["\r\n", "\n\r", "\n"]:
                next_message = lines[index + 1] if index + 1 < len(lines) else ""
                # handle subprocess messages - a message from subprocess contains a timestamp already, so do not add it again. Note also, Logger messages begin with [timestamp] so there is no need to add one more timestamp, but since the logger message does not contain pid then lets still add timestamp-pid pair here
                if re.match(self.timestamp_re, next_message.strip()):
                    found_subprocess_timestamps = True
                elif (
                    not found_subprocess_timestamps
                ):  # NB! do not add current process PID to subprocess message even when subprocess message line has no timestamp (for example, progress bar lines are such)
                    message = (
                        message + newline_prefix
                    )  # NB! newlines come before the timestamp
            elif message != "":
                if re.match(self.timestamp_re, message.strip()):
                    found_subprocess_timestamps = True
                else:
                    message = (
                        " " + message
                    )  # NB! add space in front of each line to mitigate the first character appearing at the end of last progressbar update line

            messages_with_timestamp.append(message)

        # / for index, message in enumerate(lines):

        messages_with_timestamp = "".join(messages_with_timestamp)
        messages_with_timestamp = messages_with_timestamp.encode(
            "utf-8", "ignore"
        ).decode("utf-8", "ignore")

        try:
            self.terminal.write(messages_with_timestamp)
            qqq = True  # for debugging
        except Exception as ex:
            pass

    # / def write(self, message_in):

    def flush(self):
        self.terminal.flush()

    @property
    def encoding(self):
        return self.terminal.encoding

    def fileno(self):
        return self.terminal.fileno()


# / class LogTimestamper(object):


def init_console_timestamps():
    # NB! redirect stderr first, else stderr writes would get a duplicate timestamp
    if not isinstance(sys.stderr, LogTimestamper):
        sys.stderr = LogTimestamper(sys.stdout)
    if not isinstance(sys.stdout, LogTimestamper):
        sys.stdout = LogTimestamper(sys.stdout)
