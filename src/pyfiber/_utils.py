import numpy as np
from scipy import signal
import yaml
import datetime
import inspect
import os
import shutil
from typing import List, Tuple

__all__ = ["PyFiber", "Intervals", "Events"]


Intervals = List[Tuple[float, float]]
Events = np.ndarray


class PyFiber:
    """Parent object for Behavioral, Fiber and Analysis objects.

    :cvar CFG: dictionnary containing the whole ``'pyfiber.yaml'`` file content
    :param verbose: if ``False``, activates silent mode (the log is still available in ``self.log``)
    :param kwargs: add additional attribute or modify config option at runtine"""

    FOLDER = os.path.expanduser("~/.pyfiber")
    _config_path = os.path.expanduser("~/.pyfiber/pyfiber.yaml")
    _sample_config = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())), "pyfiber.yaml"
    )
    if not os.path.exists(_config_path):
        try:
            os.mkdir(FOLDER)
        except FileExistsError:
            pass
        shutil.copyfile(_sample_config, _config_path)
    with open(_config_path, "r") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)  #: :meta hide-value:
        print(f"Configuration file at: {_config_path}")
    vars().update(CFG)
    vars().update(CFG["GENERAL"])
    vars().update(CFG["SYSTEM"])

    def __init__(self, **kwargs):
        if "verbosity" in kwargs.keys():
            self._verbosity = kwargs["verbosity"]
        else:
            self._verbosity = True
        self._log = []
        self.__dict__.update(**kwargs)

    @property
    def log(self):
        """Show the history of most of the operation (extraction, normalization, errors, ...) for all
        classes derived of ``_utils.PyFiber``.
        Users can optionnaly add information to the log, by assigning the new information:

        .. code-block:: ipython

            In [10]: obj.log
            '15:29:36 --- did something'
            '15:29:37 --- did something else'
            In [11]: obj.log = "new entry"
            In [12]: obj.log
            '15:29:36 --- did something'
            '15:29:37 --- did something else'
            '15:35:10 --- new entry'
        """
        print("\n".join([" ".join(i.split("\n")) for i in self._log]) + "\n")

    @log.setter
    def log(self, value):
        log = f"{datetime.datetime.now().strftime('%H:%M:%S')} --- {value}"
        self._log.append(log)

    @property
    def help(self):
        function_doc = [
            (k, v.__doc__)
            for k, v in self.__class__.__dict__.items()
            if (callable(v)) & (k[0] != "_")
        ]
        for a, b in function_doc:
            print(f"<obj>.\033[1m{a}\033[0m()\n {b}\n")

    @property
    def _help(self):
        function_doc = [
            (k, v) for k, v in self.__class__.__dict__.items() if (callable(v))
        ]
        for a, b in function_doc:
            print(f"{a:<20} --> {b.__doc__}")
            print(f"{inspect.getfullargspec(b)}")

    @property
    def info(self):
        print(
            "\n".join(
                [
                    "<obj>." + f"\033[1m{i}\033[0m"
                    for i in sorted(self.__dict__)
                    if i[0] != "_"
                ]
            )
        )

    def _list(self, anything):
        """Convert user input into list if not already."""
        if not anything:
            return []
        elif type(anything) in [str, int, float]:
            return [anything]
        elif isinstance(anything, np.ndarray):
            return anything.tolist()
        elif isinstance(anything, list):
            return anything
        else:
            print(f"{type(anything)} not taken care of by _list")

    def _savgol(self, data, window=10, polyorder=3, nosmoothing=False):
        if nosmoothing:
            return data
        window = round(window)
        if window % 2 == 0:
            window += 1
        return signal.savgol_filter(data, window, polyorder)

    def _print(self, thing):
        self.log = thing
        if self._verbosity:
            print(self._log[-1])
