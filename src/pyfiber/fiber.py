"""
This module mainly defines the Fiber Class.
Most of the function are tied to the class in order to take advantage of attributes to set default values.
However, important functions such as the ``fiber.normalize_signal`` and ``fiber.detect_peaks`` are defined outside
the class as they can be useful on their own."""

import numpy as np
import pandas as pd
import random
import time
import h5py
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from typing import List, Union
import matplotlib.style as st
from ._utils import PyFiber as PyFiber

st.use("ggplot")


__all__ = ["Fiber", "normalize_signal"]


def normalize_signal(
    signal: np.ndarray, control: np.ndarray, method: str
) -> np.ndarray:
    """Normalize the signal using a control channel.

    :param signal: calcium-dependant signal
    :param control: isosbestic signal
    :type signal: ``np.ndarray``
    :type control: ``np.ndarray``
    :param method: normalization method

    .. note::
       Two normalization method can be used. ``'F'`` is the standard :math:`\\frac{\\Delta F}{F_0}`.
       ``'Z'`` is the z-score difference of signal and control.
    """
    if method == "F":
        coeff = np.polynomial.polynomial.polyfit(control, signal, 1)
        fitted_control = coeff[0] + control * coeff[1]
        normalized = (signal - fitted_control) / fitted_control
    if method == "Z":
        S = (signal - signal.mean()) / signal.std()
        I = (control - control.mean()) / control.std()
        normalized = S - I
    return normalized


def detect_peaks(
    arr=[],
    time=[],
    signal=[],
    window: float = 10,
    distance: str = "50ms",
    plot: bool = True,
    figsize: tuple = (30, 10),
    zscore: str = "full",
    bMAD: float = 2.5,
    pMAD: float = 3.5,
    xlim: Union[bool, tuple] = False,
    save: Union[bool, str] = False,
    **kwargs,
):
    """Detect peaks in a normalized fiberphotometry signal.

    :param time: timestamps
    :param signal: fiber photometry signal
    :param window: size of each succesive bins were peaks are computed (default is 10s)
    :param distance: minimum distance between successive peaks (in milliseconds), see :ref:`note <spfp>`
    :param plot: output a plot
    :param figsize: figure size
    :param zscore: defines if standardisation should occur at the bin level or the whole recording level
    :param bMAD: baseline treshold (all timepoint below ``bMAD`` times the MAD are removed)
    :param pMAD: peak treshold (peaks have to be at least ``pMAD`` times higher than the computed baseline)
    :param kwargs: keyword argument passed to ``scipy.signal.find_peaks``, see :ref:`note <spfp>`

    .. _spfp:
    .. note::
       The final peak detection is handled by ``scipy.signal.find_peaks``. By default, distance, provided in string format
       in milliseconds (ex: '50ms') is converted in a number of samples based on the provided timestamp array.
       Additionnally, any keyword argument catched by **kwargs** is passed to ``find_peaks`` so more precise criteria
       can be used.
    """
    if (not len(arr)) and ((not len(signal)) or (not len(time))):
        return
    elif len(arr):
        time = arr[:, 0]
        signal = arr[:, 1]
    else:
        pass
    dF = signal.copy()
    # calculate zscores
    if zscore == "full":
        signal = (signal - signal.mean()) / signal.std()

    # distance
    distance = round(float(distance.split("ms")[0]) / (np.mean(np.diff(time)) * 1000))
    if distance == 0:
        distance = 1

    # find indexes for n second windows
    t_points = np.arange(time[0], time[-1], window)
    if t_points[-1] != time[-1]:
        t_points = np.concatenate((t_points, [time[-1]]))
    indexes = [np.where(abs(time - i) == abs(time - i).min())[0][0] for i in t_points]

    # create time bins
    bins = [
        pd.Series(
            signal[indexes[i - 1] : indexes[i]], index=time[indexes[i - 1] : indexes[i]]
        )
        for i in range(1, len(indexes))
    ]
    dFbins = [
        pd.Series(
            dF[indexes[i - 1] : indexes[i]], index=time[indexes[i - 1] : indexes[i]]
        )
        for i in range(1, len(indexes))
    ]
    if zscore == "bins":
        bins = [(b - b.mean()) / b.std() for b in bins]

    # find median for each bin and remove events >(bMAD)*MAD for baselines
    baselines = [
        b[b < np.median(b) + bMAD * np.median(abs(b - np.median(b)))] for b in bins
    ]

    # calculate peak tresholds for each bin, by default >(pMAD)*MAD of previously created baseline
    tresholds = [
        np.median(b) + pMAD * np.median(abs(b - np.median(b))) for b in baselines
    ]

    # find peaks using scipy.signal.find_peaks with thresholds
    peaks = []
    for n, bin_ in enumerate(bins):
        b = pd.DataFrame(bin_).reset_index()
        indices, heights = find_peaks(
            b.iloc[:, 1], height=tresholds[n], distance=distance, **kwargs
        )
        peaks.append(
            pd.DataFrame(
                {
                    "time": [b.iloc[i, 0] for i in indices],
                    "dF/F": [dFbins[n].iloc[i] for i in indices],
                    "zscore": list(heights.values())[0],
                }
            )
        )
    # plot
    if plot:
        plt.figure(figsize=figsize)
        peak_tresholds = [
            pd.Series(t, index=baselines[n].index) for n, t in enumerate(tresholds)
        ]
        bin_medians = [
            pd.Series(np.median(b), index=bins[n].index) for n, b in enumerate(bins)
        ]
        bin_mad = [
            pd.Series(np.median(abs(b - np.median(b))), index=bins[n].index)
            for n, b in enumerate(bins)
        ]
        for n, i in enumerate(bins):
            c = random.choice(list("bgrcmy"))
            plt.plot(i, alpha=0.6, color=c)
            plt.plot(
                baselines[n],
                color="gray",
                alpha=0.5,
                label=n * "_" + f"signal <{bMAD}MAD + median",
            )  # keeps only the first label
            plt.plot(bin_medians[n], color="k", label=n * "_" + "signal median")
            plt.plot(
                bin_medians[n] + bin_mad[n] * bMAD,
                color="darkgray",
                label=n * "_" + f">{bMAD}MAD + median",
            )
            plt.plot(
                peak_tresholds[n], color="r", label=n * "_" + f">{pMAD}MAD + baseline"
            )
        for n, p in enumerate(peaks):
            plt.scatter(p.loc[:, "time"], p.loc[:, "zscore"])
        if xlim:
            plt.xlim(xlim)
        plt.legend()
        if save:
            plt.savefig(f"{save}.pdf")
            plt.savefig(f"{save}.png", dpi=600)
        plt.show()
    return pd.concat(peaks, ignore_index=True)


class Fiber(PyFiber):
    """Extract and analyse fiberphotometry data.

    :ivar data: dataframe with all data from data file
    :ivar recordings: all separate recording
    :ivar number_of_recordings: number of separate recordings (see :ref:`note below <recs>`)
    :ivar peaks: dictionnary containing a data frame with peak data (timestamps, dF/F, zscore) for all recordings
    :ivar sampling rate: mean sampling rate accross recordings
    :ivar full_time: array containing all timestamps from the data file
    :ivar rec_intervals: time intervals for each recording
    :ivar rec_length: duration of each recording (as a list)
    :ivar df: raw data as a ``pandas.DataFrame``
    :ivar ncl: nomenclature for reading columns, defined in the config file
    :ivar columns: name of the retrieved columns (*i.e.* time, signal, control)
    :ivar raw_columns: name of the original columns
    :ivar ID: see **parameters**
    :ivar alignment: see **parameters**
    :ivar filepath: see **parameters**
    :ivar filetype: see **parameters**
    :ivar name: see **parameters**
    :cvar FIBER_FILE_TYPE: (from ``_utils.PyFiber``) system file type (default: DORIC_CSV)
    :cvar split_recordings: (from ``_utils.PyFiber``) see :ref:`note below <recs>`
    :cvar split_treshold: (from ``_utils.PyFiber``) treshold for splitting data into recording (*e.g*. 10 is 10 times the mean intersample distance)
    :cvar min_sample_per_rec: (from ``_utils.PyFiber``) minimun number of samples for a split to be considered a recordings (ignores one sample recordings that are bugs)
    :cvar trim_recording: (from ``_utils.PyFiber``) number of seconds automatically removed from recording start (to avoid the LED artifact)
    :cvar perievent_window: (from ``_utils.PyFiber``) default perievent analysis window
    :cvar default_norm: (from ``_utils.PyFiber``) default normalization method ('F' or 'Z', see detail in ``pyfiber.fiber.normalize_signal``)
    :cvar peak_window: (from ``_utils.PyFiber``) length in seconds of the peak detection window
    :cvar peak_distance: (from ``_utils.PyFiber``) minimum distance between peaks (to avoid false positives), in milliseconds
    :cvar peak_zscore: (from ``_utils.PyFiber``) **full** or **bins** choose if post normalization standardization should be done on the full recordings or for each window
    :cvar peak_baseline_MAD: (from ``_utils.PyFiber``) first treshold for peak analysis
    :cvar peak_peak_MAD: (from ``_utils.PyFiber``) second treshold for peak analysis

    :param filepath: fiber photometry data file path
    :type filepath: ``str``
    :param name: session name (default is filepath without the extension)
    :type name: ``str``
    :param ID: animal identifier
    :type ID: hashable
    :param alignment: optional shift in the timestamps
    :type alignment: ``float``
    :param filetype: data file type, defaults to Doric
    :type filetype: ``str``

    .. note::
       Alignment can be used if the exact time of recording start is provided by a behavioral system.
       It simply shift every timestamps of the fiber file accordingly.

    .. _recs:
    .. note::
       In the fiber photometry data file, locations where the timestamps are discontinuous define breakpoints
       where the data is split into **recordings**. To change this behavior, simply add ``split_recordings = False``
       at instantiation, or modify the configuration file :

    .. code-block:: python

       >> import pyfiber
       >> f = pyfiber.Fiber('filepath.csv', split_recordings = False)

    .. code-block:: yaml

       # ~/.pyfiber/pyfiber.yaml
       [...]
       FIBER:
         split_recordings: False
       [...]
    """

    vars().update(PyFiber.FIBER)

    def __init__(
        self,
        filepath: str,
        name: str = "default",
        ID: str = None,
        alignment: float = 0,
        filetype: str = None,
        **kwargs,
    ):
        start = time.time()
        super().__init__(**kwargs)

        # READ ARGUMENTS
        self.alignment = alignment
        self.filepath = filepath
        self._print(f"Importing {filepath}...")
        self.ID = ID

        # GENERATE NAME AND READ METADATA
        if name == "default":
            self.name = self.filepath.split(".csv")[0]
            if self.ID:
                self.name += ID
        self.number_of_recording = 0
        if filetype:
            self.filetype = filetype
            self.ncl = self.SYSTEM[filetype.upper()]
        else:
            if filepath.split(".")[-1] == "doric":
                self.filetype = "DORIC_HDF"
                self.ncl = self.SYSTEM["DORIC_HDF"]
            else:
                self.filetype = "DORIC_CSV"
                self.ncl = self.SYSTEM["DORIC_CSV"]

        # EXTRACT FILE
        self.df = self._read_file(filepath, alignment=alignment)
        self.full_time = np.array(
            self.df[[k for k, v in self.ncl.items() if v == "time"][0]]
        )
        self.raw_columns = list(self.df.columns)
        self.data = self._extract_data()
        self.columns = list(self.data.columns)

        # (OPTIONAL) SPLIT RECORDINGS
        if self.split_recordings:
            self.recordings = self._split_recordings()
        else:
            self.recordings = {1: self.data}

        # GET RECORDING METADATA
        self.rec_length = np.array(
            [
                v.time.to_numpy()[-1] - v.time.to_numpy()[0]
                for k, v in self.recordings.items()
            ]
        )
        self.sampling_rate = np.mean(
            [len(df) / t for df, t in zip(self.recordings.values(), self.rec_length)]
        )
        self.rec_intervals = [
            tuple(
                [self.recordings[recording]["time"].values[index] for index in [0, -1]]
            )
            for recording in range(1, self.number_of_recording + 1)
        ]

        # ANALYZE PEAKS FOR EACH RECORDING
        self.peaks = {}
        self._print("Analyzing peaks...")
        for r in self.recordings.keys():
            data = self.norm(rec=r, add_time=True)
            t = data[:, 0]
            s = data[:, 1]
            self.peaks[r] = self._detect_peaks(time=t, signal=s, plot=False)

        self._print(
            f"Importing of {filepath} finished in {time.time() - start} seconds"
        )

    def __repr__(self):
        """Give general information about the recording data."""
        general_info = f"""\
File                     : {self.filepath}
ID                       : {self.ID}
Number of recordings     : {self.number_of_recording}
Data columns             : {self.columns}
Total span               : {self.full_time[-1] - self.full_time[0]} s
Recording lengths        : {self.rec_length} ({self.trim_recording} seconds trimmed from each)
Global sampling rate     : {self.sampling_rate} S/s
Aligned to behavior file : {self.alignment} s
"""
        return general_info

    def _find_rec(self, timestamp: float) -> list:
        """Find recording number corresponding to inputed timestamp.

        :param timestamp: timestamp of which to find the recording number if any"""
        rec_num = self.number_of_recording
        time_nom = "time"
        return [
            i
            for i in range(1, rec_num + 1)
            if self.get(time_nom, recording=i)[0]
            <= timestamp
            <= self.get(time_nom, recording=i)[-1]
        ]

    def _read_file(self, filepath, alignment=0) -> pd.DataFrame:
        """Read file and align the timestamps if specified.

        :param filepath: the data file path
        :param alignment: offset in seconds to align fiber datafile to behavioral data
        """
        if self.filetype == "DORIC_HDF":
            f = h5py.File(self.filepath, "r")
            df = pd.DataFrame({v: pd.Series(f[k]) for k, v in self.ncl.items()})
            inverse_hdf_nom = {v: k for k, v in self.ncl.items()}
            df["time"] = df["time"] + alignment
            df.columns = [inverse_hdf_nom[i] for i in df.columns]
            print(df)
            f.close()
        else:
            df = pd.read_csv(
                filepath,
                usecols=[
                    k for k, v in self.ncl.items() if v in ["signal", "control", "time"]
                ],
                dtype=np.float64,
            )  # ,engine='pyarrow')
            time_nom = [k for k, v in self.ncl.items() if v == "time"][0]
            df[time_nom] = df[time_nom] + alignment
        return df

    def _extract_data(self) -> pd.DataFrame:
        """Extract raw fiber data from Doric system."""
        return pd.DataFrame(
            {
                self.ncl[i]: self.df[i].to_numpy()
                for i in self.ncl
                if i in self.raw_columns
            }
        )

    def _split_recordings(self) -> dict:
        """Cut at timestamp jumps.

        The jumps are defined by a step greater than N times the mean sample space (defined in pyfiber.yaml).
        """
        time = self.full_time
        jumps = list(
            np.where(np.diff(time) > self.split_treshold * np.mean(np.diff(time)))[0]
            + 1
        )
        indices = [0] + jumps + [len(time) - 1]
        ind_tuples = [
            (indices[i], indices[i + 1])
            for i in range(len(indices) - 1)
            if indices[i + 1] - indices[i] > self.min_sample_per_rec
        ]
        self.number_of_recording = len(ind_tuples)
        self._print(f"Found {self.number_of_recording} separate recordings.")
        rec = {
            ind_tuples.index((s, e)) + 1: self.data.iloc[s:e, :] for s, e in ind_tuples
        }
        t_ = "time"
        rec = {
            k: v[v[t_] > v[t_].to_numpy()[0] + self.trim_recording]
            for k, v in rec.items()
        }
        return rec

    def plot(
        self,
        which: Union[str, int, List[int]] = "all",
        method: str = "default",
        figsize: tuple = (20, 20),
        raw_label: list = ["Ca-dependant", "isosbestic"],
        norm_label: str = "Normalized signal",
        xlabel: str = "Time (s)",
        ylabel_raw: str = "Signal (mV)",
        ylabel_norm: str = "Signal (%)",
        title: str = "Recording",
        hspace: float = 0.5,
    ) -> None:
        """Plot fiber data.

        :param which: which recording should be plotted ; default is 'all'
        :param method: normalization method (see ``fiber.normalize_signal`` for details)
        :param figsize: figure size
        :param raw_label: plotting labels for the non normalized data
        :param norm_label: plotting labels for the normalized data
        :param xlabel: label for the x-axis
        :param ylabel_raw: y-axis label for non-normalized data
        :param ylabel_norm: y-axis label for normalized data
        :param title: plot title
        :param hspace: horizontal space between graphs"""
        if which == "all":
            recs = list(self.recordings.keys())
        else:
            recs = self._list(which)
        data = [self.norm(rec=i, method=method, add_time=True) for i in recs]
        rawdata = [self.norm(rec=i, method="raw", add_time=True) for i in recs]
        n = len(recs)
        plt.figure(figsize=figsize)
        for i in range(n):
            plt.subplot(n, 2, int(i * 2) + 1)
            plt.plot(rawdata[i][:, 0], rawdata[i][:, 1:], label=raw_label)
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel_raw)
            plt.title(title + f" ({i+1})")

            plt.subplot(n, 2, int(2 * (i + 1)))
            plt.plot(
                data[i][:, 0], data[i][:, 1], c="g", label=norm_label + f" ({method})"
            )
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel_norm)
            plt.title(title + f" ({i+1})")
        plt.subplots_adjust(hspace=hspace)

    def to_csv(
        self,
        recordings: Union[str, int, List[int]] = "all",
        auto: bool = True,
        columns: List[str] = None,
        column_names: List[str] = None,
        prefix: str = "default",
    ) -> None:
        """Export data to csv.

        :param recording: defines which recording to export (default is all)
        :param auto: automatically export signal and isosbestic separately with sampling_rate
        :param columns: export specific columns with their timestamps
        :param columns_names: change default name for columns in outputted csv (the list must correspond to the column list)
        :param prefix: filename prefix (default is '<raw filename><recording number><data column name>'')
        """
        if prefix == "default":
            prefix = self.name
        if recordings == "all":
            recordings = list(self.recordings.keys())
        sig_nom = "signal"
        ctrl_nom = "control"
        time_nom = "time"
        if auto and not columns:
            nomenclature = {sig_nom: "signal", ctrl_nom: "control"}
            for rec in recordings:
                time = self.get(time_nom, rec)
                for dataname in nomenclature.keys():
                    df = pd.DataFrame(
                        {
                            "timestamps": time,
                            "data": self.get(dataname, rec),
                            "sampling_rate": [1 / np.diff(time)]
                            + [np.nan] * (len(time) - 1),
                        }
                    )  # timestamps data sampling rate
                    df.to_csv(
                        os.path.join(
                            Fiber.FOLDER, f"{prefix}_{rec}_{nomenclature[dataname]}.csv"
                        ),
                        index=False,
                    )
        else:
            recordings = self._list(recordings)
            columns = self._list(columns)
            column_names = self._list(column_names)
            for r in recordings:
                time = self.get(time_nom, r)
                for c in columns:
                    df = pd.DataFrame(
                        {
                            "timestamps": time,
                            "data": self.get(c, r),
                            "sampling_rate": [1 / np.diff(time)]
                            + [np.nan] * (len(time) - 1),
                        }
                    )
                    df.to_csv(f"{prefix}_{r}_{c}.csv", index=False)

    def get(
        self,
        column: str,
        recording: int = 1,
        add_time: bool = False,
        as_df: bool = False,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Extracts a data array from a specific column of a recording.

        :param column: dataframe column to extract (list of columns: ``self.columns``)
        :param add_time: if ``True`` return a 2D array (stack of time and signal)
        :param as_df: if ``True`` return data as a data frame (add_time will automatically be set to ``True``)

        .. note::
           This function is a shortcut to ``<obj>.recordings[<rec_number>][<'column_name'>].to_numpy()`` with
           multiple queries (time) as an option.
        """
        time_nom = "time"
        data = np.array(self.recordings[recording][column])
        time = np.array(self.recordings[recording][time_nom])
        if as_df:
            return pd.DataFrame({time_nom: time, column: data})
        if add_time:
            return np.vstack((time, data)).T
        else:
            return data

    # def TTL(self, ttl, rec=1):
    #     """Output TTL timestamps."""
    #     ttl = self.get(f"TTL{ttl}", rec)
    #     time = self.get('time', rec)
    #     ttl[ttl < 0.01] = 0
    #     ttl[ttl >= 0.01] = 1
    #     if (ttl == 1).all():
    #         return [time[0]]
    #     if (ttl == 0).all():
    #         return []
    #     index = np.where(np.diff(ttl) == 1)[0]
    #     return [time[i] for i in index]

    def norm(
        self, rec: int = 1, method: str = "default", add_time: bool = True
    ) -> np.ndarray:
        """Normalize data with specified method.

        :param rec: recording number
        :param method: normalization method (see :ref`note <n>`)
        :param add_time: include timestamps with output

        .. _n:
        .. note::
           This method is a wrapper of the ``fiber.normalize_signal`` function, see its documentation for more details.
        """
        sig = self.get("signal", rec)
        ctrl = self.get("control", rec)
        tm = self.get("time", rec)
        if method == "default":
            method = self.default_norm
        self._print(f"Normalizing recording {rec} with method '{method}'")
        if method == "raw" or not method:
            normalized = np.vstack((sig, ctrl))
        else:
            normalized = normalize_signal(signal=sig, control=ctrl, method=method)
        if add_time:
            return np.vstack((tm, normalized)).T
        else:
            return normalized

    def _detect_peaks(
        self,
        rec=False,
        arr=[],
        time=[],
        signal=[],
        window: Union[str, float] = "default",
        distance: str = "default",
        plot: bool = True,
        figsize: tuple = (30, 10),
        zscore: str = "full",
        bMAD: Union[str, float] = "default",
        pMAD: Union[str, float] = "default",
        **kwargs,
    ) -> pd.DataFrame:
        """Detect peaks in a normalized fiberphotometry signal.

        API for ``fiber.detect_peaks``, with class default.
        :param arr: numpy array with timestamps in first column and signal values in second column
        :param time: timestamps
        :param signal: fiber photometry signal
        :param window: size of each succesive bins were peaks are computed (default is 10s)
        :param distance: minimum distance between successive peaks (in milliseconds), see :ref:`note <spfp>`
        :param plot: output a plot
        :param figsize: figure size
        :param zscore: defines if standardisation should occur at the bin level or the whole recording level
        :param bMAD: baseline treshold (all timepoint below ``bMAD`` times the MAD are removed)
        :param pMAD: peak treshold (peaks have to be at least ``pMAD`` times higher than the computed baseline)
        :param kwargs: keyword argument passed to ``scipy.signal.find_peaks``, see :ref:`note <spfp>`

        .. _spfp:
        .. note::
           The final peak detection is handled by ``scipy.signal.find_peaks``. By default, distance, provided in string format
           in milliseconds (ex: '50ms') is converted in a number of samples based on the provided timestamp array.
           Additionnally, any keyword argument catched by **kwargs** is passed to ``find_peaks`` so more precise criteria
           can be used.
        """
        if window == "default":
            window = self.peak_window
        if distance == "default":
            distance = self.peak_distance
        if zscore == "default":
            zscore = self.peak_zscore
        if bMAD == "default":
            bMAD = int(self.peak_baseline_MAD)
        else:
            bMAD = int(bMAD)
        if pMAD == "default":
            pMAD = int(self.peak_peak_MAD)
        else:
            pMAD = int(pMAD)
        if rec:
            arr = self.norm(rec)
        df = detect_peaks(
            arr=arr,
            signal=signal,
            time=time,
            distance=distance,
            window=window,
            plot=plot,
            figsize=figsize,
            zscore=zscore,
            bMAD=bMAD,
            pMAD=pMAD,
            **kwargs,
        )
        return df

    def plot_transients(
        self,
        value: str = "zscore",
        figsize: tuple = (20, 20),
        rec: Union[str, int] = "all",
        colors: str = "k",
        alpha: float = 0.3,
        save=False,
        save_dpi=600,
        save_format=["png", "pdf"],
        **kwargs,
    ) -> None:
        """Show graphical representation of detected transients with their amplitude.

        :param value: show either Z-scores (``'zscore``) or :math:`\\frac{\\Delta F}{F_0}` (``'dF/F'``)
        :param figsize: figure size
        :param rec: recordings to plot, default is all
        :param colors: bar colors
        :param alpha: transparency
        :param kwargs: keyworg arguments passed to ``matplotlib``
        :param save: default is ``False``, filepath (without file extension)
        :param save_dpi: dpi if figure is saved
        :param save_format: file extension for saving"""
        if rec == "all":
            rec = self.number_of_recording
        fig, axes = plt.subplots(rec, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes.grid(which="both")
            data = self.peaks[1]
            for i in data.index:
                axes.vlines(
                    data.loc[i, "time"],
                    ymin=0,
                    ymax=data.loc[i, value],
                    colors=colors,
                    alpha=alpha,
                    **kwargs,
                )
        else:
            for n, ax in enumerate(axes):
                ax.grid(which="both")
                data = self.peaks[n + 1]
                for i in data.index:
                    ax.vlines(
                        data.loc[i, "time"],
                        ymin=0,
                        ymax=data.loc[i, value],
                        colors=colors,
                        alpha=alpha,
                        **kwargs,
                    )
        if save:
            for ext in self._list(save_format):
                plt.savefig(f"{save}.{ext}", dpi=save_dpi)
        plt.show()

    def peakFA(self, a: float, b: float) -> dict:
        """Return peak analysis: frequency and amplitude results for given interval.

        :param a: left limit of the interval
        :param b: right limit of the interval
        :return: Mean frequency, mean peak amplitude, max peak amplitude, full peak data frame (all peaks from the interval)
        """
        r = 0
        for n, i in enumerate(self.rec_intervals):
            if (i[0] <= a < i[1]) and (i[0] < b <= i[1]):
                r = n + 1
        if r == 0:
            return
        data = self.peaks[r][(self.peaks[r]["time"] > a) & (self.peaks[r]["time"] < b)]
        return {
            "peak_frequency": len(data) / (b - a),
            "peak_avg_Z": data["zscore"].mean(),
            "peak_avg_dFF": data["dF/F"].mean(),
            "peak_max_Z": data["zscore"].max(),
            "peak_max_dFF": data["dF/F"].max(),
            "data": data,
        }
