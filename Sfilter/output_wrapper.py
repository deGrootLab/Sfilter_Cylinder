from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import sem
from scipy.stats import bootstrap


def read_cylinder(cylinder_file):
    """
    :param cylinder_file:
    :return: a pd.DataFrame
    """
    with open(cylinder_file) as f:
        lines = f.readlines()
        i = 0
        p_index = []
        p_time = []
        p_resident_time = []
        p_up = []
        meta_data_dict = {}
        while i < len(lines):
            l = lines[i]
            if ("Permeation up 4 -> 1 -> 3" in l):
                for j in range(2, len(lines) - i):
                    l = lines[i + j]
                    if ("Permeation up 3 -> 1 -> 4" in l) or "None" in l or l == "\n":
                        i = i + j - 1
                        break
                    else:
                        words = l.split(",")
                        p_index.append(int(words[0]))
                        p_time.append(float(words[2]))
                        p_resident_time.append(float(words[3]))
                        p_up.append(True)
            elif ("Permeation up 3 -> 1 -> 4" in l):
                for j in range(2, len(lines) - i):
                    l = lines[i + j]
                    if ("##############" in l) or "None" in l or l == "\n":
                        i = i + j
                        break
                    else:
                        words = l.split(",")
                        p_index.append(int(words[0]))
                        p_time.append(float(words[2]))
                        p_resident_time.append(float(words[3]))
                        p_up.append(False)
            elif "Assumed voltage (mV)" in l:
                meta_data_dict["voltage"] = float(l.split(":")[1])
            elif "Simulation time (ns)" in l:
                meta_data_dict["time"] = float(l.split(":")[1])
            elif "Permeation events up" in l:
                meta_data_dict["perm_up"] = int(l.split(":")[1])
            elif "Permeation events down" in l:
                meta_data_dict["perm_down"] = int(l.split(":")[1])
            elif "Ave current (pA)" in l:
                meta_data_dict["ave_current"] = float(l.split(":")[1])
            elif "Ave conductance (pS)" in l:
                meta_data_dict["ave_conductance"] = float(l.split(":")[1])

            i += 1
        df = pd.DataFrame({"index": p_index,
                           "enter": np.array(p_time) - np.array(p_resident_time),
                           "time": p_time,
                           "resident_time": p_resident_time,
                           "up": p_up})
        df.sort_values(by=["time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, meta_data_dict


class Perm_event_output:
    """
    A class for reading (a list of) permeation-event output
    """

    def __init__(self, files):
        if isinstance(files, Path) or isinstance(files, str):
            self.files = [files]
        elif isinstance(files, list):
            self.files = files
        else:
            raise TypeError("files must be a list of str or str")
        self.perm_list = []  # a list of pd.DataFrame
        self.meta_data = []
        for f in self.files:
            df, meta_data_dict = read_cylinder(f)
            self.perm_list.append(df)
            self.meta_data.append(meta_data_dict)

    def get_conductance(self, begin=None, end=None, voltage=300):
        """
        get averaged conductance, with s.e.m. of different trajectories
        Args:
            begin: starting time in ps
            end: end time in ps
            voltage: voltage in mV
        :return:
            average conductance in pS
            s.e.m. of conductance in pS if there are more than one trajectory, otherwise None
            a list of conductance from each trajectory
        """
        if begin is None and end is None:  # return the number from file
            conductance_list = []
            for i in self.meta_data:
                conductance_list.append(i["ave_conductance"])
            if len(conductance_list) == 1:
                error = None
            else:
                error = sem(conductance_list)
            return np.mean(conductance_list), error, conductance_list
        elif begin is None:
            begin = 0
        elif end is None:
            raise ValueError("end should be specified if begin is specified")

        # calculate conductance for part of the trajectory
        conductance_list = []
        for df in self.perm_list:
            time_mask = (df["time"] >= begin) & (df["time"] <= end)
            up_count = df[time_mask & df["up"]].shape[0]
            down_count = df[time_mask & ~df["up"]].shape[0]
            conductance_list.append((up_count - down_count) / (end - begin) * 1.602176634 * 1e8 / voltage)
        if len(conductance_list) == 1:
            error = None
        else:
            error = sem(conductance_list)
        return np.mean(conductance_list), error, conductance_list

    def get_kde_conductance(self, begin, end, bins, bandwidth):
        """
        get continuous conductance smoothed by kde for each trajectory
        Args:
            begin: starting time in ps
            end: end time in ps
            bins: sample points of kde
            bandwidth: bandwidth of kde
        :return:
            kde smoothed conductance in pS for each trajectory in a list
        """
        pass

    def get_bootstrap_conductance(self, begin, end, n_resamples=9999, confidence_level=0.95,
                                  alternative='two-sided', method='BCa'
                                  ):
        """
        Compute a two-sided bootstrap confidence interval of the average conductance
        won't work if there is only one trajectory
        This is a wrapper on top of scipy.stats.bootstrap
        Args:
            begin: starting time in ps
            end: end time in ps
            n_resamples: number of bootstrap
            confidence_level: The confidence level of the confidence interval.
            alternative: {‘two-sided’, ‘less’, ‘greater’}, default: 'two-sided'
            method: {‘percentile’, ‘basic’, ‘bca’}, default: 'BCa'
        Returns:
            resBootstrapResult
                confidence_interval : ConfidenceInterval
                    The bootstrap confidence interval as an instance of collections.namedtuple with attributes low and high.
            bootstrap_distributio : nndarray
                The bootstrap distribution, that is, the value of statistic for each resample. The last dimension corresponds with the resamples (e.g. res.bootstrap_distribution.shape[-1] == n_resamples).
            standard_error : float or ndarray
                The bootstrap standard error, that is, the sample standard deviation of the bootstrap distribution.
        """
        pass

# bootstrapping or SEM

# class for reading (a list of) std_out
# bootstrapping traj
# bootstrapping frame
