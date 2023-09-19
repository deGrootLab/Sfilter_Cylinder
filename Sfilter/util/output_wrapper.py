from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import sem
from scipy.stats import gaussian_kde
from scipy.stats import bootstrap
from collections import Counter


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
            if "Permeation up 4 -> 1 -> 3" in l:
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
            elif "Permeation up 3 -> 1 -> 4" in l:
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
                meta_data_dict["time"] = float(l.split(":")[1]) * 1000  # ps
            elif "Permeation events up" in l:
                meta_data_dict["perm_up"] = int(l.split(":")[1])
            elif "Permeation events down" in l:
                meta_data_dict["perm_down"] = int(l.split(":")[1])
            elif "Ave current (pA)" in l:
                meta_data_dict["ave_current"] = float(l.split(":")[1])
            elif "Ave conductance (pS)" in l:
                meta_data_dict["ave_conductance"] = float(l.split(":")[1])
            elif "Sfilter Version" in l:
                meta_data_dict["version"] = l.split()[-1]
            elif "time step in this xtc" in l:
                meta_data_dict["time_step"] = float(l.split()[-2])

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
        """
        :param files: pathlib.Path or str or a list of them
        """
        if isinstance(files, Path) or isinstance(files, str):
            self.files = [files]
        elif isinstance(files, list):
            self.files = files
        else:
            raise TypeError("files must be a list of str or str")
        self.perm_list = []  # a list of pd.DataFrame
        self.meta_data = []
        self.sim_time = []
        for f in self.files:
            df, meta_data_dict = read_cylinder(f)
            self.perm_list.append(df)
            self.meta_data.append(meta_data_dict)
            self.sim_time.append(meta_data_dict["time"])

    def cut_event(self, begin, end, time=None):
        """
        restrict permeation event in a certain time range
        Args:
            begin: starting time in ps
            end: end time in ps
            time: modify the simulation time to time (ps)
        """
        if (not (begin is None)) and (not (end is None)):
            if (time is None):
                time = end - begin
            else:
                pass
        else:
            raise ValueError("begin and end must be specified")
        for i in range(len(self.perm_list)):
            df = self.perm_list[i]
            mid = (df["time"] + df["enter"]) / 2
            time_mask = (mid >= begin) & (mid <= end)
            self.perm_list[i] = df[time_mask]
            self.meta_data[i]["time"] = time
            self.sim_time[i] = time

    def get_conductance(self, voltage=None):
        """
        Use the number of permeation events to compute the conductance. If the voltage is not specified, use the voltage
        from meta_data
        get averaged conductance, with s.e.m. of different trajectories
        Args:
            voltage: voltage in mV
        :return: ave_conductance, SEM, conductance_list
            average conductance in pS
            s.e.m. of conductance in pS if there are more than one trajectory, otherwise None
            a list of conductance from each trajectory
        """

        if voltage is None:
            for m in self.meta_data:
                if m["voltage"] != self.meta_data[0]["voltage"]:
                    raise ValueError("voltage is not specified and different trajectories have different voltages")
            voltage = self.meta_data[0]["voltage"]
        # calculate conductance for part of the trajectory
        conductance_list = []
        for df, time in zip(self.perm_list, self.sim_time):
            up_count = df[df["up"]].shape[0]
            down_count = df[~df["up"]].shape[0]
            conductance_list.append((up_count - down_count) / time * 1.602176634 * 1e8 / voltage)
        if len(conductance_list) == 1:
            error = None
        else:
            error = sem(conductance_list)
        return np.mean(conductance_list), error, conductance_list

    def get_kde_conductance(self, evaluation_points, bandwidth, voltage=300, event_seqaration=2, up=True):
        """
        get continuous conductance smoothed by kde for each trajectory
        By default, each event is separated to 2 events, 0.5 each, entering and leaving. You can separate them more by using \
        event_seqeration
        Args:
            bins: sample points of kde
            bandwidth: bandwidth of kde
            voltage: voltage in mV
            event_seqaration: 2 or more
            up: if True, only up events are considered, otherwise only down events are considered
        :return:
            kde smoothed conductance in pS for each trajectory in a list
        """
        kde_cond_list = []  # a list to save the kde smoothed conductance
        conductance_u_list = []
        conductance_d_list = []
        for df in self.perm_list:
            conductance = []
            conductance_u_list.append([])
            conductance_d_list.append([])
            for i in range(df.shape[0]):
                if up and df.iloc[i]["up"]:
                    conductance.extend(
                        np.linspace(df.iloc[i]["enter"], df.iloc[i]["time"], event_seqaration)
                    )
                elif (not up) and (not df.iloc[i]["up"]):
                    conductance.extend(
                        np.linspace(df.iloc[i]["enter"], df.iloc[i]["time"], event_seqaration)
                    )
                if df.iloc[i]["up"]:
                    conductance_u_list[-1].append([df.iloc[i]["enter"], df.iloc[i]["time"]])
                else:
                    conductance_d_list[-1].append([df.iloc[i]["enter"], df.iloc[i]["time"]])
            kde_cond_list.append(
                gaussian_kde(conductance, bw_method=bandwidth).evaluate(evaluation_points)
                * 1.602176634 * 1e8 / voltage * len(conductance) / event_seqaration)  # convert to pS
        return kde_cond_list, (conductance_u_list, conductance_d_list)

    def get_bootstrap_conductance(self, voltage=300, n_resamples=9999, confidence_level=0.95,
                                  method='BCa', **kwargs):
        """
        Compute a two-sided bootstrap confidence interval of the average conductance
        won't work if there is only one trajectory
        This is a wrapper on top of scipy.stats.bootstrap
        Args:
            voltage: voltage in mV
            n_resamples: number of bootstrap
            confidence_level: The confidence level of the confidence interval.
            method: {‘percentile’, ‘basic’, ‘bca’}, default: 'BCa'
        Returns:
            resBootstrapResult
                confidence_interval : ConfidenceInterval
                    The bootstrap confidence interval as an instance of collections.namedtuple with attributes low and \
                    high.
            bootstrap_distributio : np.darray
                The bootstrap distribution, that is, the value of statistic for each resample. The last dimension \
                corresponds with the resamples (e.g. res.bootstrap_distribution.shape[-1] == n_resamples).
            standard_error : float or ndarray
                The bootstrap standard error, that is, the sample standard deviation of the bootstrap distribution.
        """
        if len(self.perm_list) == 1:
            raise ValueError("There is only one trajectory, bootstrap won't work")
        ave, error, conductance_list = self.get_conductance(voltage)
        bootstrap_res = bootstrap((conductance_list,), np.mean, n_resamples=n_resamples,
                                  confidence_level=confidence_level,
                                  method=method, **kwargs)
        return bootstrap_res


def line_to_state(line):
    """
    convert a line of k_cylinder output to a state code
    return:
        state_str: "0" for empty, "W" for water, "C" for cation, "K" for potassium
        n_pot: number of potassium
        n_wat: number of water
    """
    pot, wat = line.split(",")[:2]
    n_pot = len(pot.split(":")[1].split())
    n_wat = len(wat.split(":")[1].split())
    if n_pot == 0 and n_wat == 0:
        state_str = "0"
    elif n_pot == 0 and n_wat > 0:
        state_str = "W"
    elif n_pot > 0 and n_wat > 0:
        state_str = "C"
    elif n_pot > 0 and n_wat == 0:
        state_str = "K"
    else:
        raise ValueError("n_pot and n_wat should be positive")
    return state_str, n_pot, n_wat


def read_k_cylinder(file, method="K_priority"):
    """
    read 1 output file from k_cylinder
    Args:
        file: output file from k_cylinder
        method: "K_priority" or "Co-occupy"
        In "K_priority", if there is a K in the binding site, letter K will be assigned.
        In "Co-occupy", if there is a K and one or more water in the binding site, letter C will be assigned.
    return:
        state_list, states string is a list
        meta_data, a dict
        K_occupency, a list of K occupancy
        W_occupency, a list of W occupancy
    """
    state_list = []

    with open(file) as f:
        lines = f.readlines()
        meta_data = {}
        for l in lines:
            if "Sfilter Version" in l:
                meta_data["version"] = l.split()[-1]
            elif "time step in this xtc is" in l:
                meta_data["time_step"] = float(l.split()[-2])
                break
        K_occupency = []
        W_occupency = []
        if method == "K_priority":
            i = 0
            while i < len(lines):
                l = lines[i]
                if "# S6l" in l:
                    state_list.append(l.split()[-1])
                    K_occupency.append([])
                    W_occupency.append([])
                    for j in range(1, 7):
                        try:
                            s_code, n_pot, n_wat = line_to_state(lines[i + j])
                        except:
                            raise ValueError(i+j, lines[i + j], "There is something wrong with this line")
                        K_occupency[-1].append(n_pot)
                        W_occupency[-1].append(n_wat)
                    i += 6
                else:
                    i += 1
            # for l in lines:
            #     if "# S6l" in l:
            #         K_occupency.append([])
            #         W_occupency.append([])
            #         s = l.split()[-1]
            #         state_list.append(s)
        elif method == "Co-occupy":
            i = 0
            while i < len(lines):
                l = lines[i]
                state_str = ""
                if "# S6l" in l:
                    K_occupency.append([])
                    W_occupency.append([])
                    for j in range(1, 7):
                        s_code, n_pot, n_wat = line_to_state(lines[i + j])
                        state_str += s_code
                        K_occupency[-1].append(n_pot)
                        W_occupency[-1].append(n_wat)
                    state_list.append(state_str)
                    i += 6
                else:
                    i += 1
        else:
            raise ValueError("method should be K_priority or Co-occupy")
    return state_list, meta_data, K_occupency, W_occupency

def read_k_cylinder_list(file_list, method="K_priority"):
    """
    read a list of output file from k_cylinder
    This is designed for reading a sequence of MD simulation that can be concatenated together.
    Args:
        file_list: one file or a list of files.
        method: "K_priority" or "Co-occupy"
        In "K_priority", if there is a K in the binding site, letter K will be assigned.
        In "Co-occupy", if there is a K and one or more water in the binding site, letter C will be assigned.
    return:
        state_list, states string is a list
        meta_data, a dict
        K_occupency, a list of K occupancy
        W_occupency, a list of W occupancy
    """
    # if file_list is a str, read file using read_k_cylinder, otherwise, read files one by one
    if isinstance(file_list, str) or isinstance(file_list, Path):
        state_list, meta_data, K_occupency, W_occupency = read_k_cylinder(file_list, method)
    elif isinstance(file_list, list):
        # make sure file exists
        for f in file_list:
            if not Path(f).exists():
                raise FileNotFoundError(f)
        if len(file_list) == 1:
            state_list, meta_data, K_occupency, W_occupency = read_k_cylinder(file_list[0], method)
        else:
            state_list, meta_data, K_occupency, W_occupency = read_k_cylinder(file_list[0], method)
            for f in file_list[1:]:
                s_list_tmp, meta_data_tmp, K_occu_tmp, W_occu_tmp = read_k_cylinder(f, method)
                state_list.extend(s_list_tmp[1:])
                if meta_data_tmp != meta_data:
                    raise ValueError("meta_data is different in different files. Please check " + str(f))
                K_occupency.extend(K_occu_tmp[1:])
                W_occupency.extend(W_occu_tmp[1:])
    else:
        raise TypeError("file_list must be a str or a list of str")

    return state_list, meta_data, K_occupency, W_occupency



# class for reading (a list of) std_out
class Cylinder_output:
    def __init__(self, files, start=0, end=None, step=1, method="K_priority", time_step=None):
        """
        read output file from k_cylinder
        :param files: pathlib.Path or str or list of them. This should be the std_out from count_cylinder.py
        :param start: default 0.
        :param end: default None, final frame.
        :param step: default 1, no slicing.
        if you need to subsample the trajectory by slicing, you can provide start, end, step.
        :param method: "K_priority" or "Co-occupy"
        :param time_step: The time step in the trajectory. If None, use the time step in the output file
        """
        if isinstance(files, Path) or isinstance(files, str):
            self.files = [files]
        elif isinstance(files, list):
            self.files = files
        else:
            raise TypeError("files must be a list of str or str")
        self.state_str = []
        self.meta_data = []
        data_list = []
        for f in self.files:
            data_list.append(read_k_cylinder(f, method))
        for s_list, meta_data, K_occupency, W_occupency in [read_k_cylinder(f, method) for f in self.files]:
            self.state_str.append(s_list[start:end:step])
            if time_step is None:
                if "time_step" not in meta_data:
                    raise ValueError("time_step is not provided in the output file, please provide it manually")
                else:
                    meta_data["time_step"] *= step
            else:
                meta_data["time_step"] = time_step * step
            self.meta_data.append(meta_data)

    def get_state_distribution(self, state_list=None):
        """
        get the state proportion for all the trajectories
        if state_list is provided, return the state distribution based on state_list order
        return:
            state_distribution: list of list, each list is the state proportion pair of one trajectory
                [[[state1, proportion1], [state2, proportion2], ...] # traj1
                 [[state1, proportion1], [state2, proportion2], ...] # traj2 ]
            counter_list: list of Counter, each count states for one trajectory
            counter_all: Counter, the state count all the trajectories
        """
        state_distribution = []
        counter_all = Counter()
        counter_list = []
        for traj in self.state_str:
            counter = Counter(traj)
            counter_list.append(counter)
            counter_all += counter
            distri = []
            total = counter.total()
            if state_list is None:
                for s, n in counter.most_common():
                    distri.append([s, n / total])
            else:
                for s in state_list:
                    distri.append([s, counter[s] / total])

            state_distribution.append(distri)
        return state_distribution, counter_list, counter_all

    def get_state_distribution_CI_bootstrap_frame(self, n_resamples=9999, confidence_level=0.95,
                                            method='BCa', **kwargs):
        """
        Assume each frame is independent, and each trajectory is a different condition.
        get state proportion for each trajectory (with confidence interval).
        input:
            n_resamples: number of bootstrap
            confidence_level: The confidence level of the confidence interval.
            method: {‘percentile’, ‘basic’, ‘bca’}, default: 'BCa'
            All the other args are passed to scipy.stats.bootstrap.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
        return:
            state_distribution: list of list, each list is the state proportion pair of one trajectory
                [[[state1, proportion1, low, high], [state2, proportion2, low, high], ...] # traj1
                 [[state1, proportion1, low, high], [state2, proportion2, low, high], ...] # traj2
                 ...]
        """
        state_distribution, counter_list, counter_all = self.get_state_distribution()
        state_list = [i[0] for i in counter_all.most_common()]
        state_distribution, counter_list, counter_all = self.get_state_distribution(state_list)

        def proportion(traj):
            counter = Counter(traj)
            res = []
            for s in state_list:
                res.append(counter[s] / counter.total())
            return res

        for i, (traj, state_proportion) in enumerate(zip(self.state_str, state_distribution)):
            print(i, end=" ")
            bootstrap_res = bootstrap((traj,), proportion, n_resamples=n_resamples, confidence_level=confidence_level,
                                      method=method, **kwargs)
            for i, s in enumerate(state_proportion):
                s.append(bootstrap_res.confidence_interval.low[i])
                s.append(bootstrap_res.confidence_interval.high[i])
        print("Done")

        return state_distribution

    def get_state_distribution_CI_bootstrap_traj(self, n_resamples=9999, confidence_level=0.95,
                                                 method='BCa', **kwargs):
        """
        Assume each trajectory is an independent repeat.
        Get state proportion with confidence interval on the whole sets of trajectories.
        input:
            n_resamples: number of bootstrap
            confidence_level: The confidence level of the confidence interval.
            method: {‘percentile’, ‘basic’, ‘bca’}, default: 'BCa'
            All the other args are passed to scipy.stats.bootstrap.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
        :return:
        state_distribution: a list of state proportion pair
                [[state1, proportion1, low, high],
                 [state2, proportion2, low, high],
                 ...]
        """
        if len(self.files) == 1:
            raise ValueError("Only one trajectory, cannot bootstrap trajectory")
        state_distribution_trajs, counter_list, counter_all = self.get_state_distribution()
        state_list = [i[0] for i in counter_all.most_common()]
        # make sure all states are included in the correct order
        state_distribution_trajs, counter_list, counter_all = self.get_state_distribution(state_list)

        state_distribution = []
        for i, s in enumerate(state_list):
            proportion_trajs = [p[i][1] for p in state_distribution_trajs]
            bootstrap_res = bootstrap((proportion_trajs,), np.mean, n_resamples=n_resamples,
                                      confidence_level=confidence_level,
                                      method=method, **kwargs)
            state_distribution.append([s, np.mean(proportion_trajs),
                                       bootstrap_res.confidence_interval.low,
                                       bootstrap_res.confidence_interval.high])
        return state_distribution

# bootstrapping traj
# bootstrapping frame
