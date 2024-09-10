from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import sem
from scipy.stats import gaussian_kde
from scipy.stats import bootstrap
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


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
                    if ("Permeation up 3 -> 1 -> 4" in l) or ("Permeation down 3 -> 1 -> 4" in l) or "None" in l or l == "\n":
                        i = i + j - 1
                        break
                    else:
                        words = l.split(",")
                        p_index.append(int(words[0]))
                        p_time.append(float(words[2]))
                        p_resident_time.append(float(words[3]))
                        p_up.append(True)
            elif "Permeation up 3 -> 1 -> 4" in l or ("Permeation down 3 -> 1 -> 4" in l):
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


def line_to_state(line, get_state_str=True):
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
    if get_state_str:
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
    else:
        return n_pot, n_wat


def _s6l_function1_original(line):
    """
    This is the basic S6l function which works for K and POT.
    return the last word which is the 6 letter code
    :param line:
    :return: str
    """
    return line.split()[-1]

def _s6l_function2_nonK(line, ion):
    sites = line.rstrip().split(",")
    letters = ""
    if ion in sites[0][7:]:
        letters += "K"
    elif "Wat" in sites[0][7:]:
        letters += "W"
    else:
        letters += "0"
    for s in sites[1:-1]:
        if ion in s:
            letters += "K"
        elif "Wat" in s:
            letters += "W"
        else:
            letters += "0"
    return letters

def s6l_Co_occu(K_occ_tmp, W_occ_tmp):
    """
    return the 6-letter code for Co-occupancy
    Args:
        K_occ_tmp: a np.array() of K occupancy
        W_occ_tmp: a np.array() of W occupancy
    return: str
    """
    letters = ""
    for k, w in zip(K_occ_tmp, W_occ_tmp):
        if k > 0 and w > 0:
            letters += "C"
        elif k > 0:
            letters += "K"
        elif w > 0:
            letters += "W"
        else:
            letters += "0"
    return letters


def read_k_cylinder(file, method="K_priority", get_occu=True, get_jump=False):
    """
    read 1 output file from k_cylinder
    Args:
        file: output file from k_cylinder
        method: "K_priority" or "Co-occupy" or "K_priority_S14"
            In "K_priority", if there is a K in the binding site, letter K will be assigned.
            In "Co-occupy", if there is a K and one or more water in the binding site, letter C will be assigned.
            In "K_priority_S14", the same as "K_priority", but only the S1-S4 are considered.
        get_occu: if True, compute K_occupency and W_occupency for K and W occupancy. otherwise, return empty list,
            default True
        get_jump: if True, read the ion jump information from the ion index, default False.
    return:
        state_list, states string is a list
        meta_data, a dict
        K_occupency, a np.array() of K occupancy
        W_occupency, a np.array() of W occupancy
        jump_array, a np.array( dtype=np.int8) of ion jump, if get_jump is True
    """
    state_list = []
    jump_convertion_LUT_6 = np.array(
        [[ 0,  1,  2,  3,  4,  5, -1],
         [-1,  0,  1,  2,  3,  4, -2],
         [-2, -1,  0,  1,  2,  3, -3],
         [-3, -2, -1,  0,  1,  2,  3],
         [-4, -3, -2, -1,  0,  1,  2],
         [-5, -4, -3, -2, -1,  0,  1],
         [ 1,  2,  3, -3, -2, -1,  0]
         ]
    )
    jump_convertion_LUT_4 = np.array(
        [[ 0,  1,  2, -2, -1],
         [-1,  0,  1,  2,  3],
         [-2, -1,  0,  1,  2],
         [ 2, -2, -1,  0,  1],
         [ 1, -3, -2, -1,  0]
         ]
    )

    with open(file) as f:
        lines = f.readlines()
        meta_data = {}
        ion_list = [""]
        for line_num, l in enumerate(lines):
            if "Sfilter Version" in l:
                meta_data["version"] = l.split()[-1]
            elif "Ion name(s) in this pdb" in l:
                l = l.rstrip()
                ion_list = eval(l.split(":")[-1])
                if len(ion_list) != 1:
                    raise ValueError("There should be one and only one ion in this output file.")
                if ion_list == ["POT"] or ion_list == ["K"]:
                    s6l_fun = _s6l_function1_original
                else:
                    def s6l_fun(line):
                        return _s6l_function2_nonK(line, ion=ion_list[0])
                # s6l_fun function take the first line in a frame and return the 6-letter code in K_priority
                meta_data["ion_name"] = ion_list[0]
            elif "time step in this xtc is" in l:
                meta_data["time_step"] = float(l.split()[-2])
            elif f"Number of {ion_list[0]} found is" in l:
                meta_data["num_ion"] = int(l.split()[-1])
                # read the ion index from i+1 until "]" appears
                ion_index = [int(k) for k in lines[line_num + 1][1:].split()]
                for j in range(line_num + 2, len(lines)):
                    if "]" in lines[j]:
                        # remove "]"
                        l = lines[j].rstrip()[:-1]
                        ion_index.extend([int(k) for k in l.split()])
                        break
                    ion_index.extend([int(k) for k in lines[j].split()])
                meta_data["ion_index"] = ion_index

            elif "# S6l" in l:
                break
        K_occupency = []
        W_occupency = []
        for i0, line in enumerate(lines):
            if "# S6l" in line:
                break
        if "########" in lines[-1]:
            lines_frames = lines[i0:-1]  # count_cylinder.py properly finished, the last line is "########"
        else:
            # check if the last frame is finished
            lines_frames = lines[i0:]
            for line_num, line in enumerate(lines[-1:-7:-1]):
                if "# S6l" in line: # broken frame, we ignore the last frame from this line
                    lines_frames = lines[i0:-line_num - 1]
                    break
        if method == "K_priority":
            if not get_occu and not get_jump:
                for line_num in range(0, len(lines_frames), 7):
                    state_list.append(s6l_fun(lines_frames[line_num]))
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency)
            elif get_occu and not get_jump:
                for line_num in range(0, len(lines_frames), 7):
                    state_list.append(s6l_fun(lines_frames[line_num]))
                    K_occ_tmp = np.zeros(6, dtype=np.int8)
                    W_occ_tmp = np.zeros(6, dtype=np.int8)
                    for j in range(0, 6):
                        pot, wat = lines_frames[line_num + j + 1].split(",")[:2]
                        K_occ_tmp[j] = len(pot.split(":")[1].split())
                        W_occ_tmp[j] = len(wat.split(":")[1].split())
                    K_occupency.append(K_occ_tmp)
                    W_occupency.append(W_occ_tmp)
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency)
            elif get_occu and get_jump:
                jump_list = []
                a_size = max(meta_data["num_ion"], max(meta_data["ion_index"]) - min(meta_data["ion_index"]))
                ion_state_array_old = np.zeros(a_size, dtype=np.int8)+ 6
                ion_state_array_i   = np.zeros(a_size, dtype=np.int8) + 6
                index_delta = min(meta_data["ion_index"])

                # initial state for frame 0
                for site in range(0, 6):
                    pot, wat = lines_frames[site + 1].split(",")[:2]
                    pot_index = [int(ipot)-index_delta for ipot in pot.split(":")[1].split()]
                    ion_state_array_old[pot_index] = site


                for line_num in range(0, len(lines_frames), 7):
                    state_list.append(s6l_fun(lines_frames[line_num]))
                    K_occ_tmp = np.zeros(6, dtype=np.int8)
                    W_occ_tmp = np.zeros(6, dtype=np.int8)
                    ion_state_array_i[:] = 6
                    for site in range(0, 6):
                        pot, wat = lines_frames[line_num + site + 1].split(",")[:2]
                        pot_index = [int(ipot)-index_delta for ipot in pot.split(":")[1].split()]
                        ion_state_array_i[pot_index] = site
                        K_occ_tmp[site] = len(pot_index)
                        W_occ_tmp[site] = len(wat.split(":")[1].split())
                    # compute the jump using LUT
                    jump = -jump_convertion_LUT_6[ion_state_array_old, ion_state_array_i].sum()
                    jump_list.append(jump)
                    ion_state_array_old[:] = ion_state_array_i
                    K_occupency.append(K_occ_tmp)
                    W_occupency.append(W_occ_tmp)
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency), np.array(jump_list, dtype=np.int8)
            else:
                raise ValueError("get_occu=False and get_jump=True would not work. Please give True for both")

        elif method == "Co-occupy":
            if not get_occu and not get_jump:
                for line_num in range(0, len(lines_frames), 7):

                    s_code0, _1, _2 = line_to_state(lines_frames[line_num + 1])
                    s_code1, _1, _2 = line_to_state(lines_frames[line_num + 2])
                    s_code2, _1, _2 = line_to_state(lines_frames[line_num + 3])
                    s_code3, _1, _2 = line_to_state(lines_frames[line_num + 4])
                    s_code4, _1, _2 = line_to_state(lines_frames[line_num + 5])
                    s_code5, _1, _2 = line_to_state(lines_frames[line_num + 6])
                    state_list.append( s_code0 + s_code1 + s_code2 + s_code3 + s_code4 + s_code5 )
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency)
            elif get_occu and not get_jump:
                for line_num in range(0, len(lines_frames), 7):
                    K_occ_tmp = np.zeros(6, dtype=np.int8)
                    W_occ_tmp = np.zeros(6, dtype=np.int8)
                    s_code0, K_occ_tmp[0], W_occ_tmp[0] = line_to_state(lines_frames[line_num + 1])
                    s_code1, K_occ_tmp[1], W_occ_tmp[1] = line_to_state(lines_frames[line_num + 2])
                    s_code2, K_occ_tmp[2], W_occ_tmp[2] = line_to_state(lines_frames[line_num + 3])
                    s_code3, K_occ_tmp[3], W_occ_tmp[3] = line_to_state(lines_frames[line_num + 4])
                    s_code4, K_occ_tmp[4], W_occ_tmp[4] = line_to_state(lines_frames[line_num + 5])
                    s_code5, K_occ_tmp[5], W_occ_tmp[5] = line_to_state(lines_frames[line_num + 6])
                    state_list.append( s_code0 + s_code1 + s_code2 + s_code3 + s_code4 + s_code5 )
                    K_occupency.append(K_occ_tmp)
                    W_occupency.append(W_occ_tmp)
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency)
            elif get_occu and get_jump:
                jump_list = []
                a_size = max(meta_data["num_ion"], max(meta_data["ion_index"]) - min(meta_data["ion_index"]))
                ion_state_array_old = np.zeros(a_size, dtype=np.int8) + 6
                ion_state_array_i   = np.zeros(a_size, dtype=np.int8) + 6
                index_delta = min(meta_data["ion_index"])

                # initial state for frame 0
                for site in range(0, 6):
                    pot, wat = lines_frames[site + 1].split(",")[:2]
                    pot_index = [int(ipot) - index_delta for ipot in pot.split(":")[1].split()]
                    ion_state_array_old[pot_index] = site

                for line_num in range(0, len(lines_frames), 7):
                    K_occ_tmp = np.zeros(6, dtype=np.int8)
                    W_occ_tmp = np.zeros(6, dtype=np.int8)
                    ion_state_array_i[:] = 6
                    for site in range(0, 6):
                        pot, wat = lines_frames[line_num + site + 1].split(",")[:2]
                        pot_index = [int(ipot) - index_delta for ipot in pot.split(":")[1].split()]
                        ion_state_array_i[pot_index] = site
                        K_occ_tmp[site] = len(pot_index)
                        W_occ_tmp[site] = len(wat.split(":")[1].split())
                    state_list.append(s6l_Co_occu(K_occ_tmp, W_occ_tmp))
                    # compute the jump using LUT
                    jump = -jump_convertion_LUT_6[ion_state_array_old, ion_state_array_i].sum()
                    jump_list.append(jump)
                    ion_state_array_old[:] = ion_state_array_i
                    K_occupency.append(K_occ_tmp)
                    W_occupency.append(W_occ_tmp)
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency), np.array(jump_list, dtype=np.int8)
            else:
                raise ValueError("get_occu=False and get_jump=True would not work. Please give True for both")

        elif method == "K_priority_S14":
            if not get_occu and not get_jump:
                for line_num in range(0, len(lines_frames), 7):
                    state_list.append(s6l_fun(lines_frames[line_num])[1:5])
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency)
            elif get_occu and not get_jump:
                for line_num in range(0, len(lines_frames), 7):
                    state_list.append(s6l_fun(lines_frames[line_num])[1:5] )
                    K_occ_tmp = np.zeros(4, dtype=np.int8)
                    W_occ_tmp = np.zeros(4, dtype=np.int8)
                    for j in range(1, 5):
                        pot, wat = lines_frames[line_num + j + 1].split(",")[:2]
                        K_occ_tmp[j-1] = len(pot.split(":")[1].split())
                        W_occ_tmp[j-1] = len(wat.split(":")[1].split())
                    K_occupency.append(K_occ_tmp)
                    W_occupency.append(W_occ_tmp)
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency)
            elif get_occu and get_jump:
                jump_list = []
                a_size = max(meta_data["num_ion"], max(meta_data["ion_index"]) - min(meta_data["ion_index"]))
                ion_state_array_old = np.zeros(a_size, dtype=np.int8)
                ion_state_array_i = np.zeros(a_size, dtype=np.int8)
                index_delta = min(meta_data["ion_index"])

                # initial state for frame 0
                for site in range(1, 5):
                    pot, wat = lines_frames[site + 1].split(",")[:2]
                    pot_index = [int(ipot) - index_delta for ipot in pot.split(":")[1].split()]
                    ion_state_array_old[pot_index] = site

                for line_num in range(0, len(lines_frames), 7):
                    state_list.append(s6l_fun(lines_frames[line_num])[1:5])
                    K_occ_tmp = np.zeros(4, dtype=np.int8)
                    W_occ_tmp = np.zeros(4, dtype=np.int8)
                    ion_state_array_i[:] = 0
                    for site in range(1, 5):
                        pot, wat = lines_frames[line_num + site + 1].split(",")[:2]
                        pot_index = [int(ipot) - index_delta for ipot in pot.split(":")[1].split()]
                        ion_state_array_i[pot_index] = site
                        K_occ_tmp[site-1] = len(pot_index)
                        W_occ_tmp[site-1] = len(wat.split(":")[1].split())
                    # compute the jump using LUT
                    jump = -jump_convertion_LUT_6[ion_state_array_old, ion_state_array_i].sum()
                    jump_list.append(jump)
                    ion_state_array_old[:] = ion_state_array_i
                    K_occupency.append(K_occ_tmp)
                    W_occupency.append(W_occ_tmp)
                return state_list, meta_data, np.array(K_occupency), np.array(W_occupency), np.array(jump_list, dtype=np.int8)
        else:
            raise ValueError("method should be K_priority, Co-occupy, or K_priority_S14")

def read_k_cylinder_list(file_list, method="K_priority", get_occu=True):
    """
    read a list of output file from k_cylinder
    This is designed for reading a sequence of MD simulation that can be concatenated together.
    The first frame of the second (and later) simulation will be discarded. Gromacs writes the initial frame to the traj
    file, which is the same as the last frame of the previous simulation.
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
        state_list, meta_data, K_occupency, W_occupency = read_k_cylinder(file_list, method, get_occu)
    elif isinstance(file_list, list):
        # make sure file exists
        for f in file_list:
            if not Path(f).exists():
                raise FileNotFoundError(f)
        if len(file_list) == 1:
            state_list, meta_data, K_occupency, W_occupency = read_k_cylinder(file_list[0], method, get_occu)
        else:
            state_list, meta_data, K_occupency, W_occupency = read_k_cylinder(file_list[0], method, get_occu)
            for f in file_list[1:]:
                s_list_tmp, meta_data_tmp, K_occu_tmp, W_occu_tmp = read_k_cylinder(f, method, get_occu)
                state_list.extend(s_list_tmp[1:])
                if meta_data_tmp != meta_data:
                    raise ValueError("meta_data is different in different files. Please check " + str(f))
                # K_occupency.extend(K_occu_tmp[1:])
                # W_occupency.extend(W_occu_tmp[1:])
                K_occupency = np.concatenate((K_occupency, K_occu_tmp[1:]))
                W_occupency = np.concatenate((W_occupency, W_occu_tmp[1:]))
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
        :param method: "K_priority" or "Co-occupy" or "K_priority_S14"
            In "K_priority", if there is a K in the binding site, letter K will be assigned.
            In "Co-occupy", if there is a K and one or more water in the binding site, letter C will be assigned.
            In "K_priority_S14", the same as "K_priority", but only the S1-S4 are considered.
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
        for f in self.files:
            s_list, meta_data, K_occu, W_occu = read_k_cylinder(f, method, get_occu=False)
            s_list_sliced = s_list[start:end:step]
            self.state_str.append(s_list_sliced)
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

    def _bootstrap_worker(self, traj, state_list, n_resamples, confidence_level, method, kwargs):
        """
        Worker function to perform bootstrap for a single trajectory.
        """
        def proportion(traj):
            counter = Counter(traj)
            return [counter[s] / counter.total() for s in state_list]

        bootstrap_res = bootstrap((traj,), proportion, n_resamples=n_resamples,
                                  confidence_level=confidence_level, method=method, **kwargs)
        prop = proportion(traj)
        return [[s, p, bootstrap_res.confidence_interval.low[i], bootstrap_res.confidence_interval.high[i]]
                for i, (s, p) in enumerate(zip(state_list, prop))]

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
        # state_distribution, counter_list, counter_all = self.get_state_distribution(state_list)

        # Collect all futures here
        futures = []

        # Using ProcessPoolExecutor to parallelize the task
        with ProcessPoolExecutor() as executor:
            for i, traj in enumerate(self.state_str):
                # Submit the bootstrap worker function to the executor for each trajectory
                future = executor.submit(self._bootstrap_worker, traj, state_list, n_resamples,
                                         confidence_level, method, kwargs)
                futures.append(future)

        # Now retrieve the results as they are completed
        result = []
        for i, future in enumerate(futures):
            print(f"Processing trajectory {i}")
            state_proportion = future.result()
            # Update the state_distribution with the bootstrap results
            result.append(state_proportion)

        print("Done")
        return result

        # state_distribution, counter_list, counter_all = self.get_state_distribution()
        # state_list = [i[0] for i in counter_all.most_common()]
        # state_distribution, counter_list, counter_all = self.get_state_distribution(state_list)
        # def proportion(traj):
        #     counter = Counter(traj)
        #     return [counter[s] / counter.total() for s in state_list]
        # for i, (traj, state_proportion) in enumerate(zip(self.state_str, state_distribution)):
        #     print(i, end=" ")
        #     bootstrap_res = bootstrap((traj,), proportion, n_resamples=n_resamples, confidence_level=confidence_level,
        #                               method=method, **kwargs)
        #     for i, s in enumerate(state_proportion):
        #         s.append(bootstrap_res.confidence_interval.low[i])
        #         s.append(bootstrap_res.confidence_interval.high[i])
        # print("Done")
        # return state_distribution

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
