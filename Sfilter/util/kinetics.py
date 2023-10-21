import warnings
from collections import Counter
import numpy as np
import pandas as pd
from .output_wrapper import read_k_cylinder
from collections import Counter
import gc
import networkx as nx
import matplotlib.pyplot as plt


def count_passage(traj_list, num_of_node, print_progress=False):
    """
    Count the passage on a list of trajectory.
    :param traj_list: A list of trajectory. Each trajectory is a list(np.array) of int.
    :param num_of_node: number of nodes. Assume the nodes are 0, 1, 2, ..., num_of_node-1.
    :return:
        passage_time_length_every_traj
            a list of matrix. One matrix for each traj. matrix[i][j] is a list of passage time(unit in step)
            from node_i to node_j.
        passage_time_point_every_traj
            a list of matrix. One matrix for each traj. matrix[i][j] is a list of time point (unit in step)
            when the passage from node_i to node_j finished .
    """
    passage_time_length_every_traj = []
    passage_time_point_every_traj = []  # the time point when the passage finished
    traj_count = 0
    for traj in traj_list:
        if print_progress:
            traj_count += 1
            print(f"{traj_count}/{len(traj_list)}", end=" ")
        passage_time_length = []
        passage_time_point = []
        for i in range(num_of_node):
            passage_time_length.append([[] for _ in range(num_of_node)])
            passage_time_point.append([[] for _ in range(num_of_node)])
        starting_time = np.zeros((num_of_node, num_of_node), dtype=np.int64) - 1
        j = traj[0]
        starting_time[j, :] = 0
        for frame_i, (si, sj) in enumerate(zip(traj[:-1], traj[1:])):
            if si != sj:
                for i, start_i in enumerate(starting_time[:, sj]):
                    if start_i != -1 and i != sj:
                        # successful passage from i to sj, from time start_i to frame_i+1
                        passage_time_length[i][sj].append(frame_i + 1 - start_i)
                        passage_time_point[i][sj].append(frame_i + 1)
                        starting_time[i, sj] = -1
                starting_time[sj, :][starting_time[sj, :] == -1] = frame_i + 1

        # Convert list to numpy array to save memory
        for i in range(num_of_node):
            for j in range(num_of_node):
                passage_time_length[i][j] = np.array(passage_time_length[i][j], dtype=np.int64)
                passage_time_point[i][j] = np.array(passage_time_point[i][j], dtype=np.int64)

        passage_time_length_every_traj.append(passage_time_length)
        passage_time_point_every_traj.append(passage_time_point)
    if print_progress:
        print()

    return passage_time_length_every_traj, passage_time_point_every_traj


class Sf_model:
    """
    This is a class to analyse the mechanism of a selectivity filter.
    """

    def __init__(self, file_list=None, start=0, end=None, step=1, method="K_priority", lag_step=1, traj_dtype=np.int16):
        """
        This is the normal way to initialize a SF_model object. You can provide a list of output files from count_cylinder.py.
        :param file_list: file name or a list of file names. This should be the std_out from count_cylinder.py.
            If None, you can set everything later.
        :param start: starting frame. The default is 0.
        :param end: end frame. The default is None (final).
        :param step: step frame, steps when reading trajectory, default is 1 (every frame).
        :param method: method to calculate the state, default is "K_priority".
            "K_priority", if there is a K in the binding site, letter K will be assigned.
            "Co-occupy", if there is a K and one or more water in the binding site, letter C will be assigned.
            No other method is implemented.
        :param lag_step: lag step for calculating properties (transition matrix), default is 1.
        :param traj_dtype: data type of the trajectory. default is np.int8.
        """
        # check arguments and initialize variables
        if file_list is None:
            self.file_list = None
        elif isinstance(file_list, str):
            self.file_list = [file_list]
        elif isinstance(file_list, list):
            self.file_list = file_list

        # variables for raw trajectory
        self.time_step = 0  # time step between frames
        self.total_frame = 0  # total number of frames
        self.traj_raw = []  # raw trajectory, a list of np.array. not lumped
        self.state_map_int_2_s = {}  # map from int to state
        self.state_map_s_2_int = {}  # map from state to int
        self.state_Counter = None  # Counter of states(str)
        self.state_distribution = {}  # proportion of each state(str)

        # variables for lumped trajectory
        self.traj_node = []  # lumped trajectory
        self.node_map_int_2_s = {}  # what does each node mean (in self.traj_lumped). key:int, value: list of str
        self.node_map_s_2_int = {}  # Which node does a state belong to (in self.traj_lumped). key:str, value: int
        self.node_Counter = None  # Counter of nodes, key is int
        self.node_distribution = {}  # proportion of each node, key is int
        self.node_distribution_str = {}  # proportion of each node, key is tuple of string

        # variables for properties
        self.lag_step = lag_step  # lag step for computing properties
        self.flux_matrix = None
        self.flux_matrix_every_traj = None
        self.flux_raw = None  # raw traj
        self.net_flux_raw = None  # raw traj
        self.transition_probability_matrix = None
        self.transition_probability_matrix_every_traj = None
        self.net_flux_matrix = None
        self.net_flux_matrix_every_traj = None
        self.passage_time_point_every_traj = None  # lumped traj
        self.passage_time_length_every_traj = None  # lumped traj
        self.passage_time_length_every_traj_raw = None  # raw traj
        self.rate_raw = None
        self.mechanism_G = None

        # initialization finished

        # read file(s)
        if self.file_list is not None:
            time_step_list = []
            traj_tmp_list = []
            map_s_2_int = {}
            map_int_2_s = {}
            state_index = 0
            for file in self.file_list:
                print(file)
                traj, meta_data, K_occupency, W_occupency = read_k_cylinder(file, method, get_occu=False)
                time_step_list.append(meta_data["time_step"])
                unique_state = sorted(set(traj))
                for s in unique_state:
                    if s not in map_s_2_int:
                        map_s_2_int[s] = state_index
                        map_int_2_s[state_index] = s
                        state_index += 1
                traj = traj[start:end:step]
                traj_int = np.array([map_s_2_int[s] for s in traj], dtype=traj_dtype)
                traj_tmp_list.append(traj_int)
                del traj  # free memory
                del K_occupency
                del W_occupency
                gc.collect()
            # check if time step (float) is almost the same
            if not np.allclose(time_step_list, time_step_list[0]):
                raise ValueError("The time step between files are not the same.", str(time_step_list))

            self.set_traj_from_int(traj_tmp_list, time_step_list[0] * step, map_int_2_s, dtype_lumped=traj_dtype)

    def _init_raw_properties(self, dtype_lumped=np.int16):
        self.set_lumping_from_str([], dtype_lumped, calc_passage_time=True,
                                  build_graph=False)  # At the end of this function, properties will be calculated.
        self.flux_raw = self.calc_flux_matrix()
        self.net_flux_raw = self.calc_net_flux_matrix()
        self.calc_passage_time_raw()
        self.rate_raw, _ = self.get_rate_passage_time(traj_type="raw")

        # build a DataFrame for the raw traj
        index_A = []
        index_B = []
        index_pair = []
        A_list = []
        B_list = []
        A_proportion = []
        B_proportion = []
        net_flux_AB = []
        flux_AB = []
        flux_BA = []
        rate_AB = []
        rate_BA = []
        dist_AB = []
        dist_BA = []
        for i in range(len(self.net_flux_raw)):
            for j in range(len(self.net_flux_raw)):
                if i != j:
                    if self.flux_raw[i, j] > 0 and self.net_flux_raw[i, j] >= 0 and ((j, i) not in index_pair):
                        index_pair.append((i, j))
                        index_A.append(i)
                        index_B.append(j)
                        A_list.append(self.state_map_int_2_s[i])
                        B_list.append(self.state_map_int_2_s[j])
                        A_proportion.append(self.state_distribution[self.state_map_int_2_s[i]])
                        B_proportion.append(self.state_distribution[self.state_map_int_2_s[j]])
                        net_flux_AB.append(self.net_flux_raw[i, j])
                        flux_AB.append(self.flux_raw[i, j])
                        flux_BA.append(self.flux_raw[j, i])
                        rate_AB.append(self.rate_raw[i, j])
                        rate_BA.append(self.rate_raw[j, i])
                        d_ab, d_ba = k_distance(self.state_map_int_2_s[i], self.state_map_int_2_s[j])
                        dist_AB.append(d_ab)
                        dist_BA.append(d_ba)
        self.raw_traj_df = pd.DataFrame({"index_A": index_A, "index_B": index_B, "A": A_list, "B": B_list,
                                         "A_proportion": A_proportion, "B_proportion": B_proportion,
                                         "net_flux_AB": net_flux_AB, "flux_AB": flux_AB, "flux_BA": flux_BA,
                                         "rate_AB": rate_AB, "rate_BA": rate_BA,
                                         "dist_AB": dist_AB, "dist_BA": dist_BA})
        self.raw_traj_df["rate_AB_x_rate_BA"] = self.raw_traj_df["rate_AB"] * self.raw_traj_df["rate_BA"]
        self.build_graph()

    def set_traj_from_str(self, traj_list, time_step, dtype=np.int16, dtype_lumped=np.int16):
        """
        Set the trajectory from lists of string.
        :param traj_list: a list of traj. Each traj is a sequence of str.
        :param time_step: time step between frames.
        :param dtype: data type of the trajectory. default is np.int16.
        :return: None
        """
        if dtype not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError("dtype should be np.int8, np.int16, np.int32, np.int64.")

        self.time_step = time_step
        self.state_Counter = Counter([s for traj in traj_list for s in traj])
        self.total_frame = self.state_Counter.total()

        # loop over state_Counter from the most common state to the least common state
        for i, (s, frame) in enumerate(self.state_Counter.most_common()):
            self.state_distribution[s] = frame / self.total_frame
            self.state_map_int_2_s[i] = s
            self.state_map_s_2_int[s] = i
            self.node_map_int_2_s[i] = [s]
            self.node_map_s_2_int[s] = i

        self.traj_raw = []
        for traj in traj_list:
            self.traj_raw.append(np.array([self.state_map_s_2_int[s] for s in traj], dtype=dtype))

        self._init_raw_properties(dtype_lumped=dtype_lumped)

    def set_traj_from_int(self, traj_list, time_step, map_int_2_s, dtype=np.int16, dtype_lumped=np.int16):
        """
        Set the trajectory from lists of int.
        :param traj_list: a list of np.array(). Each np.array() is a sequence of int.
        :param time_step: time step between frames.
        :param map_int_2_s: map from int to state (str).
        :param dtype: data type of the trajectory. default is np.int16.
        :param dtype_lumped: data type of the lumped trajectory. default is np.int16.
        :return: None
        """
        if dtype not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError("dtype should be np.int8, np.int16, np.int32, np.int64.")
        if dtype_lumped not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError("dtype_lumped should be np.int8, np.int16, np.int32, np.int64.")

        self.time_step = time_step
        Counter_tmp = Counter([])
        for traj in traj_list:
            Counter_tmp.update(traj)

        self.state_Counter = Counter([])
        self.total_frame = Counter_tmp.total()
        map_old_2_new = {}
        for i_new, (i_old, n) in enumerate(Counter_tmp.most_common()):
            self.state_Counter[map_int_2_s[i_old]] = n
            self.state_distribution[map_int_2_s[i_old]] = n / self.total_frame
            self.state_map_int_2_s[i_new] = map_int_2_s[i_old]
            self.state_map_s_2_int[map_int_2_s[i_old]] = i_new
            self.node_map_int_2_s[i_new] = [map_int_2_s[i_old]]
            self.node_map_s_2_int[map_int_2_s[i_old]] = i_new
            map_old_2_new[i_old] = i_new
        # update traj_list
        self.traj_raw = []
        for traj in traj_list:
            self.traj_raw.append(np.array([map_old_2_new[i] for i in traj], dtype=dtype))

        self._init_raw_properties(dtype_lumped=dtype_lumped)

    def set_lumping_from_str(self, lumping_list, dtype=np.int16, calc_passage_time=False, letter=6, build_graph=True):
        """
        Set the lumping from lists of string. such as [("A", "B"), ("C", "D")].
        :param lumping_list: a list of lumping. Each lumping is a list of str. There should be no repeated states in lumping_list.
        :param dtype: data type of the trajectory. default is np.int16.
        :param calc_passage_time: whether to calculate the passage time. default is False.
        :param letter: number of letters in the state. default is 6. 6 : S0-S5, 5 : S0-S4, 4 : S1-S4
        Those variables will be updated:
        self.traj_node = []           # lumped trajectory
        self.node_map_int_2_s = {}    # what does each node mean (in self.traj_lumped). key:int, value: list of str
        self.node_map_s_2_int = {}    # Which node does a state belong to (in self.traj_lumped). key:str, value: int
        self.node_Counter = None      # Counter of nodes, key is int
        self.node_distribution = {}   # proportion of each node, key is int
        self.node_distribution_str = {}  # proportion of each node, key is tuple of string
        """
        # There should be no repeated states in lumping_list
        state_set = set([s for lumping in lumping_list for s in lumping])
        if len(state_set) != len([s for lumping in lumping_list for s in lumping]):
            raise ValueError("There are repeated states in lumping_list.")
        if dtype not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError("dtype should be np.int8, np.int16, np.int32, np.int64.")

        self.node_distribution_str = {}
        self.node_distribution = {}
        self.node_map_int_2_s = {}
        self.node_map_s_2_int = {}

        # update self.node_distribution_str
        for lumping in lumping_list:
            node_sum = 0
            for s in lumping:
                node_sum += self.state_Counter[s]
            self.node_distribution_str[tuple(lumping)] = node_sum / self.total_frame
        # update states that are not lumped
        lumping_list_flatten = [s for lumping in lumping_list for s in lumping]
        for s in self.state_Counter:
            if s not in lumping_list_flatten:
                self.node_distribution_str[(s,)] = self.state_distribution[s]

        # update self.node_distribution, node 0 has the largest proportion
        for i, (lumping, proportion) in enumerate(
                sorted(self.node_distribution_str.items(), key=lambda x: x[1], reverse=True)):
            self.node_distribution[i] = proportion
            self.node_map_int_2_s[i] = list(lumping)
            for s in lumping:
                self.node_map_s_2_int[s] = i
        # update self.traj_node
        self.traj_node = []
        for traj in self.traj_raw:
            self.traj_node.append(
                np.array([self.node_map_s_2_int[self.state_map_int_2_s[si]] for si in traj], dtype=dtype))
        # update self.node_Counter
        self.node_Counter = Counter([i for traj in self.traj_node for i in traj])

        # calculate properties
        self.calc_flux_matrix()
        self.calc_transition_probability_matrix()
        self.calc_net_flux_matrix()
        if build_graph:
            self.build_graph(letter)
        if calc_passage_time:
            self.calc_passage_time()
            if build_graph:
                self.update_graph_rate()
        else:
            self.passage_time_point_every_traj = None  # lumped traj
            self.passage_time_length_every_traj = None  # lumped traj

    def set_lag_step(self, lag_step=1):
        """
        Set the lag step.
        self.time_step will not be changed. If you calculate some properties, use self.time_step * self.lag_step.
        :param lag_step: int, the lag step.
        :return: None
        """
        # should be int
        if not isinstance(lag_step, int):
            raise ValueError("lag_step should be int.")
        # should be positive
        elif lag_step <= 0:
            raise ValueError("lag_step should be positive.")
        self.lag_step = lag_step

    def get_traj_in_string(self):
        """
        Get the trajectory in string format. To save memory, the state is represented by int in self.traj_raw.
        :return: a list of traj, the state is represented by string.
        """
        return [[self.state_map_int_2_s[i] for i in traj] for traj in self.traj_raw]

    def calc_flux_matrix(self):
        """
        Calculate the flux matrix. The flux matrix is a matrix of size (n, n), where n is the number of nodes.
        flux_matrix[i, j] is the flux from node i to node j.
        :return: flux_matrix, a np.array of size (n, n).
        """
        num_of_node = len(self.node_map_int_2_s)
        flux_matrix_alltraj = []
        for traj in self.traj_node:
            flux_matrix_tmp = np.zeros((num_of_node, num_of_node), dtype=np.int64)
            node_start = traj[:-self.lag_step]
            node_end = traj[self.lag_step:]
            for i in range(num_of_node):
                for j in range(num_of_node):
                    flux_matrix_tmp[i, j] = np.sum((node_start == i) & (node_end == j))
            flux_matrix_alltraj.append(flux_matrix_tmp)
        flux_matrix_alltraj = np.array(flux_matrix_alltraj)
        flux_matrix = np.sum(flux_matrix_alltraj, axis=0)
        self.flux_matrix_every_traj = flux_matrix_alltraj
        self.flux_matrix = flux_matrix
        return flux_matrix

    def calc_transition_probability_matrix(self):
        """
        Calculate the transition probability matrix.
        :return:
        """
        if self.flux_matrix is None or self.flux_matrix_every_traj is None:
            self.calc_flux_matrix()

        num_of_node = len(self.node_map_int_2_s)
        self.transition_probability_matrix = np.zeros((num_of_node, num_of_node))
        for i in range(num_of_node):
            flux_sum = np.sum(self.flux_matrix[i, :])
            if flux_sum == 0:
                self.transition_probability_matrix[i, :] = 0
            else:
                self.transition_probability_matrix[i, :] = self.flux_matrix[i, :] / flux_sum
        self.transition_probability_matrix_every_traj = np.zeros(
            (len(self.flux_matrix_every_traj), num_of_node, num_of_node))
        for traj_i, flux_i in enumerate(self.flux_matrix_every_traj):
            for i in range(num_of_node):
                flux_sum = np.sum(flux_i[i, :])
                if flux_sum == 0:
                    self.transition_probability_matrix_every_traj[traj_i, i, :] = 0
                else:
                    self.transition_probability_matrix_every_traj[traj_i, i, :] = flux_i[i, :] / flux_sum

        return self.transition_probability_matrix

    def calc_net_flux_matrix(self):
        """
        Calculate the net flux matrix.
        :return:
        """
        if self.flux_matrix is None or self.flux_matrix_every_traj is None:
            self.calc_flux_matrix()
        num_of_node = len(self.node_map_int_2_s)
        net_flux_matrix = np.zeros((num_of_node, num_of_node), dtype=np.int64)
        for i in range(num_of_node):
            for j in range(i + 1, num_of_node):
                net_flux_matrix[i, j] = self.flux_matrix[i, j] - self.flux_matrix[j, i]
                net_flux_matrix[j, i] = -net_flux_matrix[i, j]
        self.net_flux_matrix = net_flux_matrix
        self.net_flux_matrix_every_traj = np.zeros((len(self.flux_matrix_every_traj), num_of_node, num_of_node),
                                                   dtype=np.int64)
        for traj_i, flux_i in enumerate(self.flux_matrix_every_traj):
            for i in range(num_of_node):
                for j in range(i + 1, num_of_node):
                    self.net_flux_matrix_every_traj[traj_i, i, j] = flux_i[i, j] - flux_i[j, i]
                    self.net_flux_matrix_every_traj[traj_i, j, i] = -self.net_flux_matrix_every_traj[traj_i, i, j]

        return net_flux_matrix

    def calc_passage_time(self):
        """
        Calculate the passage time from every node to every node. This function does not respond to self.lag_step.
        :return: passage_time_length_every_traj
            A list of matrix. One matrix for each traj.
            matrix[i][j] is a list of passage time from node_i to node_j.
        """
        passage_time_len, passage_time_point = count_passage(self.traj_node, len(self.node_map_int_2_s))
        self.passage_time_length_every_traj = passage_time_len
        self.passage_time_point_every_traj = passage_time_point
        return passage_time_len

    def calc_passage_time_raw(self):
        """
        Calculate the passage time from every state to every state on raw traj. This function does not respond to self.lag_step.
        :return: passage_time_length_every_traj
            A list of matrix. One matrix for each traj.
            matrix[i][j] is a list of passage time from node_i to node_j.
        """
        passage_time_len, passage_time_point = count_passage(self.traj_raw, len(self.state_map_int_2_s),
                                                             print_progress=True)
        self.passage_time_length_every_traj_raw = passage_time_len
        return passage_time_len

    def get_time_step(self):
        """
        Get the time step between frames. This is not relevant to the lag step that you use to calculate properties.
        This is the time step between frames in the trajectory.
        :return: time_step, float.
        """
        return self.time_step

    def get_lag_step(self):
        """
        Get the lag step that you use to calculate properties.
        :return: lag_step, int.
        """
        return self.lag_step

    def get_lag_time(self):
        """
        Get the lag time that you use to calculate properties.
        time_step * lag_step
        time_step is the time between frames in the trajectory.
        lag_step is the lag step that you use to calculate properties.
        :return: lag_time, float.
        """
        return self.time_step * self.lag_step

    def get_mfpt(self, traj_type="lumped"):
        """
        Compute and return the mfpt from self.passage_time_length_every_traj
        :param traj_type: "lumped" or "raw"
        :return: mfpt, a np.array of size (n, n), mfpt from every node to every node.
                 mfpt_every_traj, mfpt for individual traj.
        """
        if traj_type == "raw":
            if self.passage_time_length_every_traj_raw is None:
                warnings.warn("self.passage_time_length_every_traj_raw is None. Calculating passage time.")
                self.calc_passage_time_raw()
            passage_time = self.passage_time_length_every_traj_raw
            node_num_length = len(self.state_map_int_2_s)
        elif traj_type == "lumped":
            if self.passage_time_length_every_traj is None:
                warnings.warn("self.passage_time_length_every_traj is None. Calculating passage time.")
                self.calc_passage_time()
            passage_time = self.passage_time_length_every_traj
            node_num_length = len(self.node_map_int_2_s)
        else:
            raise ValueError("traj_type should be 'lumped' or 'raw'.")

        mfpt_every_traj = []

        for passage_time_length in passage_time:
            mfpt_tmp = np.zeros((node_num_length, node_num_length))
            for i in range(node_num_length):
                for j in range(node_num_length):
                    if i != j:
                        if len(passage_time_length[i][j]) == 0:
                            mfpt_tmp[i, j] = np.nan
                        else:
                            mfpt_tmp[i, j] = np.mean(passage_time_length[i][j]) * self.time_step

            mfpt_every_traj.append(mfpt_tmp)
        mfpt = np.zeros((node_num_length, node_num_length))
        for i in range(node_num_length):
            for j in range(node_num_length):
                if i != j:
                    passage_list = [p_time for traj in passage_time for p_time in traj[i][j]]
                    if len(passage_list) == 0:
                        mfpt[i, j] = np.nan
                        # warnings.warn(f"no passage found for {self.node_map_int_2_s[i]} to {self.node_map_int_2_s[j]}.")
                    else:
                        mfpt[i, j] = np.mean(passage_list) * self.time_step

        return mfpt, mfpt_every_traj

    def get_rate_inverse_mfpt(self, traj_type="lumped"):
        """
        Compute and return the rate. This rate_ij is defined as : 1 / mfpt_ij
        :param traj_type: "lumped" or "raw"
        :return: rate, a np.array of size (n, n), rate from every node to every node.
                 rate_every_traj, a list of rate for individual traj.
        """
        mfpt, mfpt_every_traj = self.get_mfpt(traj_type)
        rate = np.zeros_like(mfpt)
        for i in range(mfpt.shape[0]):
            for j in range(mfpt.shape[1]):
                if i != j:
                    rate[i, j] = 1 / mfpt[i, j]
        rate_every_traj = []
        for mfpt_tmp in mfpt_every_traj:
            rate_tmp = np.zeros_like(mfpt_tmp)
            for i in range(mfpt.shape[0]):
                for j in range(mfpt.shape[1]):
                    if i != j:
                        rate_tmp[i, j] = 1 / mfpt_tmp[i, j]
            rate_every_traj.append(rate_tmp)
        return rate, rate_every_traj

    def get_rate_passage_time(self, traj_type="lumped"):
        """
        Compute and return the rate. This rate_ij is defined as : number of passage / (total frame of i * time_step)
        :param traj_type: "lumped" or "raw"
        :return: rate, a np.array of size (n, n), rate from every node to every node.
                 rate_every_traj, a list of rate for individual traj.
        """
        if traj_type == "raw":
            if self.passage_time_length_every_traj_raw is None:
                warnings.warn("self.passage_time_length_every_traj_raw is None. Calculating passage time.")
                self.calc_passage_time_raw()
            passage_time = self.passage_time_length_every_traj_raw
            node_num_length = len(self.state_map_int_2_s)
            # counter = self.state_Counter
            counter = Counter([i for traj in self.traj_raw for i in traj])
            traj_list = self.traj_raw
        elif traj_type == "lumped":
            if self.passage_time_length_every_traj is None:
                warnings.warn("self.passage_time_length_every_traj is None. Calculating passage time.")
                self.calc_passage_time()
            passage_time = self.passage_time_length_every_traj
            node_num_length = len(self.node_map_int_2_s)
            counter = self.node_Counter
            traj_list = self.traj_node
        else:
            raise ValueError("traj_type should be 'lumped' or 'raw'.")

        rate = np.zeros((node_num_length, node_num_length))
        for i in range(node_num_length):
            for j in range(node_num_length):
                if i != j:
                    passage_list = [p_time for traj in passage_time for p_time in traj[i][j]]
                    rate[i, j] = len(passage_list) / (counter[i] * self.time_step)
        rate_every_traj = []
        for passage_time_length, traj in zip(passage_time, traj_list):
            rate_tmp = np.zeros((node_num_length, node_num_length))
            for i in range(node_num_length):
                frame_num_i = np.sum(traj == i)
                time_i = frame_num_i * self.time_step
                for j in range(node_num_length):
                    if i != j:
                        if len(passage_time_length[i][j]) == 0:
                            rate_tmp[i, j] = 0
                        else:
                            rate_tmp[i, j] = len(passage_time_length[i][j]) / time_i
            rate_every_traj.append(rate_tmp)
        return rate, rate_every_traj

    def build_graph(self, letter=6):
        """
        :param letter: int, the number of binding sites to use.
            4: S1 - S4
            5: S0 - S4
            6: S0 - S5
        build a graph using networkx.
            if the net flux from i to j is larger than 0, there will be an edge from i to j.
            Each edge has attributes
                net_flux,
                flux_ij,
                flux_ji,
                distance_ij,
                distance_ji,
        :return:
        """
        # rate, _ = self.get_rate_passage_time()
        self.mechanism_G = Mechanism_Graph(
            self.net_flux_matrix, self.flux_matrix, self.net_flux_raw, self.flux_raw, self.state_distribution, self.node_distribution,
            self.node_map_int_2_s, self.node_map_s_2_int, self.state_map_int_2_s, self.state_map_s_2_int, self.raw_traj_df,
            letter)

    def update_graph_rate(self):
        """
        Update rate to the graph
        :return:
        """
        rate, _ = self.get_rate_passage_time()
        self.mechanism_G.set_graph_rate(rate)


def k_distance(A, B):
    """
    Compute the distance between two K-string. The length of A and B should be equal. They can be 4, 5, 6...
    """
    # test length equal
    if len(A) != len(B):
        raise ValueError("len(A) should be equal to len(B)")

    A_index = [index for index, char in enumerate(A) if char == 'K' or char == "C"]
    B_index = [index for index, char in enumerate(B) if char == 'K' or char == "C"]
    if len(A_index) == len(B_index):
        A_to_B = np.sum([j - i for i, j in zip(A_index, B_index)]) / (len(A) + 1)
        return -A_to_B, +A_to_B
    else:
        if len(A_index) > len(B_index):
            A_to_KB = np.sum([j - i for i, j in zip(A_index, [-1] + B_index)]) / (len(A) + 1)
            A_to_BK = np.sum([j - i for i, j in zip(A_index, B_index + [len(A)])]) / (len(A) + 1)
            if abs(A_to_KB) < abs(A_to_BK):
                return -A_to_KB, A_to_KB
            else:
                return -A_to_BK, A_to_BK
        else:
            KA_to_B = np.sum([j - i for i, j in zip([-1] + A_index, B_index)]) / (len(A) + 1)
            AK_to_B = np.sum([j - i for i, j in zip(A_index + [len(A)], B_index)]) / (len(A) + 1)
            if abs(KA_to_B) < abs(AK_to_B):
                return -KA_to_B, KA_to_B
            else:
                return -AK_to_B, AK_to_B


class Mechanism_Graph:
    """

    """

    def __init__(self, net_flux, flux, net_flux_raw, flux_raw, state_distribution, node_distribution,
                 node_map_int_2_s, node_map_s_2_int, state_map_int_2_s, state_map_s_2_int, raw_traj_df,
                 letter):
        """
        :param net_flux:
        :param flux:
        :param net_flux_raw:
        :param flux_raw:
        :param state_distribution:
        :param node_map_int_2_s:
        :param node_map_s_2_int:
        :param state_map_int_2_s:
        :param state_map_s_2_int:
        :param letter: int, the number of binding sites to use.
            4: S1 - S4
            5: S0 - S4
            6: S0 - S5
        """
        self.net_flux_raw = net_flux_raw
        self.flux_raw = flux_raw
        self.net_flux = net_flux
        self.flux = flux
        self.node_map_int_2_s = node_map_int_2_s
        self.node_map_s_2_int = node_map_s_2_int
        self.state_distribution = state_distribution
        self.node_distribution = node_distribution
        self.state_map_int_2_s = state_map_int_2_s
        self.state_map_s_2_int = state_map_s_2_int
        self.raw_traj_df = raw_traj_df
        self.letter = letter
        self.rate = None
        # build a DiGraph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(len(node_map_int_2_s)))
        for i in range(len(node_map_int_2_s)):
            for j in range(len(node_map_int_2_s)):
                if flux[i, j] > 0 and i != j and net_flux[i, j] >= 0:

                    # pick i,j that has the maximum flux
                    max_net_flux = 0
                    for si in node_map_int_2_s[i]:
                        for sj in node_map_int_2_s[j]:
                            if flux_raw[state_map_s_2_int[si], state_map_s_2_int[sj]] > max_net_flux:
                                max_net_flux = net_flux_raw[state_map_s_2_int[si], state_map_s_2_int[sj]]
                                max_si = si
                                max_sj = sj
                    if letter == 4:
                        distance_ij, distance_ji = k_distance(max_si[1:5], max_sj[1:5])
                    elif letter == 5:
                        distance_ij, distance_ji = k_distance(max_si[0:5], max_sj[0:5])
                    elif letter == 6:
                        distance_ij, distance_ji = k_distance(max_si[0:6], max_sj[0:6])
                    else:
                        raise ValueError("letter should be 4(S1~S4), 5(S0~S4) or 6(S0~S5).")
                    self.G.add_edge(i, j, net_flux=net_flux[i, j], flux_ij=flux[i, j], flux_ji=flux[j, i],
                                    distance_ij=distance_ij, distance_ji=distance_ji)

    def set_graph_rate(self, rate):
        self.rate = rate
        for i in range(len(self.node_map_int_2_s)):
            for j in range(len(self.node_map_int_2_s)):
                if self.G.has_edge(i, j):
                    self.G.edges[i, j]["rate_ij"] = rate[i, j]
                    self.G.edges[i, j]["rate_ji"] = rate[j, i]

    def get_node_proportion(self, node):
        """
        Get the proportion of each state in a node.
        :param node: int, the node.
        :return: sum proportion of every state.
        """
        return np.sum([self.state_distribution[s] for s in self.node_map_int_2_s[node]])

    def get_info_df(self):
        """
        Get all the edge information in a DataFrame.
        :return: pd.DataFrame
        """
        index_A = []
        index_B = []
        A_list = []
        B_list = []
        A_proportion = []
        B_proportion = []
        net_flux_AB = []
        flux_AB = []
        flux_BA = []
        rate_AB = []
        rate_BA = []
        dist_AB = []
        dist_BA = []
        for i, j, data in self.G.edges(data=True):
            index_A.append(i)
            index_B.append(j)
            A_list.append(self.node_map_int_2_s[i])
            B_list.append(self.node_map_int_2_s[j])
            A_proportion.append(self.get_node_proportion(i))
            B_proportion.append(self.get_node_proportion(j))
            net_flux_AB.append(data["net_flux"])
            flux_AB.append(data["flux_ij"])
            flux_BA.append(data["flux_ji"])
            rate_AB.append(data["rate_ij"])
            rate_BA.append(data["rate_ji"])
            dist_AB.append(data["distance_ij"])
            dist_BA.append(data["distance_ji"])
        df = pd.DataFrame({"index_A": index_A, "index_B": index_B,
                           "A": A_list, "B": B_list, "A_proportion": A_proportion, "B_proportion": B_proportion,
                           "net_flux_AB": net_flux_AB, "flux_AB": flux_AB, "flux_BA": flux_BA,
                           "rate_AB": rate_AB, "rate_BA": rate_BA,
                           "dist_AB": dist_AB, "dist_BA": dist_BA})
        df["rate_AB_x_rate_BA"] = df["rate_AB"] * df["rate_BA"]
        return df

    def test_lump(self, node_a, node_b):
        """
        if we lump 2 nodes, will it affect the permeation?
        :return: cycle_list, a list of cycle. The permeation in those cycle will lost.
        """
        # make sure there is an edge from node_a to node_b
        if self.G.has_edge(node_b, node_a):
            node_a, node_b = node_b, node_a
        elif node_a == node_b:
            return []
        elif not self.G.has_edge(node_a, node_b):
            return []

        # Is there a 3-node-cycle including node_a and node_b?
        cycle_list = []
        for node_c in self.G.nodes():
            if self.G.has_edge(node_b, node_c) and self.G.has_edge(node_c, node_a):
                # check if all the distance_ij are positive
                dist_ab = self.G.edges[node_a, node_b]["distance_ij"]
                dist_bc = self.G.edges[node_b, node_c]["distance_ij"]
                dist_ca = self.G.edges[node_c, node_a]["distance_ij"]
                if dist_ab > 0 and dist_bc > 0 and dist_ca > 0:
                    min_net_flux = min(self.G[node_a][node_b]["net_flux"],
                                       self.G[node_b][node_c]["net_flux"],
                                       self.G[node_c][node_a]["net_flux"])
                    if min_net_flux > 0:
                        cycle_list.append(((node_a, node_b, node_c), (dist_ab, dist_bc, dist_ca), min_net_flux))
        cycle_list.sort(key=lambda x: x[-1], reverse=True)
        return cycle_list

    def guess_init_position(self, str_2_xy=None):
        """
        :param str_2_xy: function, convert a string to a position, if not given, ??
        :return: position, a dictionary from node index to position
        """
        if str_2_xy is None:
            def str_2_xy(string):
                weight_dict = {"K": 1,
                               "W": 0.2,
                               "C": 1.1,
                               "0": 0, }
                center_mass = 0  # y
                total_mass = 0
                for i, site in enumerate(string):
                    center_mass += weight_dict[site] * i
                    total_mass += weight_dict[site]
                center_mass /= total_mass
                x = string[:5].count("K") + string[:5].count("C")
                return x, center_mass
        position = {}
        for i in self.G.nodes:
            p_i = [str_2_xy(si) for si in self.node_map_int_2_s[i]]
            p_i = np.mean(p_i, axis=0)
            position[i] = p_i
        return position

    def set_node_label(self, label_function=None, **kwargs):
        """
        set the label of nodes.
        if you do it twice, all of the labels will be overwritten.
        :param label_function: a function to convert node (a list of 6-letter code) to label.
            if label_function is None, use the default function,
            x will be the number of K or C, and y will be the center of mass for S0-S4.
        :return:
        """
        if label_function is None:
            def label_function(i, add_index=True):
                node = self.node_map_int_2_s[i]
                if not add_index:
                    string = ""
                elif len(node) <= 1:
                    string = f"{i}:"
                else:
                    string = f"{i} : \n"

                for s in node[:-1]:
                    string += s
                    string += "\n"
                string += node[-1]
                return string
        label_dict = {i: label_function(i, **kwargs) for i in self.G.nodes}
        nx.set_node_attributes(self.G, label_dict, "label")

    def draw_dirty(self, ax, node_list=None, node_size_range=((0.01, 0.3), (30, 900)), edge_width=None,
                   net_flux_cutoff=10,
                   node_alpha=0.7, edge_alpha=0.5, add_index=True,
                   spring_iterations=5, spring_k=10, pos=None,
                   label_bbox=None):
        """
        """
        if label_bbox is None:
            label_bbox = {"boxstyle": "round", "ec": (1.0, 1.0, 1.0, 0), "fc": (1.0, 1.0, 1.0, 0.5)}

        # 1 select node based on net_flux cutoff
        if node_list is None:
            node_list = []
            for i in self.G.nodes:
                if np.any(self.net_flux[i, :] > net_flux_cutoff) or np.any(self.net_flux[:, i] > net_flux_cutoff):
                    node_list.append(i)
        sub_G = self.G.subgraph(node_list)
        # 2 spring_layout
        # get a subgraph and spring_layout
        if pos is None:
            pos_init_all = self.guess_init_position()
            pos = nx.spring_layout(sub_G, iterations=spring_iterations, k=spring_k,
                                   pos={i: pos_init_all[i] for i in node_list})
        # 3 node size based on distribution
        self.set_node_label(add_index=add_index)
        ((min_p, max_p), (min_size, max_size)) = node_size_range
        size_dict = {i: min_size + (max_size - min_size) * (self.node_distribution[i] - min_p) / (max_p - min_p) for i
                     in self.G.nodes}
        # color
        color_list = []
        for node in sub_G:
            K_number = self.node_map_int_2_s[node][0][:5].count("K") + self.node_map_int_2_s[node][0][:5].count("C")
            K_number = (K_number + 8) % 10
            color_list.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][K_number])
        nx.draw_networkx_nodes(sub_G, ax=ax, pos=pos,
                               node_size=[size_dict[i] for i in sub_G.nodes], node_color=color_list, alpha=node_alpha)
        nx.draw_networkx_labels(sub_G, ax=ax, pos=pos, labels=nx.get_node_attributes(sub_G, "label"),
                                font_family='monospace')

        # 4 draw edge based on net_flux cutoff, rescale edge_width
        edge_list = []
        edge_label = {}
        edge_rate = []
        if self.rate is None:
            for u, v, d in sub_G.edges(data=True):
                if d["net_flux"] > net_flux_cutoff:
                    edge_list.append((u, v))
                    edge_label[(u, v)] = f"{d['net_flux']:d}"
                    edge_rate.append(d["net_flux"])
        else:
            for u, v, d in sub_G.edges(data=True):
                if d["net_flux"] > net_flux_cutoff:
                    edge_list.append((u, v))
                    edge_label[(u, v)] = f"{d['net_flux']:d}"
                    edge_rate.append(d["rate_ij"])
        # rescale edge_width
        if edge_width is None:
            edge_width = ((min(edge_rate), max(edge_rate)), (0.05, 10))
        ((min_rate, max_rate), (min_width, max_width)) = edge_width
        width_list = [min_width + (max_width - min_width) * (rate - min_rate) / (max_rate - min_rate) for rate in
                      edge_rate]
        nx.draw_networkx_edges(sub_G, ax=ax, pos=pos, edgelist=edge_list, width=width_list, alpha=edge_alpha,
                               connectionstyle='arc3,rad=0.05', )
        nx.draw_networkx_edge_labels(sub_G, ax=ax, pos=pos, edge_labels=edge_label, bbox=label_bbox)

        # return
        # node_list
        # 2 print rate distribution
        # 3 pos (for further refine)
        return node_list, pos, sub_G, edge_width

    def plot_grid(self, ax):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xticks(np.arange(round(xmin, 1), round(xmax, 1), 0.1))
        ax.set_yticks(np.arange(round(ymin, 1), round(ymax, 1), 0.1))
        ax.grid()



