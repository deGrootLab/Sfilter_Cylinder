import warnings
from collections import Counter
import numpy as np
import pandas as pd
from .output_wrapper import read_k_cylinder
from collections import Counter


def count_passage(traj_list, num_of_node):
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
    for traj in traj_list:
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

    return passage_time_length_every_traj, passage_time_point_every_traj

class Sf_model:
    """
    This is a class to analyse the mechanism of a selectivity filter.
    """
    def __init__(self, file_list=None, start=0, end=None, step=1, method="K_priority", lag_step=1, traj_dtype=np.int8):
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
        self.time_step = 0    # time step between frames
        self.total_frame = 0  # total number of frames
        self.traj_raw = []    # raw trajectory, a list of np.array. not lumped
        self.state_map_int_2_s = {}   # map from int to state
        self.state_map_s_2_int = {}   # map from state to int
        self.state_Counter = None     # Counter of states(str)
        self.state_distribution = {}  # proportion of each state(str)

        # variables for lumped trajectory
        self.traj_node = []           # lumped trajectory
        self.node_map_int_2_s = {}    # what does each node mean (in self.traj_lumped). key:int, value: list of str
        self.node_map_s_2_int = {}    # Which node does a state belong to (in self.traj_lumped). key:str, value: int
        self.node_Counter = None      # Counter of nodes, key is int
        self.node_distribution = {}   # proportion of each node, key is int
        self.node_distribution_str = {}  # proportion of each node, key is tuple of string

        # variables for properties
        self.lag_step = lag_step  # lag step for computing properties
        self.flux_matrix = None
        self.flux_matrix_every_traj = None
        self.transition_probability_matrix = None
        self.transition_probability_matrix_every_traj = None
        self.net_flux_matrix = None
        self.net_flux_matrix_every_traj = None
        self.passage_time_point_every_traj = None   # lumped traj
        self.passage_time_length_every_traj = None  # lumped traj
        self.passage_time_length_every_traj_raw = None # raw traj

        # initialization finished

        # read file(s)
        if self.file_list is not None:
            time_step_list = []
            traj_tmp_list = []
            for traj, meta_data, K_occupency, W_occupency in [read_k_cylinder(file, method) for file in file_list]:
                time_step_list.append(meta_data["time_step"])
                traj_tmp_list.append(traj[start:end:step])
            # check if time step (float) is almost the same
            if not np.allclose(time_step_list, time_step_list[0]):
                raise ValueError("The time step between files are not the same.", str(time_step_list))

            self.set_traj_from_str(traj_tmp_list, time_step_list[0] * step, dtype=traj_dtype, dtype_lumped=traj_dtype)
            self.calc_passage_time_raw()

    def set_traj_from_str(self, traj_list, time_step, dtype=np.int8, dtype_lumped=np.int8):
        """
        Set the trajectory from lists of string.
        :param traj_list: a list of traj. Each traj is a sequence of str.
        :param time_step: time step between frames.
        :param dtype: data type of the trajectory. default is np.int8.
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

        self.set_lumping_from_str([], dtype_lumped)  # At the end of this function, properties will be calculated.
        # including
        # self.flux_matrix,
        # self.flux_matrix_every_traj,
        # self.transition_probability_matrix,
        # self.transition_probability_matrix_every_traj,
        # self.net_flux_matrix,
        # self.net_flux_matrix_every_traj,
        # self.passage_time_point_every_traj,
        # self.passage_time_length_every_traj

    def set_traj_from_int(self, traj_list, time_step):
        """
        Set the trajectory from lists of int.
        :param traj_list: a list of traj. Each traj is a sequence of int.
        :param time_step: time step between frames.
        :return: None
        """
        pass

    def set_lumping_from_str(self, lumping_list, dtype=np.int8):
        """
        Set the lumping from lists of string. such as [("A", "B"), ("C", "D")].
        :param lumping_list: a list of lumping. Each lumping is a list of str. There should be no repeated states in lumping_list.
        :param dtype: data type of the trajectory. default is np.int8.
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
        for i, (lumping, proportion) in enumerate(sorted(self.node_distribution_str.items(), key=lambda x: x[1], reverse=True)):
            self.node_distribution[i] = proportion
            self.node_map_int_2_s[i] = list(lumping)
            for s in lumping:
                self.node_map_s_2_int[s] = i
        # update self.traj_node
        self.traj_node = []
        for traj in self.traj_raw:
            self.traj_node.append(np.array([self.node_map_s_2_int[self.state_map_int_2_s[si]] for si in traj], dtype=dtype))
        # update self.node_Counter
        self.node_Counter = Counter([i for traj in self.traj_node for i in traj])

        # calculate properties
        self.calc_flux_matrix()
        self.calc_transition_probability_matrix()
        self.calc_net_flux_matrix()
        self.calc_passage_time()

    def set_lumping_from_int(self, lumping_list, dtype=np.int8):
        """
        Set the lumping from lists of int. such as [[0, 1], [2, 3]].
        :param lumping_list: a list of lumping. Each lumping is a list of int. There should be no repeated states in lumping_list.
        :param dtype: data type of the trajectory. default is np.int8.
        Those variables will be updated:
        self.traj_node = []           # lumped trajectory
        self.node_map_int_2_s = {}    # what does each node mean (in self.traj_lumped). key:int, value: list of str
        self.node_map_s_2_int = {}    # Which node does a state belong to (in self.traj_lumped). key:str, value: int
        self.node_Counter = None      # Counter of nodes, key is int
        self.node_distribution = {}   # proportion of each node, key is int
        self.node_distribution_str = {}  # proportion of each node, key is tuple of string
        :return: None
        """
        lumping_list_str = []
        for lumping in lumping_list:
            lumping_list_str.append([self.state_map_int_2_s[i] for i in lumping])
        self.set_lumping_from_str(lumping_list_str, dtype=dtype)

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
            flux_matrix_tmp = np.zeros((num_of_node, num_of_node))
            node_start = traj[             :-self.lag_step]
            node_end   = traj[self.lag_step:              ]
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
        self.transition_probability_matrix_every_traj = np.zeros((len(self.flux_matrix_every_traj), num_of_node, num_of_node))
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
        net_flux_matrix = np.zeros((num_of_node, num_of_node))
        for i in range(num_of_node):
            for j in range(i+1, num_of_node):
                net_flux_matrix[i, j] = self.flux_matrix[i, j] - self.flux_matrix[j, i]
                net_flux_matrix[j, i] = -net_flux_matrix[i, j]
        self.net_flux_matrix = net_flux_matrix
        self.net_flux_matrix_every_traj = np.zeros((len(self.flux_matrix_every_traj), num_of_node, num_of_node))
        for traj_i, flux_i in enumerate(self.flux_matrix_every_traj):
            for i in range(num_of_node):
                for j in range(i+1, num_of_node):
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
        passage_time_len, passage_time_point = count_passage(self.traj_raw, len(self.state_map_int_2_s))
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
        compute and return the mfpt from self.passage_time_length_every_traj
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

        mfpt_every_traj = []

        for passage_time_length in passage_time:
            mfpt_tmp = np.zeros((node_num_length, node_num_length))
            for i in range(node_num_length):
                for j in range(node_num_length):
                    if i != j:
                        if len(passage_time_length[i][j]) == 0 :
                            mfpt_tmp[i, j] = np.nan
                        else:
                            mfpt_tmp[i, j] = np.mean(passage_time_length[i][j]) * self.time_step

            mfpt_every_traj.append(mfpt_tmp)
        mfpt = np.zeros((node_num_length, node_num_length))
        for i in range(node_num_length):
            for j in range(node_num_length):
                if i != j:
                    passage_list = [p_time for traj in self.passage_time_length_every_traj for p_time in traj[i][j] ]
                    if len(passage_list) == 0:
                        mfpt[i, j] = np.nan
                        # warnings.warn(f"no passage found for {self.node_map_int_2_s[i]} to {self.node_map_int_2_s[j]}.")
                    else:
                        mfpt[i, j] = np.mean(passage_list) * self.time_step

        return mfpt, mfpt_every_traj

    def get_rate_inverse_mfpt(self):
        """
        compute and return the rate. This rate_ij is defined as : 1 / mfpt_ij
        :return: rate, a np.array of size (n, n), rate from every node to every node.
                 rate_every_traj, a list of rate for individual traj.
        """
        mfpt, mfpt_every_traj = self.get_mfpt()
        rate = np.zeros_like(mfpt)
        for i in range(len(self.node_map_int_2_s)):
            for j in range(len(self.node_map_int_2_s)):
                if i != j:
                    rate[i, j] = 1 / mfpt[i, j]
        rate_every_traj = []
        for mfpt_tmp in mfpt_every_traj:
            rate_tmp = np.zeros_like(mfpt_tmp)
            for i in range(len(self.node_map_int_2_s)):
                for j in range(len(self.node_map_int_2_s)):
                    if i != j:
                        rate_tmp[i, j] = 1 / mfpt_tmp[i, j]
            rate_every_traj.append(rate_tmp)
        return rate, rate_every_traj

    def get_rate_passage_time(self):
        """
        compute and return the rate. This rate_ij is defined as : number of passage / (total frame of i * time_step)
        :return: rate, a np.array of size (n, n), rate from every node to every node.
                 rate_every_traj, a list of rate for individual traj.
        """
        if self.passage_time_length_every_traj is None:
            warnings.warn("self.passage_time_length_every_traj is None. Calculating passage time.")
            self.calc_passage_time()
        rate = np.zeros((len(self.node_map_int_2_s), len(self.node_map_int_2_s)))
        for i in range(len(self.node_map_int_2_s)):
            for j in range(len(self.node_map_int_2_s)):
                if i != j:
                    passage_list = [p_time for traj in self.passage_time_length_every_traj for p_time in traj[i][j] ]
                    rate[i, j] = len(passage_list) / (self.node_Counter[i] * self.time_step)
        rate_every_traj = []
        for passage_time_length, traj in zip(self.passage_time_length_every_traj, self.traj_node):
            rate_tmp = np.zeros((len(self.node_map_int_2_s), len(self.node_map_int_2_s)))
            for i in range(len(self.node_map_int_2_s)):
                frame_num_i = np.sum(traj == i)
                time_i = frame_num_i * self.time_step
                for j in range(len(self.node_map_int_2_s)):
                    if i != j:
                        rate_tmp[i, j] = len(passage_time_length[i][j]) / time_i
            rate_every_traj.append(rate_tmp)
        return rate, rate_every_traj

