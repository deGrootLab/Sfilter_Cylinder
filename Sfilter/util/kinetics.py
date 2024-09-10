import copy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from deeptime.markov.tools.flux import flux_matrix

from .output_wrapper import read_k_cylinder
from collections import Counter
import gc
import networkx as nx
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor
from .passage_cycle_correct import Passage_cycle_correct


# inorder to run count_passage in parallel, we need to define a function count_passage_single_traj that process a single traj.
def count_passage_single_traj(traj, num_of_node):
    """
    Count the passage on a single trajectory.
    :param traj: a np.array of int.
    :param num_of_node: number of nodes. Assume the nodes are 0, 1, 2, ..., num_of_node-1.
    :return: (passage_time_length, passage_time_point)
        passage_time_length, a matrix. passage_time_length[i][j] is a np.array of passage time(unit in step)
        passage_time_point, a matrix. passage_time_point[i][j] is a np.array of passage finishing time (unit in step)
    """
    passage_time_length = []
    passage_time_point = []
    for i in range(num_of_node):
        passage_time_length.append([[] for _ in range(num_of_node)])  # length of each passage
        passage_time_point.append([[] for _ in range(num_of_node)])   # finishing time of each passage
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

    return passage_time_length, passage_time_point

def count_passage(traj_list, num_of_node, num_workers=None):
    """
    Count the passage on a list of trajectory.
    :param traj_list: A list of trajectory. Each trajectory is a list(np.array) of int.
    :param num_of_node: number of nodes. Assume the nodes are 0, 1, 2, ..., num_of_node-1.
    :return:
        passage_time_length_alltraj
            a list of matrix. One matrix for each traj. matrix[i][j] is a list of passage time(unit in step)
            from node_i to node_j.
        passage_time_point_alltraj
            a list of matrix. One matrix for each traj. matrix[i][j] is a list of time point (unit in step)
            when the passage from node_i to node_j finished .
    """
    passage_time_length_every_traj = []
    passage_time_point_every_traj = []

    # if num_workers is None, it will use the number of logical CPUs
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for traj in traj_list:
            future = executor.submit(count_passage_single_traj, traj, num_of_node)
            futures.append(future)

        for future in futures:
            passage_time_length, passage_time_point = future.result()
            passage_time_length_every_traj.append(passage_time_length)
            passage_time_point_every_traj.append(passage_time_point)

    return passage_time_length_every_traj, passage_time_point_every_traj


class Sf_model:
    """
    This is a class to analyse the mechanism of a selectivity filter.
    """

    def __init__(self, file_list=None, start=0, end=None, step=1, method="K_priority", lag_step=1,
                 traj_dtype=np.int16, print_time=False):
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
            "K_priority_S14", the same as "K_priority", but only the S1-S4 are considered.
        :param lag_step: lag step for calculating properties (transition matrix), default is 1.
        :param traj_dtype: data type of the trajectory. default is np.int16. np.int8 is not safe.
            There are 6 binding sites S0-S5.
            If one uses K/W/0, there are 3^6=729 possibilities.
            If one uses K/W/C/0, there are 4^6=4096 possibilities.
            int8  (-128 to 127) is not enough.
            int16 (-32768 to 32767) is enough.
        """

        t0 = time.time()
        # check arguments and initialize variables
        if file_list is None:
            self.file_list = None
        elif isinstance(file_list, str):
            self.file_list = [file_list]
        elif isinstance(file_list, list):
            self.file_list = [Path(f) for f in file_list]
        if traj_dtype == np.int8:
            warnings.warn("np.int8 (-128 to 127) is not safe for this application. Please use np.int16 (-32768 to 32767).")
        elif traj_dtype not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError("traj_dtype should be np.int16, np.int32, np.int64.")

        # variables for raw trajectory (state)
        self.time_step = 0  # time step between frames
        self.total_frame = 0  # total number of frames
        self.traj_raw_alltraj = []  # raw trajectory, a list of np.array. not lumped
        self.state_map_int_2_s = {}  # map from int to state
        self.state_map_s_2_int = {}  # map from state to int
        self.state_Counter = None  # Counter of states(str)
        self.state_distribution = {}  # proportion of each state(str)
        self.state_distribution_alltraj = {}  # proportion of each state(str) in each traj

        # variables for lumped trajectory (node)
        self.traj_node = []  # lumped trajectory
        self.node_map_int_2_s = {}  # what does each node mean (in self.traj_lumped). key:int, value: list of str
        self.node_map_s_2_int = {}  # Which node does a state belong to (in self.traj_lumped). key:str, value: int
        self.node_Counter = None  # Counter of nodes, key is int
        self.node_distribution = {}  # proportion of each node, key is int
        self.node_distribution_str = {}  # proportion of each node, key is tuple of string

        # variables for properties
        self.lag_step = lag_step  # lag step for computing properties
        # flux
        self.flux_matrix,     self.flux_matrix_alltraj = None, None # lumped traj (nodes)
        self.net_flux_matrix, self.net_flux_matrix_alltraj = None, None
        self.flux_matrix_raw,     self.flux_matrix_raw_alltraj = None, None # raw traj (states)
        self.net_flux_matrix_raw, self.net_flux_matrix_raw_alltraj = None, None

        self.transition_probability_matrix = None # lumped traj (nodes)
        self.transition_probability_matrix_alltraj = None

        self.passage_time_point_alltraj = None  # lumped traj
        self.passage_time_length_alltraj = None  # lumped traj
        self.passage_time_point_alltraj_raw = None  # raw traj
        self.passage_time_length_alltraj_raw = None  # raw traj
        self.jump_array_alltraj = []
        self.rate_raw = None

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
                # if the cached jump file exists, use it. otherwise used get_jump. It should be in the same folder as the file.
                if file.with_suffix(".jump_np_array.npy").exists():
                    traj, meta_data, K_occ, W_occ = read_k_cylinder(file, method, get_occu=False)
                    jump_array = np.load(file.with_suffix(".jump_np_array.npy"))
                    if len(jump_array) != len(traj):
                        raise ValueError(f"The cached jump file is broken. Please delete it and rerun. {file.with_suffix('.jump_np_array.npy')}")
                    self.jump_array_alltraj.append(jump_array)
                else :
                    traj, meta_data, K_occ, W_occ, jump_array = read_k_cylinder(file, method, get_occu=True, get_jump=True)
                    self.jump_array_alltraj.append(jump_array)
                    np.save(file.with_suffix(".jump_np_array.npy"), jump_array)
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
                gc.collect()
            # check if time step (float) is almost the same
            if not np.allclose(time_step_list, time_step_list[0]):
                raise ValueError("The time step between files are not the same.", str(time_step_list))
            t2 = time.time()
            self.set_traj_from_int(traj_tmp_list, time_step_list[0] * step, map_int_2_s, dtype=traj_dtype,
                                   init_raw_properties=False)
            t3 = time.time()
            self._init_raw_properties(dtype_lumped=traj_dtype)
            self.passage_cycle_correct = Passage_cycle_correct(self.traj_raw_alltraj,
                                                               self.passage_time_length_alltraj_raw,
                                                               self.passage_time_point_alltraj_raw,
                                                               self.jump_array_alltraj, self.time_step)
            # this is a class that saves the rate_ij (mfpt or passage) with cycle correction.
            # User should not use this directly.
            t4 = time.time()
            if print_time:
                print("load files                (s):", t2 - t0)
                print("set traj and lumped traj  (s):", t3 - t2)
                print("compute raw properties    (s):", t4 - t3)

    def set_traj_from_str(self, traj_list, time_step, dtype=np.int16, dtype_lumped=np.int16, init_raw_properties=True):
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


        self.traj_raw_alltraj = []
        for traj in traj_list:
            self.traj_raw_alltraj.append(np.array([self.state_map_s_2_int[s] for s in traj], dtype=dtype))

        if init_raw_properties:
            self._init_raw_properties(dtype_lumped=dtype_lumped)

    def set_traj_from_int(self, traj_list, time_step, map_int_2_s, dtype=np.int16, init_raw_properties=True):
        """
        Set the trajectory from lists of int.
        :param traj_list: a list of np.array(). Each np.array() is a sequence of int. State 0 doesn't need to be the
            most common state. We will re-arrange the internal state index.
        :param time_step: time step between frames. Unit is ps.
        :param map_int_2_s: map from int to state (str).
        :param dtype: data type of the trajectory. default is np.int16.
        :return: None
        """
        if dtype not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError("dtype should be np.int16, np.int32, np.int64.")

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
        self.traj_raw_alltraj = []
        for traj in traj_list:
            self.traj_raw_alltraj.append(np.array([map_old_2_new[i] for i in traj], dtype=dtype))

        if init_raw_properties:
            self._init_raw_properties(dtype_lumped=dtype)

    def _init_raw_properties(self, dtype_lumped=np.int16):
        """
        Initialize the basic properties of the raw trajectory.
        This should only be used in the initialization, when the raw trajectory is set.
        :param dtype_lumped: default is np.int16 (-32768 to 32767).
        The following properties will be calculated:
        self.flux_matrix_raw, self.flux_matrix_raw_alltraj
            self.flux_matrix_raw[i,j] is the number of flux from node i to node j.
            self.flux_matrix_raw_alltraj[rep][i,j] is the number of flux from node i to node j in the rep-th traj.

        self.net_flux_matrix_raw, self.net_flux_matrix_raw_alltraj
            self.net_flux_matrix_raw[i,j] is the net flux from state i to state j.
            self.net_flux_matrix_raw_alltraj[rep][i,j] is the net flux from state i to state j in the rep-th traj.

        self.passage_time_length_alltraj_raw : A list of 2D list
            self.passage_time_length_alltraj_raw[rep][i][j] is the passage time (unit in step) from state i to state j in
            the rep-th traj.

        self.passage_time_point_alltraj_raw : A list of 2D list
            self.passage_time_point_alltraj_raw[rep][i][j] is the time point (unit in step) when the passage from
            state i to state j finished in the rep-th traj.

        self.state_distribution_alltraj : A dictionary
            self.state_distribution_alltraj[s] is a list of proportion of state s in each traj.
        :return: None
        """
        self.calc_flux_raw()
        # self.flux_matrix_raw, self.flux_matrix_raw_alltraj
        # self.net_flux_matrix_raw, self.net_flux_matrix_raw_alltraj will be updated
        self.set_lumping_from_str([], dtype_lumped, calc_passage_time=False)  # At the end of this function, properties will be calculated.
        self.calc_passage_time()
        self.passage_time_length_alltraj_raw = copy.deepcopy(self.passage_time_length_alltraj)
        self.passage_time_point_alltraj_raw = copy.deepcopy(self.passage_time_point_alltraj)
        self.rate_raw, _ = self.get_rate_passage_time(traj_type="raw")

        self.state_distribution_alltraj = {s: np.zeros(len(self.traj_raw_alltraj)) for s, count in
                                           self.state_Counter.items()}
        for rep, traj in enumerate(self.traj_raw_alltraj):
            count = Counter(traj)
            for s_index, frame in count.items():
                s = self.state_map_int_2_s[s_index]
                self.state_distribution_alltraj[s][rep] = frame / len(traj)



    def get_state_index(self, state):
        """
        given a state, return the index of the state.
        :param state: str
        :return: int
        """
        return self.state_map_s_2_int[state]

    def get_flux_AB(self, state_A, state_B):
        """
        Get the flux from state_A to state_B.
        :param state_A: str or int
        :param state_B: str or int
        :return: flux, flux_alltraj
            flux : int, the flux from state_A to state_B.
            flux_alltraj : a list of int, the flux from state_A to state_B in each replica.
        """
        if isinstance(state_A, str):
            state_A = self.get_state_index(state_A)
        if isinstance(state_B, str):
            state_B = self.get_state_index(state_B)
        flux = self.flux_matrix_raw[state_A, state_B]
        flux_alltraj = [f_rep[state_A, state_B]    for f_rep in self.flux_matrix_raw_alltraj]
        return flux, flux_alltraj

    def get_concentration(self, state):
        """
        Get the concentration(population) of a state.
        :param state: str or int
        :return: c, c_alltraj
            c : float, the concentration of the state.
            c_alltraj : a list of float, the concentration of the state in each replica.
        """
        if isinstance(state, int):
            state = self.state_map_int_2_s[state]
        c = self.state_distribution[state]
        c_alltraj = self.state_distribution_alltraj[state]
        return c, c_alltraj

    def get_passage_AB(self, state_A, state_B):
        """
        Get the passage info from state_A to state_B.
        passage_A_to_B can correspond to different jumps. WKK0KW to KK0KKW can be -2 or +5. This function will return all
        possible jumps.
        :param state_A: str or int
        :param state_B: str or int
        :return:
            A dictionary
                key: number of jumps
                value: (length, start, end)
                    length[rep][k] is the k-th passage time length in replica rep.
                    start[rep][k]  is the k-th passage time starting point in replica rep.
                    end[rep][k]    is the k-th passage time ending point in replica rep.
        """
        if isinstance(state_A, str):
            state_A = self.get_state_index(state_A)
        if isinstance(state_B, str):
            state_B = self.get_state_index(state_B)
        return self.passage_cycle_correct.get_passage_ij(state_A, state_B)

    def get_passage_AB_shortest(self, state_A, state_B):
        """
        Get the passage info from state_A to state_B.
        passage_A_to_B can correspond to different jumps. WKK0KW to KK0KKW can be -2 or +5. This function will only
        return the passage info for the shortest passage. For example, -2 for WKK0KW to KK0KKW.
        :param state_A: str or int
        :param state_B: str or int
        :return:
            (length, start, end)
                length[rep][k] is the k-th passage time length in replica rep.
                start[rep][k]  is the k-th passage time starting point in replica rep.
                end[rep][k]    is the k-th passage time ending point in replica rep
        """
        if isinstance(state_A, str):
            state_A = self.get_state_index(state_A)
        if isinstance(state_B, str):
            state_B = self.get_state_index(state_B)
        passage_all = self.passage_cycle_correct.get_passage_ij(state_A, state_B)
        if passage_all == {}:
            warnings.warn(f"No passage from {state_A} to {state_B}.")
            rep_n = len(self.traj_raw_alltraj)
            return ([[]]*rep_n, [[]]*rep_n, [[]]*rep_n), None
        else :
            min_pass_jump = min(list(passage_all.keys()), key=abs)
            return passage_all[min_pass_jump], min_pass_jump


    def get_mfpt_AB_shortest_passage(self, state_A, state_B):
        """
        Get the passage info from state_A to state_B.
        passage_A_to_B can correspond to different jumps. WKK0KW to KK0KKW can be -2 or +5. This function will only
        return the passage mfpt for the shortest passage. For example, -2 for WKK0KW to KK0KKW.
        :param state_A: str or int
        :param state_B: str or int
        :return: mfpt, mfpt_alltraj
            mfpt : float, the mean first passage time from state_A to state_B (average over all replicas).
            mfpt_alltraj : a list of float, the mean first passage time from state_A to state_B in each replica.
        """
        (length, start, end), jump = self.get_passage_AB_shortest(state_A, state_B)
        if jump is None:
            return np.inf, [np.inf]*len(self.traj_raw_alltraj), None
        mfpt = np.concatenate(length).mean() * self.time_step
        mfpt_alltraj =[np.mean(l)* self.time_step for l in length]
        return mfpt, mfpt_alltraj, jump


    def set_lumping_from_str(self, lumping_list, dtype=np.int16, calc_passage_time=False, letter=6):
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
        # for traj in self.traj_raw_alltraj: # this is slow
        #     self.traj_node.append(
        #         np.array([self.node_map_s_2_int[self.state_map_int_2_s[si]] for si in traj], dtype=dtype))
        node_map_vectorized = np.vectorize(lambda si: self.node_map_s_2_int[self.state_map_int_2_s[si]])
        self.traj_node = [node_map_vectorized(traj).astype(dtype) for traj in self.traj_raw_alltraj]
        # update self.node_Counter
        self.node_Counter = Counter([])
        for traj in self.traj_node:
            self.node_Counter.update(traj)

        # calculate properties
        self.calc_flux_matrix()
        self.calc_transition_probability_matrix()
        if calc_passage_time:
            self.calc_passage_time()
        else:
            self.passage_time_point_alltraj = None  # lumped traj
            self.passage_time_length_alltraj = None  # lumped traj



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
        Get the trajectory in string format. To save memory, the state is represented by int in self.traj_raw_alltraj.
        :return: a list of traj, the state is represented by string.
        """
        return [[self.state_map_int_2_s[i] for i in traj] for traj in self.traj_raw_alltraj]

    def calc_flux_raw(self):
        """
        Calculate the flux matrix on the raw trajectory.
        flux_matrix_raw[i, j] is the flux from state i to state j.
        :return: flux_matrix_raw, a np.array of size (n, n).
        """
        # flux
        num_of_state = len(self.state_map_int_2_s)
        flux_matrix_raw_alltraj = [np.zeros((num_of_state, num_of_state), dtype=np.int64) for i in range(len(self.traj_raw_alltraj))]
        for i, traj in enumerate(self.traj_raw_alltraj):
            state_start = traj[:-self.lag_step]
            state_end = traj[self.lag_step:]
            unique, counts = np.unique(np.vstack((state_start, state_end)).T, axis=0, return_counts=True)
            flux_matrix_raw_alltraj[i][unique[:, 0], unique[:, 1]] += counts
        flux_matrix_raw = np.sum(flux_matrix_raw_alltraj, axis=0)
        self.flux_matrix_raw, self.flux_matrix_raw_alltraj = (flux_matrix_raw,flux_matrix_raw_alltraj)

        # net flux
        self.net_flux_matrix_raw_alltraj = [np.zeros((num_of_state, num_of_state), dtype=np.int64) for i in range(len(self.traj_raw_alltraj))]
        for rep, traj in enumerate(self.traj_raw_alltraj):
            for s_i in range(num_of_state):
                for s_j in range(num_of_state):
                    self.net_flux_matrix_raw_alltraj[rep][s_i, s_j] = flux_matrix_raw_alltraj[rep][s_i, s_j] - flux_matrix_raw_alltraj[rep][s_i, s_j]
        self.net_flux_matrix_raw = np.sum(self.net_flux_matrix_raw_alltraj, axis=0)
        return self.flux_matrix_raw, self.flux_matrix_raw_alltraj


    def calc_flux_matrix(self):
        """
        Calculate the flux matrix.
        Based on what states are lumped, calculate the flux matrix.
        :return:
            flux_matrix, a np.array of size (n, n).
            flux_matrix_alltraj, a list of np.array of size (n, n). Each element is a flux matrix of one trajectory.
        """
        node_num = len(self.node_map_int_2_s)
        flux_matrix = np.zeros((node_num, node_num), dtype=np.int64)
        flux_matrix_alltraj = [np.zeros((node_num, node_num), dtype=np.int64) for i in range(len(self.traj_node))]
        # the flux from node_i(state i1, i2) to node_j(state j1, j2) is the sum of flux from i1 to j1, i1 to j2, i2 to j1, i2 to j2
        for rep in range(len(self.traj_node)):
            for node_i in range(node_num):
                for node_j in range(node_num):
                    for s_i in self.node_map_int_2_s[node_i]:
                        for s_j in self.node_map_int_2_s[node_j]:
                            flux_matrix_alltraj[rep][node_i, node_j] += np.sum(
                                self.flux_matrix_raw_alltraj[rep][self.state_map_s_2_int[s_i], self.state_map_s_2_int[s_j]])


        flux_matrix[:] = np.sum(flux_matrix_alltraj, axis=0)
        self.flux_matrix, self.flux_matrix_alltraj = flux_matrix, flux_matrix_alltraj
        # net flux
        self.net_flux_matrix = np.zeros((node_num, node_num), dtype=np.int64)
        self.net_flux_matrix_alltraj = [np.zeros((node_num, node_num), dtype=np.int64) for i in range(len(self.traj_node))]
        for rep in range(len(self.traj_node)):
            for i in range(node_num):
                for j in range(node_num):
                    self.net_flux_matrix_alltraj[rep][i, j] = flux_matrix_alltraj[rep][i, j] - flux_matrix_alltraj[rep][j, i]
        self.net_flux_matrix = np.sum(self.net_flux_matrix_alltraj, axis=0)

        return self.flux_matrix, self.flux_matrix_alltraj

    def calc_transition_probability_matrix(self):
        """
        Calculate the transition probability matrix.
        :return:
        """
        if self.flux_matrix is None or self.flux_matrix_alltraj is None:
            self.calc_flux_matrix()

        num_of_node = len(self.node_map_int_2_s)
        self.transition_probability_matrix = np.zeros((num_of_node, num_of_node))
        for i in range(num_of_node):
            flux_sum = np.sum(self.flux_matrix[i, :])
            if flux_sum == 0:
                self.transition_probability_matrix[i, :] = 0
            else:
                self.transition_probability_matrix[i, :] = self.flux_matrix[i, :] / flux_sum
        self.transition_probability_matrix_alltraj = np.zeros(
            (len(self.flux_matrix_alltraj), num_of_node, num_of_node))
        for traj_i, flux_i in enumerate(self.flux_matrix_alltraj):
            for i in range(num_of_node):
                flux_sum = np.sum(flux_i[i, :])
                if flux_sum == 0:
                    self.transition_probability_matrix_alltraj[traj_i, i, :] = 0
                else:
                    self.transition_probability_matrix_alltraj[traj_i, i, :] = flux_i[i, :] / flux_sum

        return self.transition_probability_matrix

    def calc_net_flux_matrix(self):
        """
        Calculate the net flux matrix.
        :return:
        """
        if self.flux_matrix is None or self.flux_matrix_alltraj is None:
            self.calc_flux_matrix()

        node_num = len(self.node_map_int_2_s)
        self.net_flux_matrix = np.zeros((node_num, node_num), dtype=np.int64)
        self.net_flux_matrix_alltraj = [np.zeros((node_num, node_num), dtype=np.int64) for i in
                                        range(len(self.traj_node))]
        for rep in range(len(self.traj_node)):
            for i in range(node_num):
                for j in range(node_num):
                    self.net_flux_matrix_alltraj[rep][i, j] = self.flux_matrix_alltraj[rep][i, j] - self.flux_matrix_alltraj[rep][
                        j, i]
        self.net_flux_matrix = np.sum(self.net_flux_matrix_alltraj, axis=0)

        return self.net_flux_matrix, self.net_flux_matrix_alltraj

    def calc_passage_time(self):
        """
        Calculate the passage time from every node to every node. This function does not respond to self.lag_step.
        :return: passage_time_length_alltraj
            A list of matrix. One matrix for each traj.
            matrix[i][j] is a list of passage time from node_i to node_j.
        """
        passage_time_len, passage_time_point = count_passage(self.traj_node, len(self.node_map_int_2_s))
        self.passage_time_length_alltraj = passage_time_len
        self.passage_time_point_alltraj = passage_time_point
        return passage_time_len

    def calc_passage_time_raw(self):
        """
        Calculate the passage time from every state to every state on raw traj. This function does not respond to self.lag_step.
        :return: passage_time_length_alltraj
            A list of matrix. One matrix for each traj.
            matrix[i][j] is a list of passage time from node_i to node_j.
        """
        passage_time_len, passage_time_point = count_passage(self.traj_raw_alltraj, len(self.state_map_int_2_s))
        self.passage_time_length_alltraj_raw = passage_time_len
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
        Compute and return the mfpt from self.passage_time_length_alltraj
        This result has not been cycle corrected.
        :param traj_type: "lumped" or "raw"
        :return: mfpt, a np.array of size (n, n), mfpt from every node to every node.
                 mfpt_every_traj, mfpt for individual traj.
        """
        if traj_type == "raw":
            if self.passage_time_length_alltraj_raw is None:
                warnings.warn("self.passage_time_length_alltraj_raw is None. Calculating passage time.")
                self.calc_passage_time_raw()
            passage_time = self.passage_time_length_alltraj_raw
            node_num_length = len(self.state_map_int_2_s)
        elif traj_type == "lumped":
            if self.passage_time_length_alltraj is None:
                warnings.warn("self.passage_time_length_alltraj is None. Calculating passage time.")
                self.calc_passage_time()
            passage_time = self.passage_time_length_alltraj
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
        This result has not been cycle corrected.
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
        This result has not been cycle corrected.
        :param traj_type: "lumped" or "raw"
        :return: rate, a np.array of size (n, n), rate from every node to every node.
                 rate_every_traj, a list of rate for individual traj.
        """
        if traj_type == "raw":
            if self.passage_time_length_alltraj_raw is None:
                warnings.warn("self.passage_time_length_alltraj_raw is None. Calculating passage time.")
                self.calc_passage_time_raw()
            passage_time = self.passage_time_length_alltraj_raw
            node_num_length = len(self.state_map_int_2_s)
            # counter = self.state_Counter
            counter = Counter([i for traj in self.traj_raw_alltraj for i in traj])
            traj_list = self.traj_raw_alltraj
        elif traj_type == "lumped":
            if self.passage_time_length_alltraj is None:
                warnings.warn("self.passage_time_length_alltraj is None. Calculating passage time.")
                self.calc_passage_time()
            passage_time = self.passage_time_length_alltraj
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
    This is the class for plotting the mechanism graph. Especially for the lumped graph.

    """

    def __init__(self, model):
        """
        :param model : Sf_model
        """
        if isinstance(model, Sf_model):
            self.model = model
        else:
            raise ValueError("model should be an instance of Sf_model.")

    def build_graph(self, flux_cutoff=2):
        """
        Build the graph (Directed).
        Each edge has the following attributes:
            flux_ij : the flux from node i to node j.
            flux_ji : the flux from node j to node i.
            mfpt_ij : the mean first passage time of the states pair that contributes the most to the flux_ij. (ps)
            mfpt_ji : the mean first passage time of the states pair that contributes the most to the flux_ji. (ps)
            Only the shortest passage is considered.
        If you change the lumping, you should call this function again.
        :return:
        """
        self.G = nx.DiGraph()
        for node_i in self.model.node_map_int_2_s:
            self.G.add_node(node_i)
        for i_node in self.model.node_map_int_2_s:
            for j_node in self.model.node_map_int_2_s:
                if i_node != j_node:
                    flux_ij = self.model.flux_matrix[i_node, j_node]
                    flux_ji = self.model.flux_matrix[j_node, i_node]
                    mfpt_ij = np.inf
                    mfpt_ji = np.inf
                    df = self.get_edge_info_df(i_node, j_node, flux_cutoff)
                    n_rep = len(self.model.traj_raw_alltraj)
                    f_ij_alltraj = np.array([self.model.flux_matrix_raw_alltraj[rep][i_node, j_node] for rep in range(n_rep)])
                    f_ji_alltraj = np.array([self.model.flux_matrix_raw_alltraj[rep][j_node, i_node] for rep in range(n_rep)])
                    if np.all(f_ij_alltraj > flux_cutoff):
                        df = df.sort_values(by="flux_ij", ascending=False)
                        mfpt_ij = df.iloc[0]["mfpt_ij"]
                    if np.all(f_ji_alltraj > flux_cutoff):
                        df = df.sort_values(by="flux_ji", ascending=False)
                        mfpt_ji = df.iloc[0]["mfpt_ji"]
                    self.G.add_edge(i_node, j_node, flux_ij=flux_ij, flux_ji=flux_ji, mfpt_ij=mfpt_ij, mfpt_ji=mfpt_ji)

    def get_edge_info_df(self, node_i, node_j, flux_cutoff=2):
        """
        Get all the edge info in a pd.DataFrame.
        :param node_i : int
        :param node_j : int
        :param flux_cutoff : int, only when the flux in every replica is larger than flux_cutoff, the mfpt will be calculated.
        only when there is flux between states, the mfpt will be calculated.
        | i_state | j_state | i_name | j_name | flux_ij | flux_ji | mfpt_ij | mfpt_ji |
        :return: df, pd.DataFrame
        """
        node_i_states = [self.model.state_map_s_2_int[s]  for s in self.model.node_map_int_2_s[node_i]]
        node_j_states = [self.model.state_map_s_2_int[s]  for s in self.model.node_map_int_2_s[node_j]]
        i_state_list = []
        j_state_list = []
        i_name_list = []
        j_name_list = []
        flux_ij_list = []
        flux_ji_list = []
        mfpt_ij_list = []
        mfpt_ji_list = []
        n_rep = len(self.model.traj_raw_alltraj)
        for i_s in node_i_states:
            for j_s in node_j_states:
                flux_ij = self.model.flux_matrix_raw[i_s, j_s]
                flux_ji = self.model.flux_matrix_raw[j_s, i_s]
                mfpt_ij = np.inf
                mfpt_ji = np.inf
                f_ij_alltraj = np.array(
                    [self.model.flux_matrix_raw_alltraj[rep][i_s, j_s] for rep in range(n_rep)])
                f_ji_alltraj = np.array(
                    [self.model.flux_matrix_raw_alltraj[rep][j_s, i_s] for rep in range(n_rep)])
                if np.all(f_ij_alltraj > flux_cutoff):
                    mfpt_ij, mfpt_alltraj, jump = self.model.get_mfpt_AB_shortest_passage(i_s, j_s)
                if np.all(f_ji_alltraj > flux_cutoff):
                    mfpt_ji, mfpt_alltraj, jump = self.model.get_mfpt_AB_shortest_passage(j_s, i_s)
                i_state_list.append(i_s)
                j_state_list.append(j_s)
                i_name_list.append(self.model.state_map_int_2_s[i_s])
                j_name_list.append(self.model.state_map_int_2_s[j_s])
                flux_ij_list.append(flux_ij)
                flux_ji_list.append(flux_ji)
                mfpt_ij_list.append(mfpt_ij)
                mfpt_ji_list.append(mfpt_ji)
        df = pd.DataFrame({"i_state": i_state_list,
                           "j_state": j_state_list,
                           "i_name": i_name_list,
                           "j_name": j_name_list,
                           "flux_ij": flux_ij_list,
                           "flux_ji": flux_ji_list,
                           "mfpt_ij": mfpt_ij_list,
                           "mfpt_ji": mfpt_ji_list})
        return df


    def get_node_info_df(self, node_ind):
        """
        Get all the node info in a pd.DataFrame.
        | state | proportion |
        :return: df, pd.DataFrame
        """
        state_list = []
        proportion_list = []
        for state in self.model.node_map_int_2_s[node_ind]:
            state_list.append(state)
            proportion_list.append(self.model.state_distribution[state])
        df = pd.DataFrame({"state": state_list, "proportion": proportion_list})
        return df


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
            p_i = [str_2_xy(si) for si in self.model.node_map_int_2_s[i]]
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
                node = self.model.node_map_int_2_s[i]
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
                   net_flux_cutoff=10, net_flux_node_cutoff=15,
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
                if np.any(self.model.net_flux_matrix[i, :] > net_flux_node_cutoff) or np.any(self.model.net_flux_matrix[:, i] > net_flux_node_cutoff):
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
        size_dict = {i: min_size + (max_size - min_size) * (self.model.node_distribution[i] - min_p) / (max_p - min_p) for i
                     in self.G.nodes}
        # color
        color_dict = {1: [1  , 0.1, 0.1, 1],
                      2: [0.5, 0.5, 1  , 1],
                      3: [1  , 0.5, 0.0, 1],
                      4: [0.0, 0.5, 1.0, 1],
                      5: [0.5, 1  , 1  , 1],
                      }
        color_list = []
        for node in sub_G:
            s_list = [(s, self.model.state_Counter[s]) for s in self.model.node_map_int_2_s[node]]
            s_list = sorted(s_list, key=lambda x: x[1], reverse=True)
            K_number = s_list[0][0].count("K") + s_list[0][0].count("C")
            # K_number = self.model.node_map_int_2_s[node][0][:5].count("K") + self.model.node_map_int_2_s[node][0][:5].count("C")
            # K_number = (K_number + 8) % 10
            color_list.append(color_dict[K_number])
        nx.draw_networkx_nodes(sub_G, ax=ax, pos=pos,
                               node_size=[size_dict[i] for i in sub_G.nodes], node_color=color_list, alpha=node_alpha)
        nx.draw_networkx_labels(sub_G, ax=ax, pos=pos, labels=nx.get_node_attributes(sub_G, "label"),
                                font_family='monospace')

        # 4 draw edge based on net_flux cutoff, rescale edge_width
        edge_list = []
        edge_label = {}
        edge_rate = []

        for u, v, d in sub_G.edges(data=True):
            if d["flux_ij"] - d["flux_ji"] > net_flux_cutoff:
                edge_list.append((u, v))
                edge_label[(u, v)] = f'{d["flux_ij"] - d["flux_ji"]:d}'
                edge_rate.append(1000/d["mfpt_ij"]) # 1/ns
        # rescale edge_width to k * rate ^ a
        if edge_width is None:
            # make the min visible (width=0.1)
            e_width_a = 0.5
            edge_rate_ = [rate for rate in edge_rate if rate > 0]
            e_width_k = 0.1/(min(edge_rate_) ** e_width_a)
            edge_width = (e_width_k, e_width_a)
        (e_width_k, e_width_a) = edge_width
        width_list = [e_width_k*rate**e_width_a for rate in
                      edge_rate]
        nx.draw_networkx_edges(sub_G, ax=ax, pos=pos, edgelist=edge_list, width=width_list, alpha=edge_alpha,
                               connectionstyle='arc3,rad=0.05',
                               node_size=[size_dict[i] for i in sub_G.nodes])
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
