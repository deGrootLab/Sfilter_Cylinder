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
            self.raw_traj_df = None
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

        # # build a DataFrame for the raw traj
        # index_A = []
        # index_B = []
        # index_pair = []
        # A_list = []
        # B_list = []
        # A_proportion = []
        # B_proportion = []
        # net_flux_AB = []
        # flux_AB = []
        # flux_BA = []
        # rate_AB = []
        # rate_BA = []
        # dist_AB = []
        # dist_BA = []
        # for i in range(len(self.flux_matrix_raw)):
        #     for j in range(len(self.flux_matrix_raw)):
        #         if i != j:
        #             if self.flux_matrix_raw[i, j] > 0 and self.net_flux_matrix_raw[i, j] >= 0 and ((j, i) not in index_pair):
        #                 index_pair.append((i, j))
        #                 index_A.append(i)
        #                 index_B.append(j)
        #                 A_list.append(self.state_map_int_2_s[i])
        #                 B_list.append(self.state_map_int_2_s[j])
        #                 A_proportion.append(self.state_distribution[self.state_map_int_2_s[i]])
        #                 B_proportion.append(self.state_distribution[self.state_map_int_2_s[j]])
        #                 net_flux_AB.append(self.net_flux_raw[i, j])
        #                 flux_AB.append(self.flux_matrix_raw[i, j])
        #                 flux_BA.append(self.flux_matrix_raw[j, i])
        #                 rate_AB.append(self.rate_raw[i, j])
        #                 rate_BA.append(self.rate_raw[j, i])
        #                 d_ab, d_ba = k_distance(self.state_map_int_2_s[i], self.state_map_int_2_s[j])
        #                 dist_AB.append(d_ab)
        #                 dist_BA.append(d_ba)
        # self.raw_traj_df = pd.DataFrame({"index_A": index_A, "index_B": index_B, "A": A_list, "B": B_list,
        #                                  "A_proportion": A_proportion, "B_proportion": B_proportion,
        #                                  "net_flux_AB": net_flux_AB, "flux_AB": flux_AB, "flux_BA": flux_BA,
        #                                  "rate_AB": rate_AB, "rate_BA": rate_BA,
        #                                  "dist_AB": dist_AB, "dist_BA": dist_BA})
        # self.raw_traj_df["rate_AB_x_rate_BA"] = self.raw_traj_df["rate_AB"] * self.raw_traj_df["rate_BA"]

    def get_state_index(self, state):
        pass

    def get_flux_AB(self, state_A, state_B):
        pass

    def get_concentration(self, state):
        pass

    def get_mfpt_AB(self, state_A, state_B):
        pass

    def _cycle_correct(self, p ):
        i, j, perm_list = p
        df_ij = self.rate_cycle_correct(i, j, perm_list)
        df_ji = self.rate_cycle_correct(j, i, perm_list)
        df_ij_cc = df_ij[df_ij["h_Permeation"] <= 1]
        df_ji_cc = df_ji[df_ji["h_Permeation"] <= 1]
        r_AB_npass_cc, r_BA_npass_cc = (len(df_ij_cc) / (self.state_Counter[self.state_map_int_2_s[i]] * self.time_step),
                                        len(df_ji_cc) / (self.state_Counter[self.state_map_int_2_s[j]] * self.time_step))
        r_AB_inv_mfpt_cc, r_BA_inv_mfpt_cc = 1/df_ij_cc.mean()["passage_time"], 1/df_ji_cc.mean()["passage_time"]
        return r_AB_npass_cc, r_BA_npass_cc, r_AB_inv_mfpt_cc, r_BA_inv_mfpt_cc

    def init_raw_properties_2(self, perm_list, clean_lump=False, dtype_lumped=np.int16,
                              proportion_cutoff=1e-5, flux_cutoff=1, n_cpu=None, ):
        """

        :param perm_list:
        :param clean_lump:
        :param dtype_lumped:
        :param proportion_cutoff: default 1e-5
            when proportion of both state_A and state_B > proportion_cutoff, edge [A,B] will be counted.
        :param flux_cutoff: default 1
            when flux_AB > flux_cutoff and flux_BA > flux_cutoff, edge [A,B] will be counted.
        :param n_cpu:
        pre-screen the edge by proportion_cutoff and flux_cutoff will save time.
        :return:
        """
        if clean_lump:
            self.set_lumping_from_str([], dtype_lumped, calc_passage_time=False)  # At the end of this function, properties will be calculated.
            self.calc_passage_time()
        self.flux_raw = copy.deepcopy(self.flux_matrix)
        self.net_flux_raw = copy.deepcopy(self.net_flux_matrix)

        self.passage_time_length_alltraj_raw = copy.deepcopy(self.passage_time_length_alltraj)
        rate_n_passage = self.get_rate_passage_time(traj_type="raw")[0]
        rate_inv_mfpt  = self.get_rate_inverse_mfpt(traj_type="raw")[0]


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
        rate_AB_cc = []  # with cycle correction
        rate_BA_cc = []
        task_list = []
        dist_AB = []
        dist_BA = []
        for i in range(len(self.net_flux_raw)):
            for j in range(len(self.net_flux_raw)):

                if i != j:
                    proportion_check = self.state_distribution[self.state_map_int_2_s[i]] >= proportion_cutoff and \
                                        self.state_distribution[self.state_map_int_2_s[j]] >= proportion_cutoff
                    flux_check = self.flux_raw[i, j] > flux_cutoff and self.flux_raw[j, i] > flux_cutoff and \
                                 self.net_flux_raw[i, j] >= 0
                    if proportion_check and flux_check and ((j, i) not in index_pair):
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


                        # without cycle correction
                        r_AB_npass,    r_BA_npass    = rate_n_passage[i, j], rate_n_passage[j, i]
                        r_AB_inv_mfpt, r_BA_inv_mfpt = rate_inv_mfpt[i, j],  rate_inv_mfpt[j, i]
                        rate_AB.append([r_AB_npass,  r_AB_inv_mfpt, ])
                        rate_BA.append([r_BA_npass,  r_BA_inv_mfpt, ])

                        # with cycle correction
                        task_list.append((i, j, perm_list))

                        d_ab, d_ba = k_distance(self.state_map_int_2_s[i], self.state_map_int_2_s[j])
                        dist_AB.append(d_ab)
                        dist_BA.append(d_ba)
        # parallelization
        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            futures = []
            for p in task_list:
                future = executor.submit(self._cycle_correct, p)
                futures.append(future)

            for future in futures:
                r_AB_npass_cc, r_BA_npass_cc, r_AB_inv_mfpt_cc, r_BA_inv_mfpt_cc = future.result()
                rate_AB_cc.append([r_AB_npass_cc, r_AB_inv_mfpt_cc])
                rate_BA_cc.append([r_BA_npass_cc, r_BA_inv_mfpt_cc])
        self.raw_traj_df = pd.DataFrame({"index_A": index_A, "index_B": index_B, "A": A_list, "B": B_list,
                                         "A_proportion": A_proportion, "B_proportion": B_proportion,
                                         "net_flux_AB": net_flux_AB, "flux_AB": flux_AB, "flux_BA": flux_BA,
                                         "rate_AB_npass":    [i[0] for i in rate_AB],
                                         "rate_AB_inv_mfpt": [i[1] for i in rate_AB],
                                         "rate_BA_npass":    [i[0] for i in rate_BA],
                                         "rate_BA_inv_mfpt": [i[1] for i in rate_BA],
                                         "rate_ABcc_npass":    [i[0] for i in rate_AB_cc],
                                         "rate_ABcc_inv_mfpt": [i[1] for i in rate_AB_cc],
                                         "rate_BAcc_npass":    [i[0] for i in rate_BA_cc],
                                         "rate_BAcc_inv_mfpt": [i[1] for i in rate_BA_cc],
                                         "dist_AB": dist_AB, "dist_BA": dist_BA})
        self.raw_traj_df["rate_AB_x_rate_BA"] = self.raw_traj_df["rate_ABcc_inv_mfpt"] * self.raw_traj_df["rate_BAcc_inv_mfpt"]
        self.raw_traj_df = self.raw_traj_df.sort_values(by='rate_AB_x_rate_BA', ascending=False)
        return self.raw_traj_df

    def rate_cycle_correct(self, ind_i, ind_j, perm_list):
        rep_index_list = []
        passage_index_list = []
        passage_time = []
        half_p_count = []

        for rep in range(len(perm_list)):
            pt_len = self.passage_time_length_alltraj_raw[rep][ind_i][ind_j]
            pt_point = self.passage_time_point_alltraj_raw[rep][ind_i][ind_j]
            perm_array = perm_list[rep][["enter", "time"]].to_numpy()  # unit in ps. up/down will be ignored

            for i_passage, (end, length) in enumerate(zip(pt_point, pt_len)):
                start = end - length  # unit in frame
                passage_index_list.append(i_passage)
                rep_index_list.append(rep)
                passage_time.append(length * self.time_step)
                half_p_count.append(np.sum(np.logical_and(perm_array > start * self.time_step,
                                                          perm_array < end * self.time_step)))  # convert unit to ps
        Df_ij = pd.DataFrame({"replica": rep_index_list,
                              "passage_index": passage_index_list,
                              "passage_time": passage_time,
                              "h_Permeation": half_p_count,
                              })
        return Df_ij
        # return Df_ij[Df_ij["h_Permeation"] <= 1].mean()["passage time"]

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

    """

    def __init__(self, model):
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
        if isinstance(model, Sf_model):
            self.model = model
        else:
            raise ValueError("model should be an instance of Sf_model.")

        # build a DiGraph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(len(self.model.node_map_int_2_s)))
        for i in range(len(self.model.node_map_int_2_s)):
            for j in range(len(self.model.node_map_int_2_s)):
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


    def get_graph_rate_df_node_ij(self, node_i, node_j):
        """
        for a given edge (node_i -> node_j), use the rate with the maximum flux.
        :param df:
        :param node_map_int_2_s:
        :param node_i:
        :param node_j:
        :return: state_i, state_j, rate_ij, rate_ji
        """
        df = self.raw_traj_df
        node_map_int_2_s = self.node_map_int_2_s
        # loop over all states
        max_flux_ij = 0
        rate_ij = 0
        rate_ji = 0
        str_i = ""
        str_j = ""
        for s_i in node_map_int_2_s[node_i]:
            for s_j in node_map_int_2_s[node_j]:
                # find the row in df, it can either be A->B or B->A
                row_AB = df[(df["A"] == s_i) & (df["B"] == s_j)]
                row_BA = df[(df["A"] == s_j) & (df["B"] == s_i)]

                if len(row_AB) == 0 and len(row_BA) == 0:
                    continue
                elif len(row_AB) == 0 and len(row_BA) == 1:
                    if row_BA["flux_BA"].values[0] > max_flux_ij:
                        max_flux_ij = row_BA["flux_BA"].values[0]
                        rate_ij = row_BA["rate_ABcc_inv_mfpt"].values[0]
                        rate_ji = row_BA["rate_BAcc_inv_mfpt"].values[0]
                        str_i = s_i
                        str_j = s_j
                elif len(row_AB) == 1 and len(row_BA) == 0:
                    if row_AB["flux_AB"].values[0] > max_flux_ij:
                        max_flux_ij = row_AB["flux_AB"].values[0]
                        rate_ij = row_AB["rate_ABcc_inv_mfpt"].values[0]
                        rate_ji = row_AB["rate_BAcc_inv_mfpt"].values[0]
                        str_i = s_i
                        str_j = s_j
                else:
                    raise ValueError("There should be only one row in df.")
        return str_i, str_j, rate_ij, rate_ji




    def set_graph_rate_df(self):
        """
        for each edge, use the rate with the maximum flux.
        :param df:
        :return:
        """
        df = self.raw_traj_df
        node_map_int_2_s = self.node_map_int_2_s
        # loop over edges
        for node_i, node_j, data in self.G.edges(data=True):
            str_i, str_j, rate_ij, rate_ji = self.get_graph_rate_df_node_ij(node_i, node_j)
            self.G.edges[node_i, node_j]["rate_ij"] = rate_ij
            self.G.edges[node_i, node_j]["rate_ji"] = rate_ji




    def get_node_proportion(self, node):
        """
        Get the proportion of each state in a node.
        :param node: int, the node.
        :return: sum proportion of every state.
        """
        return np.sum([self.state_distribution[s] for s in self.node_map_int_2_s[node]])



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
        color_dict = {1: [1  , 0.1, 0.1, 1],
                      2: [0.5, 0.5, 1  , 1],
                      3: [1  , 0.5, 0.0, 1],
                      4: [0.0, 0.5, 1.0, 1],
                      5: [0.5, 1  , 1  , 1],
                      }
        color_list = []
        for node in sub_G:
            K_number = self.node_map_int_2_s[node][0][:5].count("K") + self.node_map_int_2_s[node][0][:5].count("C")
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



