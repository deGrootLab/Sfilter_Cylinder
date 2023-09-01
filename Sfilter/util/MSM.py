import copy
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from .output_wrapper import read_k_cylinder
import pyemma
from scipy.optimize import minimize


def matrix_to_df(matrix, msm, cut_off=0.01):
    """
    convert to pd.DataFrame and print
    """
    for node_num in range(len(msm.int_2_s)):
        if msm.distribution[node_num] < cut_off:
            node_num -= 1
            break
    node_num += 1
    data = {}
    data["name"] = [str(msm.int_2_s[i]) for i in range(node_num)]
    for i in range(node_num):
        data[i] = matrix[:node_num, i]
    return pd.DataFrame(data)


def MFPT_A_to_B(traj, A, B):
    """
    compute mean first passage time from state A to state B
    A -> B
    alg1:
    B A-A C A-A C-C-C B-B
      ^               ^  8 steps
    :param traj: a np.array of state int
    :param A: int, node A
    :param B: int, node B
    :return:
        MFPT_list, a list of FPT from state A to state B
    """
    if A == B:
        raise ValueError("When calculating MFPT, A and B should not be the same")
    map_A = traj == A
    to_A = np.where(~map_A[:-1] & map_A[1:])[0] + 1  # search X to A transition
    map_B = traj == B
    to_B = np.where(~map_B[:-1] & map_B[1:])[0] + 1  # search X to B transition
    if traj[0] == A:
        to_A = np.insert(to_A, 0, 0)
    elif traj[0] == B:
        to_B = np.insert(to_B, 0, 0)

    MFPT_list = []

    if len(to_B) == 0:
        return MFPT_list
    elif len(to_B) == 1:
        if len(to_A) == 0:
            return MFPT_list
        elif to_A[0] < to_B[0]:
            return [to_B[0] - to_A[0]]
        else:
            return MFPT_list
    else:
        if len(to_A) == 0:
            return MFPT_list
        else:
            ia = 0
            if to_A[0] < to_B[0]:
                ia = 1
                MFPT_list.append(to_B[0] - to_A[0])
            for b0, b1 in zip(to_B[:-1], to_B[1:]):
                while ia < len(to_A):
                    if to_A[ia] > b0:
                        if to_A[ia] > b1:
                            break
                        else:
                            MFPT_list.append(b1 - to_A[ia])
                            break
                    ia += 1
            return MFPT_list


class SF_msm:
    def __init__(self, file_list, start=0, end=None, step=1, method="K_priority"):
        """
        file_list: list of file names, if leave empty, please do set_state_str() later
        start: starting frame
        end: ending frame
        """
        self.state_str = []  # each trajectory is a list of state string, this is a list of trajectories
        self.state_array = []  # each trajectory is a list of state int, this is a list of trajectories
        self.s_2_int = {}  # state string to int
        self.int_2_s = {}  # int to state string
        self.distribution = {}
        self.transition_matrix = None
        self.node_counter = None
        self.merge_list = []
        self.time_step = []
        if isinstance(file_list, list):
            pass
        elif isinstance(file_list, str):
            file_list = [file_list]
        for traj, meta_data, K_occupency, W_occupency in [read_k_cylinder(file, method) for file in file_list]:
            self.state_str.append(traj[start:end:step])
            self.time_step.append(meta_data["time_step"] * step)
        flattened = [num for sublist in self.state_str for num in sublist]
        self.state_counter = Counter(flattened)

    def set_state_str(self, state_str):
        """
        set state_str
        update self.state_counter, a Counter, state to number of occurrence
        """
        self.state_str = state_str
        flattened = [num for sublist in self.state_str for num in sublist]
        self.state_counter = Counter(flattened)

    def set_state_array(self, state_array):
        """
        set state_array
        update
            self.state_counter, a Counter, state to number of occurrence
            self.state_str, a list of trajectories, each trajectory is a list of state string
            int_2_s and s_2_int
        """
        if self.state_str is None or self.state_str == []:
            self.state_str = []
        else:
            raise ValueError("state_str is exist. You should not set state_str based on state_array")
        self.state_array = state_array
        if isinstance(state_array, list):
            for traj_array in state_array:
                self.state_str.append([str(i) for i in traj_array])
        elif isinstance(state_array, np.ndarray):
            self.state_str.append([str(i) for i in state_array])
        else:
            raise ValueError("state_array must be list or np.ndarray")

        # update self.state_counter
        flattened = [num for sublist in self.state_str for num in sublist]
        self.state_counter = Counter(flattened)

        # update int_2_s and s_2_int
        self.int_2_s = {}
        self.s_2_int = {}
        for i in self.state_counter:
            self.int_2_s[int(i)] = [i]
            self.s_2_int[i] = int(i)

    def calc_state_array(self, merge_list=None):
        """
        convert state_str to state_array
        Those variable will be updated:
            self.merge_list
            self.state_array, a list of trajectories, each trajectory is a list of state int
            self.s_2_int, a dictionary, state to int
            self.int_2_s, a dictionary, int to state(s)
            self.distribution, a dictionary, int to distribution
            self.node_counter, a Counter, node(grouped states) to number of occurrence
        """
        if merge_list is None:
            merge_list = []
        self.merge_list = merge_list
        state_set = set(self.state_counter.keys())
        # check if every state in merge_list is in state_set
        for node in merge_list:
            for s in node:
                if s not in state_set:
                    raise ValueError("state {} not in state_set".format(s))
        node_count_list = []
        # add the merged node
        for node in merge_list:
            num = 0  # count occurrence
            for s in node:
                num += self.state_counter[s]
                state_set.remove(s)
            node_count_list.append([node, num])
        # add the non-merged node
        for s in state_set:
            node_count_list.append([[s], self.state_counter[s]])
        node_count_list = sorted(node_count_list, key=lambda x: x[1], reverse=True)
        self.int_2_s = {}
        self.s_2_int = {}
        self.distribution = {}
        self.node_counter = Counter()
        num_frame = sum(self.state_counter.values())
        for i, (node, _) in enumerate(node_count_list):
            self.int_2_s[i] = node
            for s in node:
                self.s_2_int[s] = i
            self.distribution[i] = _ / num_frame
            self.node_counter[i] = _

        # update self.state_array, a list of trajectories, each trajectory is a list of state int
        self.state_array = []
        for traj in self.state_str:
            traj_str = np.array(traj)
            traj_int = np.zeros(len(traj), dtype=np.int64)
            for s in self.s_2_int.keys():
                traj_int[traj_str == s] = self.s_2_int[s]
            self.state_array.append(traj_int)

    def get_transition_matrix(self, lag_step=1):
        """
        compute states transition matrix.
        a numpy array, each element is the number of transition from state i to state j
        """""
        state_num = max([max(traj) for traj in self.state_array])
        f_matrix = np.zeros((state_num + 1, state_num + 1), dtype=np.int64)
        # for traj in self.state_array:
        #    state_start = traj[:-lag_step]
        #    state_end = traj[lag_step:]
        #    for m_step in np.array([state_start, state_end]).T:
        #        tran_matrix[m_step[0], m_step[1]] += 1
        for traj in self.state_array:
            state_start = traj[:-lag_step]
            state_end = traj[lag_step:]
            for m_step in np.array([state_start, state_end]).T:
                f_matrix[m_step[0], m_step[1]] += 1

        return f_matrix

    def f_matrix_2_rate_matrix(self, f_matrix, physical_time):
        """
        compute rate matrix
        a numpy array, each element is the rate from state i to state j
        input:
            f_matrix: a numpy array, each element is the number of transition from state i to state j
            physical_time: physical time for each step
        return:
            rate_matrix: a numpy array, each element is the rate from state i to state j
        """
        rate_matrix = np.array(f_matrix, dtype=np.float64)  # convert int to float
        for i in range(rate_matrix.shape[0]):
            rate_matrix[i, :] /= self.node_counter[i] * physical_time
            rate_matrix[i, i] = 0
        return rate_matrix

    def get_rate_matrix(self, lag_step=1, physical_time=None):
        """
        compute rate matrix
        a numpy array, each element is the rate from state i to state j
        input:
            lag_step: lag time in step
            physical_time: physical time for each step, if not given, use the time_step that was read from file
        return:
            rate_matrix: a numpy array, each element is the rate from state i to state j
            t_matrix: a numpy array, each element is the number of transition from state i to state j
        """
        if physical_time is None:  # use the time_step that was read from file
            if np.allclose(self.time_step, self.time_step[0]):
                physical_time = self.time_step[0]
            else:
                raise ValueError("physical_time is not given, and time_step is not equal")

        f_matrix = self.get_transition_matrix(lag_step)
        rate_matrix = self.f_matrix_2_rate_matrix(f_matrix, physical_time)
        return rate_matrix

    def f_matrix_2_transition_probability(self, f_matrix):
        """
        compute transition probability matrix (between steps)
        return: transition_probability_matrix
            a numpy array, each element is the probability of transition from state i to state j
            The sum of each row is 1.
        """
        p_matrix = np.array(f_matrix, dtype=np.float64)  # int to float
        p_matrix /= p_matrix.sum(axis=1, keepdims=True)  # normalize each row
        return p_matrix

    def get_transition_probability(self, lag_step=1):
        """
        compute transition probability matrix (between steps)
        return: transition_probability_matrix
            a numpy array, each element is the probability of transition from state i to state j
            The sum of each row is 1.
        """
        f_matrix = self.get_transition_matrix(lag_step)
        return self.f_matrix_2_transition_probability(f_matrix)

    def get_CK_test(self, lag_step=1, test_time=[2, 4]):
        """
        run Chapman-Kolmogorov test
        :param lag_step: int, lag step
        :param test_time: a list of int, a list of lag step you want to test
        :return: (reality, prediction)
        """
        reality = []
        prediction = []
        p_matrix_0 = self.get_transition_probability(lag_step)
        reality.append(p_matrix_0)
        prediction.append(p_matrix_0)
        for t in test_time:
            reality.append(self.get_transition_probability(t * lag_step))
            prediction.append(np.linalg.matrix_power(p_matrix_0, t))
        return np.array(reality), np.array(prediction)

    # def plot_CK_test(self, lag_step=1, test_time=[2, 4], ax=None, num_node=5):
    #     """
    #     plot Chapman-Kolmogorov test result
    #     :param lag_step: int, lag step
    #     :param test_time: a list of int, a list of lag step you want to test
    #     :param ax:
    #     :param num_node: plot the first num_node nodes
    #     :return:
    #     """
    #     pass

    def get_MFPT_pair(self, A, B):
        """
        giving a pair of nodes, return the first-passage time from state A to state B
        A -> B
        alg1:
        B A-A C A-A C-C-C B-B
          ^               ^  8 steps
        :param A: int, node A
        :param B: int, node B
        :return:
            FPT_list, a list of FPT from state A to state B, unit in step
        """
        FPT_list = []
        for traj in self.state_array:
            FPT_list += MFPT_A_to_B(traj, A, B)
        return FPT_list

    def get_nfp_rate_pair(self, A, B):
        """
        giving a pair of nodes, return the nfp_rate from state A to state B
        The nft_rate(A -> B) is defined as this:
        nfp_rate = number of first passage / occurring time of state A
        :param A: int, node A
        :param B: int, node B
        :return:
            nfp_rate : float, the nfp_rate from state A to state B
            FPT_list : list of float, the first passage time from state A to state B, unit in time
        """
        FPT_list = self.get_MFPT_pair(A, B)
        nfp_rate = len(FPT_list) / (self.node_counter[A] * self.time_step[0])
        return nfp_rate, FPT_list

    def get_mfpt_rate_pair(self, A, B):
        """
        giving a pair of nodes, return the mfpt_rate from state A to state B
        The mfpt_rate(A -> B) is defined as this: 1/(mean first passage time)
        :param A: int, node A
        :param B: int, node B
        :return:
            mfpt_rate : float, the mfpt_rate from state A to state B
            FPT_list : list of float, the first passage time from state A to state B, unit in time
        """
        FPT_list = self.get_MFPT_pair(A, B)
        if len(FPT_list) == 0:
            mfpt_rate = 0
        else:
            mfpt_rate = 1 / (np.mean(FPT_list) * self.time_step[0])
        return mfpt_rate, FPT_list

    def population_cutoff(self, population_cutoff=0.01):
        """
        find the node that is above the cutoff.
        All the node in range(n) is above the cutoff.
        :param population_cutoff:
        :return: n, the number of nodes that is above the cutoff
        """
        total_count = self.node_counter.total()
        finish_flag = True
        for n, node_count in self.node_counter.items():
            if node_count / total_count < population_cutoff:
                finish_flag = False
                break
        if finish_flag:
            n += 1
        return n

    def get_MFPT_matrix(self, population_cutoff=0.01):
        """
        compute mean first passage time for every node pairs above population_cutoff
        A -> C
        alg1:
        C A-A B A-A B-B-B C-C
          ^               ^  8 steps
        :param population_cutoff:
        :return:
            MFPT_matrix: a numpy array, each element is the MFPT from state i to state j, unit in step*physical_time_step,
                         If no transition found, np.inf is used
            FPT_list, each element is a list of FPT from state i to state j, unit in step
        """
        # find the node that is above the cutoff, if all nodes are above the cutoff, n = N+1
        n = self.population_cutoff(population_cutoff)
        MFPT_matrix = np.zeros((n, n), dtype=np.float64)
        FPT_list = []
        for i in range(n):
            FPT_list.append([])
            for j in range(n):  # i -> j
                if i == j:
                    fpt_ij = []
                    FPT_list[i].append(fpt_ij)
                else:
                    fpt_ij = self.get_MFPT_pair(i, j)
                    FPT_list[i].append(fpt_ij)
                    if len(fpt_ij) == 0:
                        MFPT_matrix[i, j] = np.inf
                    else:
                        MFPT_matrix[i, j] = np.mean(fpt_ij) * self.time_step[0]
        return MFPT_matrix, FPT_list

    def get_nfp_rate_matrix(self, population_cutoff=0.01):
        """
        compute the nfp_rate for every node pairs above population_cutoff
        :param population_cutoff:
        :return:
            nfp_rate_matrix: a numpy array, each element is the rate from state i to state j, unit in step*physical_time_step,
                         If no transition found, 0 is used
            FPT_list, each element is a list of FPT from state i to state j, unit in step
        """
        # find the node that is above the cutoff
        n = self.population_cutoff(population_cutoff)
        rate_matrix = np.zeros((n, n), dtype=np.float64)
        FPT_list = []
        for i in range(n):
            FPT_list.append([])
            for j in range(n):  # i -> j
                if i == j:
                    rate_matrix[i, j] = 0
                    FPT_list[i].append([])
                else:
                    rate_matrix[i, j], fpt_ij = self.get_nfp_rate_pair(i, j)
                    FPT_list[i].append(fpt_ij)
        return rate_matrix, FPT_list

    def get_matrix(self, lag_step=1, physical_time=None):
        """
        calculate transition matrix, rate_matrix, and transition probability
        return:
            f_matrix: each element is the number of flux from state i to state j
            net_f_matrix: each element is the number of net event between state i to state j (f_ij - f_ji)
            rate_matrix: each element is the rate (number of event / observation time) from state i to state j
            p_matrix: each element is the probability of transition from state i to state j
        input:
            lag_step: lag time in step
            physical_time: physical time for each step, if not given, use the time_step that was read from file
        """
        if physical_time is None:  # use the time_step that was read from file
            if np.allclose(self.time_step, self.time_step[0]):
                physical_time = self.time_step[0]
            else:
                raise ValueError("physical_time is not given, and time_step is not equal")

        f_matrix = self.get_transition_matrix(lag_step)  # transition between states (not steps)
        net_t_matrix = f_matrix - f_matrix.T
        rate_matrix = self.f_matrix_2_rate_matrix(f_matrix, physical_time)
        p_matrix = self.f_matrix_2_transition_probability(f_matrix)
        return f_matrix, net_t_matrix, rate_matrix, p_matrix

    def get_resident_time(self):
        """
        get the resident time distribution for each node (states group)
        """
        res_time = {i: [] for i in range(len(self.int_2_s))}
        for traj in self.state_array:
            time_count = 1
            for s0, s1 in zip(traj[:-1], traj[1:]):
                if s0 == s1:
                    time_count += 1
                else:
                    res_time[s0].append(time_count)
                    time_count = 1
        return res_time

    def get_pyemma_TPT_rate(self):
        rate_matrix = np.zeros((len(self.int_2_s), len(self.int_2_s)))
        msm = pyemma.msm.estimate_markov_model(self.state_array, lag=1,
                                               reversible=False, dt_traj=str(self.time_step[0]) + " ps")
        for i in range(len(self.int_2_s)):
            for j in range(len(self.int_2_s)):
                if i != j:
                    rate_matrix[i, j] = pyemma.msm.tpt(msm, [i], [j]).rate
                else:
                    rate_matrix[i, j] = 0
        return rate_matrix


def states_2_name(node, index=None):
    """
    Convert a list of states to a string
    """
    if index is None:
        string = ""
    elif len(node) <= 1:
        string = f"{index}:"
    else:
        string = f"{index} : \n"

    for s in node[:-1]:
        string += s
        string += "\n"
    string += node[-1]
    return string


def plot_net_T_matrix(ax, msm, cut_off, edge_cutoff, net_t_matrix, iterations, k, position=None,
                      colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
                      text_bbox={"boxstyle": "round", "ec": (1.0, 1.0, 1.0), "fc": (1.0, 1.0, 1.0)}, add_index=True,
                      edge_factor=100
                      ):
    """
    Wrap up function to plot net transition matrix
    """
    G = nx.DiGraph()
    # cutoff at certain density
    node_num = msm.population_cutoff(cut_off)

    # prepare node with name and color
    node_colors = []
    for i in range(node_num):
        if add_index:
            name = states_2_name(msm.int_2_s[i], i)
        else:
            name = states_2_name(msm.int_2_s[i])
        G.add_node(i, label=name)
        K_number = msm.int_2_s[i][0][:5].count("K") + msm.int_2_s[i][0][:5].count("C")
        node_colors.append(colors[K_number])

    # prepare edge
    e_list = []
    for i in range(node_num):
        for j in range(node_num):
            if net_t_matrix[i, j] > edge_cutoff:
                G.add_edge(i, j, weight=net_t_matrix[i, j])
                e_list.append((i, j))

    # prepare position of each node, if position is not given, use spring_layout
    if position is None:
        node_positions = {}
        for i in range(node_num):
            node_positions[i] = list(computer_pos(msm.int_2_s[i], end=5))
        node_positions = nx.spring_layout(G, pos=node_positions, iterations=iterations, k=k)
    else:
        node_positions = position

    # prepare node size
    node_sizes = []
    counter = msm.node_counter
    frame_sum = sum(counter.values())
    for i in range(node_num):
        node_sizes.append(counter[i] / frame_sum * 4000)

    # plt

    nx.draw_networkx_nodes(G, ax=ax, pos=node_positions, node_color=node_colors, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_labels(G, ax=ax, pos=node_positions, labels=nx.get_node_attributes(G, 'label'),
                            font_family='monospace')
    width = [nx.get_edge_attributes(G, 'weight')[i] / edge_factor for i in G.edges()]
    nx.draw_networkx_edges(G, ax=ax, pos=node_positions, width=width, connectionstyle='arc3,rad=0.05',
                           alpha=0.5, arrowsize=15, node_size=np.array(node_sizes) * 2.0)
    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax,
                                 bbox=text_bbox
                                 )

    return e_list, node_positions


def letter_code_pos(string):
    """
    compute a position for a string
    The string can only contain "0", "K", "W", "C".
    The number of "K" and "C" is the x coordinate, and the center of mass of the string is the y coordinate.
    """
    weight_dict = {"K": 1,
                   "W": 0.2,
                   "C": 1.5,
                   "0": 0, }
    center_mass = 0
    total_mass = 0
    for i, site in enumerate(string[::-1]):
        center_mass += weight_dict[site] * i
        total_mass += weight_dict[site]
    center_mass /= total_mass
    x = string.count("K") + string.count("C")
    return x, center_mass


def computer_pos(strings, end=None):
    """
    compute a position for a list of strings
    The string can only contain "0", "K", "W", "C".
    The number of "K" and "C" is the x coordinate, and the center of mass of the string is the y coordinate.
    """
    if isinstance(strings, str):
        return letter_code_pos(strings[:end:])
    elif isinstance(strings, list):
        coord = np.array([letter_code_pos(s[:end:]) for s in strings])
        x = np.mean(coord[:, 0])
        y = np.mean(coord[:, 1])
        return x, y


class Graph:
    def __init__(self, model: SF_msm):
        """
        :param model: Sfilter.MSM.SF_msm
        This is the class to plot the mechanism graph.
        """
        self.model = model
        self.G = None
        self.population_cutoff = None
        self.node_color = "#1f78b4"

    def set_node_from_population(self, population_cutoff=0.01):
        """
        set the node based on population.
        This function will reset the self.G and add nodes which has the population above population_cutoff.
        :param population_cutoff: float
        :return: None
        """
        self.G = nx.DiGraph()
        n = self.model.population_cutoff(population_cutoff)
        for i in range(n):
            self.G.add_node(i)
        self.population_cutoff = population_cutoff

    def sef_node_color(self, color_list=None):
        """
        set the color of nodes.
        :param color_list: a list of color for each node
        :return: None
        """
        if color_list is None:
            # set node color based on the number of K and C, and use matplotlib default color cycle
            color_list = []
            for node in self.G.nodes:
                K_number = self.model.int_2_s[node][0][:5].count("K") + self.model.int_2_s[node][0][:5].count("C")
                K_number = (K_number + 8) % 10
                color_list.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][K_number])
        if len(color_list) != len(self.G.nodes):
            raise ValueError("color_list must have the same length as the number of nodes")
        self.node_color = color_list

    def set_node_size(self, size_dict=None):
        """
        set the size of nodes. If size_dict is None, use the population of each state as the size.
        :param size_dict: a dictionary from node index to size
        :return: None
        """
        total = self.model.node_counter.total()
        if size_dict is None:
            size_dict = {i: self.model.node_counter[i] / total * 100 for i in self.G.nodes}
        nx.set_node_attributes(self.G, size_dict, "size")

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
                node = self.model.int_2_s[i]
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

    def add_weighted_edges_from_net_F(self, cut_off, net_F=None):
        """
        set the edge based on net flux.
        Net flux above cut_off will be added.
        :param cut_off: Net flux cutoff
        :param net_F: net flux matrix, if not given, will recalculate from model.get_matrix()
        :return: None
        """
        if cut_off < 0:
            raise ValueError("cut_off must be positive")

        if net_F is None:
            f_matrix, net_F, r_matrix, p_matrix = self.model.get_matrix()
        for i in self.G.nodes:
            for j in self.G.nodes:
                if i > j:
                    if net_F[i, j] > net_F[j, i] and net_F[i, j] > cut_off:
                        self.G.add_edge(i, j, weight=self.model.net_t_matrix[i, j])
                    elif net_F[i, j] <= net_F[j, i] and net_F[j, i] > cut_off:
                        self.G.add_edge(j, i, weight=self.model.net_t_matrix[j, i])

    def add_weighted_edges_from_rate_nfp(self, cut_off, rate_matrix=None):
        """
        set the edge based on the rate (number of first passage / observation time).
        Rate above cut_off will be added.
        :param cut_off: Rate cutoff
        :param rate_matrix: rate matrix, if not given, will recalculate from model.get_nfp_rate_matrix()
        :return: None
        """

        if rate_matrix is None:
            rate_matrix, FP_list = self.model.get_nfp_rate_matrix(population_cutoff=self.population_cutoff)
        for i in self.G.nodes:
            for j in self.G.nodes:
                if i > j:
                    if rate_matrix[i, j] > rate_matrix[j, i] and rate_matrix[i, j] > cut_off:
                        self.G.add_edge(i, j, weight=rate_matrix[i, j])
                    elif rate_matrix[i, j] <= rate_matrix[j, i] and rate_matrix[j, i] > cut_off:
                        self.G.add_edge(j, i, weight=rate_matrix[j, i])

    def add_weighted_edges_from_rate_nfp_netF(self, rate_cut_off, net_F_cut_off=0, rate_matrix=None, net_F=None,
                                              label_format="{:d}"):
        """
        Add edge based on rate and net flux.
        if a positive net flux has a rate above rate_cut_off, add the edge.
        The weight of the edge is the rate.
        The label of the edge is the net flux.
        If you do this twice, all of the edges will be overwritten.
        :param rate_cut_off: minimum rate to add an edge
        :param net_F_cut_off: minimum net flux to add an edge
        :param rate_matrix: rate matrix, if not given, will recalculate from model.get_nfp_rate_matrix()
        :param net_F: net flux matrix, if not given, will recalculate from model.get_matrix()
        :param label_format: format of the label. default is "{:d}"
        :return: None
        """
        if rate_matrix is None:
            rate_matrix, FP_list = self.model.get_nfp_rate_matrix(population_cutoff=self.population_cutoff)
        if net_F is None:
            f_matrix, net_F, _r_matrix, p_matrix = self.model.get_matrix()
        self.G.remove_edges_from(list(self.G.edges))
        for i in self.G.nodes:
            for j in self.G.nodes:
                if i > j:
                    if net_F[i, j] > net_F_cut_off and rate_matrix[i, j] > rate_cut_off:
                        self.G.add_edge(i, j, weight=rate_matrix[i, j])
                        self.G.edges[i, j]["label"] = label_format.format(net_F[i, j])
                    elif net_F[j, i] > net_F_cut_off and rate_matrix[j, i] > rate_cut_off:
                        self.G.add_edge(j, i, weight=rate_matrix[j, i])
                        self.G.edges[j, i]["label"] = label_format.format(net_F[j, i])

    def add_edge_label_from_net_F(self, net_F=None, label_format="{:.2e}"):
        """
        add label to existing edges based on net flux.
        :return: None
        """
        if net_F is None:
            f_matrix, net_F, r_matrix, p_matrix = self.model.get_matrix()
        for i, j in self.G.edges:
            # only modify the label
            self.G.edges[i, j]["label"] = label_format.format(net_F[i, j])

    def add_edge_label_from_rate_nfp(self, rate_matrix=None, label_format="{:.2e}"):
        """
        add label to existing edges based on rate.
        :return: None
        """
        if rate_matrix is None:
            rate_matrix, FP_list = self.model.get_nfp_rate_matrix(population_cutoff=self.population_cutoff)
        for i, j in self.G.edges:
            # only modify the label
            self.G.edges[i, j]["label"] = label_format.format(rate_matrix[i, j])

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
            p_i = [str_2_xy(si) for si in self.model.int_2_s[i]]
            p_i = np.mean(p_i, axis=0)
            position[i] = p_i
        return position

    def draw_grid(self, ax, grid=0.1):
        if grid:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            # round to a multiple of f
            xmin = np.ceil(xmin / grid) * grid
            ymin = np.ceil(ymin / grid) * grid
            ax.set_xticks(np.arange(xmin, xmax, grid))
            ax.set_yticks(np.arange(ymin, ymax, grid))
            ax.grid()

    def draw_dirty(self, ax, node_size_factor=1.0, node_alpha=0.7, edge_alpha=0.5,
                   spring_iterations=5, spring_k=10,
                   edge_factor=None, pos=None,
                   label_bbox=None):
        """

        :param ax: matplotlib.axes
        :param node_size_factor: float, set it larger to make the node larger, default is 1.0
        :param edge_factor: float, set it larger to make the edge wider, default is 0.1/min_weight,
            which means the thinnest edge will be 0.1.
        :param pos: position, a dictionary from node index to position.
            If not given, use nx.spring_layout to guess the initial position.
        :param label_bbox: a dictionary, the bbox for edge label
        :return: position, a dictionary from node index to position
        """
        if label_bbox is None:
            label_bbox = {"boxstyle": "round", "ec": (1.0, 1.0, 1.0, 0), "fc": (1.0, 1.0, 1.0, 0.5)}
        if pos is None:
            # guess the initial position and optimize
            pos = nx.spring_layout(self.G, iterations=spring_iterations, k=spring_k, pos=self.guess_init_position())
        if edge_factor is None:
            edge_factor = 0.1 / min([self.G.edges[i, j]["weight"] for i, j in self.G.edges])

        # draw nodes
        node_sizes = [self.G.nodes[i]["size"] * 50 * node_size_factor for i in self.G.nodes]
        nx.draw_networkx_nodes(self.G, ax=ax, pos=pos, node_size=node_sizes,
                               node_color=self.node_color, alpha=node_alpha)
        nx.draw_networkx_labels(self.G, ax=ax, pos=pos, labels=nx.get_node_attributes(self.G, "label"),
                                font_family='monospace')

        # draw edges
        edge_width = [self.G.edges[i, j]["weight"] * edge_factor for i, j in self.G.edges]
        nx.draw_networkx_edges(self.G, ax=ax, pos=pos, width=edge_width, connectionstyle='arc3,rad=0.05',
                               alpha=edge_alpha, node_size=np.array(node_sizes) * 2.0)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=nx.get_edge_attributes(self.G, 'label'), ax=ax,
                                     bbox=label_bbox)

        return pos, edge_factor

    def lj_potential(self, pos, sigma=0.2, epsilon=0.5):
        """
        compute the Lennard-Jones potential between every pair of nodes and edge.
        :param pos: position, a dictionary from node index to position
        :param sigma:
        :param epsilon:
        :return: energy, float
        """
        # Loop through all unique pairs of nodes/labels
        energy = 0
        position_list = []
        for i in self.G.nodes:
            position_list.append(np.array(pos[i]))
        for i, j in self.G.edges:
            position_list.append((np.array(pos[i]) + np.array(pos[j])) / 2)
        for i, pos_i in enumerate(position_list):
            for j, pos_j in enumerate(position_list):
                if i > j:
                    energy += lj_pair(pos_i, pos_j, sigma, epsilon)
        return energy

    def coulomb_potential(self, pos,  distance_cutoff=0.2):
        """
        compute the potential between every pair of nodes and edge.
        E = 1/r**3
        :param pos: position, a dictionary from node index to position
        :param distance_cutoff: float, interaction outside this distance will not be considered
        :return: energy, float
        """
        # Loop through all unique pairs of nodes/labels
        energy = 0
        position_list = []
        cut_off_energy = 1 / (distance_cutoff)*3 / 1000
        for i in self.G.nodes:
            position_list.append(np.array(pos[i]))
        for i, j in self.G.edges:
            position_list.append((np.array(pos[i]) + np.array(pos[j])) / 2)
        for i, pos_i in enumerate(position_list):
            for j, pos_j in enumerate(position_list):
                if i > j:
                    distance = np.linalg.norm(pos_i - pos_j)
                    if distance > distance_cutoff:
                        energy += cut_off_energy
                    else:
                        energy += 1 / (np.linalg.norm(pos_i - pos_j))*3 / 1000
        return energy

    def optimize_positions(self, initial_pos=None, maxiter=5, distance_cutoff=0.1):
        """
        Waring, This potential only separate things away.
        optimize the position of nodes to minimize the Lennard-Jones potential.
        :param initial_pos: position, a dictionary from node index to position
        :return: position, a dictionary from node index to position
        """
        if initial_pos is None:
            initial_pos = nx.spring_layout(self.G, iterations=5, pos=self.guess_init_position())
        pos_array = np.array([initial_pos[node] for node in self.G.nodes()]).flatten()

        def opject_function(pos_array):
            pos_dict = {node: pos_array[2 * i : 2 * i + 2] for i, node in enumerate(self.G.nodes())}
            energy = self.coulomb_potential(pos_dict, distance_cutoff=distance_cutoff)
            return energy

        result = minimize(opject_function, pos_array,
                          options={"maxiter": maxiter})
        pos_array = result.x
        optimized_pos = {node: pos_array[2 * i : 2 * i + 2] for i, node in enumerate(self.G.nodes())}
        return optimized_pos


def lj_pair(xy0, xy1, sigma: float, epsilon: float):
    """
    compute the Lennard-Jones potential between two coordinate
    :param xy0: position of node 0
    :param xy1: position of node 1
    :return: energy, float
    """
    r = np.linalg.norm(np.array(xy0) - np.array(xy1))
    s6 = (sigma / r) ** 6
    return 4 * epsilon * (s6 ** 2 - s6)
