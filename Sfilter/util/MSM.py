import copy
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from .output_wrapper import read_k_cylinder





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
        for traj, meta_data in [read_k_cylinder(file, method) for file in file_list]:
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
        compute transition matrix
        a numpy array, each element is the number of transition from state i to state j
        """""
        state_num = max([max(traj) for traj in self.state_array])
        tran_matrix = np.zeros((state_num + 1, state_num + 1), dtype=np.int64)
        for traj in self.state_array:
            state_start = traj[:-lag_step]
            state_end = traj[lag_step:]
            for m_step in np.array([state_start, state_end]).T:
                tran_matrix[m_step[0], m_step[1]] += 1
        return tran_matrix

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

        t_matrix = self.get_transition_matrix(lag_step)
        rate_matrix = np.array(t_matrix, dtype=np.float64)
        for i in range(rate_matrix.shape[0]):
            rate_matrix[i, :] /= self.node_counter[i] * physical_time
            rate_matrix[i, i] = 0
        return rate_matrix

    def get_transition_probability(self, lag_step=1):
        """
        compute transition probability
        return: transition_probability_matrix
            a numpy array, each element is the probability of transition from state i to state j
            The sum of each row is 1.
        """
        t_matrix = self.get_transition_matrix(lag_step)
        p_matrix = np.array(t_matrix, dtype=np.float64)
        p_matrix /= p_matrix.sum(axis=1, keepdims=True)  # normalize each row
        return p_matrix

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
            reality.append(self.get_transition_probability(t*lag_step))
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
        compute mean first passage time from state A to state B
        A -> B
        alg1:
        B A-A C A-A C-C-C B-B
          ^               ^  8 steps
        :param A: int, node A
        :param B: int, node B
        :return:
            MFPT_list, a list of FPT from state A to state B, unit in step
        """
        MFPT_list = []
        for traj in self.state_array:
            MFPT_list += MFPT_A_to_B(traj, A, B)
        return MFPT_list

    def get_MFPT_matrix(self, population_cutoff=0.01):
        """
        compute mean first passage time for every node pairs above population_cutoff
        Multiple algorithms are available,
        A -> C
        alg1:
        C A-A B A-A B-B-B C-C
          ^               ^  8 steps
        :param lag_step:
        :param population_cutoff:
        :return:
            MFPT_matrix, a numpy array, each element is the MFPT from state i to state j, unit in step*physical_time_step
            FPT_list, each element is a list of FPT from state i to state j, unit in step
        """
        # find the node that is above the cutoff
        total_count = self.node_counter.total()
        for n, node_count in self.node_counter.items():
            if node_count / total_count < population_cutoff:
                break
        MFPT_matrix = np.zeros((n, n), dtype=np.float64)
        FPT_list = []
        for i in range(n):
            FPT_list.append([])
            for j in range(n):
                if i == j:
                    continue
                fpt_ij = self.get_MFPT_pair(i, j)
                FPT_list[i].append(fpt_ij)
                MFPT_matrix[i, j] = np.mean(fpt_ij) * self.time_step[0]
        return MFPT_matrix, FPT_list

    def get_matrix(self, lag_step=1, physical_time=None):
        """
        calculate transition matrix, rate_matrix, and transition probability
        return:
            t_matrix: each element is the number of transition from state i to state j
            net_t_matrix: each element is the number of net event between state i to state j (t_matrix - t_matrix.T)
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

        t_matrix = self.get_transition_matrix(lag_step)
        net_t_matrix = t_matrix - t_matrix.T
        rate_matrix = np.array(t_matrix, dtype=np.float64)
        for i in range(rate_matrix.shape[0]):
            rate_matrix[i, :] /= self.node_counter[i] * physical_time
            rate_matrix[i, i] = 0
        p_matrix = np.array(t_matrix, dtype=np.float64)
        p_matrix /= p_matrix.sum(axis=1, keepdims=True)
        return t_matrix, net_t_matrix, rate_matrix, p_matrix

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

    def find_merge_states(self, cut_off=0.01, lag_step=1, physical_time=None, method="rate_square"):
        """
        rank states pairs with ( (rate_ij + rate_ji) * (T_ij + T_ji) )
        input:
            cut_off: the cut-off of the distribution of states, states with occurrence less than cut_off will be ignored
            lag_step: lag time in step
            physical_time: physical time for each step, if not given, use the time_step that was read from file
        return: f_list, (t_matrix, net_t_matrix, rate_matrix, p_matrix), node_num
            f_list: a list of [state_i, state_j, object_function]
            matrices: (t_matrix, net_t_matrix, rate_matrix, p_matrix)
            node_num: the final states with occurrence larger than cut_off
        """
        if physical_time is None:  # use the time_step that was read from file
            if np.allclose(self.time_step, self.time_step[0]):
                physical_time = self.time_step[0]
            else:
                raise ValueError("physical_time is not given, and time_step is not equal")

        t_matrix, net_t_matrix, rate_matrix, p_matrix = self.get_matrix(lag_step=lag_step, physical_time=physical_time)
        for node_num in range(len(self.int_2_s)):
            if self.distribution[node_num] <= cut_off:
                node_num -= 1
                break
                # node_num: the final states with occurrence larger than cut_off
        f_list = []
        if method == "rate_T":
            for i in range(node_num + 1):
                for j in range(i):
                    f_list.append([j, i,
                                   (rate_matrix[i, j] + rate_matrix[j, i])
                                   * (t_matrix[i, j] + t_matrix[j, i])])
        elif method == "rate":
            for i in range(node_num + 1):
                for j in range(i):
                    f_list.append([j, i,
                                   (rate_matrix[i, j] + rate_matrix[j, i])])
        elif method == "rate_square":
            for i in range(node_num + 1):
                for j in range(i):
                    f_list.append([j, i,
                                   (rate_matrix[i, j] * rate_matrix[j, i])])
        elif method == "rate_triangle":
            for i in range(node_num + 1):
                for j in range(i):
                    if rate_matrix[i, j] * rate_matrix[j, i] == 0:
                        f_list.append([j, i, 0])
                    else:
                        f_list.append([j, i,
                                       rate_matrix[i, j] * rate_matrix[j, i]
                                       * min(rate_matrix[i, j], rate_matrix[j, i])
                                       / max(rate_matrix[i, j], rate_matrix[j, i])
                                       ])
        else:
            raise ValueError("method must be rate_T, rate, or rate_square")
        f_list = sorted(f_list, key=lambda x: x[2], reverse=True)
        return f_list, (t_matrix, net_t_matrix, rate_matrix, p_matrix), node_num

    def merge_until(self, rate_cut_off, rate_square_cut_off, node_cut_off=0.01, step_cut_off=30, lag_step=1, physical_time=None,
                    method="rate_square", min_node=3):
        """
        Merge states until a certain cutoff
        I suggest to set rate_cut_off as 0.5/physical_time or smaller
        input:
            rate_cut_off: the maximum rate of the merged states
            rate_square_cut_off: the maximum rate_square of the merged states
                only when both rate and rate_square are smaller than the cut_off iteration will stop
            node_cut_off: Nodes with occurrence less than cut_off will be ignored.
            lag_step: The number of steps that is used to calculate all matrix
            physical_time: physical time for each step, if not given, use the time_step that was read from file
            method: "rate_T", "rate", or "rate_square"
                rate_square is recommended
            min_node: the minimum number of nodes that will stop iteration.
        """
        if physical_time is None:  # use the time_step that was read from file
            if np.allclose(self.time_step, self.time_step[0]):
                physical_time = self.time_step[0]
            else:
                raise ValueError("physical_time is not given, and time_step is not equal")

        merge_step = 0
        print(" Step,    rate^2,      rate,  rate_tri,")
        while True:
            # get state pairs with the largest (rate_ij + rate_ji) * (T_ij + T_ji)
            m_list, (t_mat, net_t_mat, rate_mat, p_mat), node_num = self.find_merge_states(
                cut_off=node_cut_off,
                lag_step=lag_step,
                physical_time=physical_time,
                method=method)

            # check convergence
            if node_num + 1 <= min_node:
                reason = "minimum node reached"
                break
            n_0, n_1, _ = m_list[0]
            r_square = rate_mat[n_0, n_1] * rate_mat[n_1, n_0]
            max_rate = np.max(rate_mat[:node_num + 1, :node_num + 1])
            if max_rate < rate_cut_off and r_square < rate_square_cut_off:
                reason = "rate and rate^2 cut off reached"
                break
            if len(m_list) == 1:
                reason = "no more merging"
                break
            if merge_step > step_cut_off:
                reason = "step cut off reached"
                break
            # merge states
            # n_0, n_1, _ = m_list[0]
            merge_step += 1
            rate_triangle = rate_mat[n_0, n_1] * rate_mat[n_1, n_0] * min(rate_mat[n_0, n_1], rate_mat[n_1, n_0]) \
                            / max(rate_mat[n_0, n_1], rate_mat[n_1, n_0])
            print(f"{merge_step:5d}, {r_square:9.5f}, {max_rate:9.4f}, {rate_triangle:9.4f}, {self.int_2_s[n_0]}+{self.int_2_s[n_1]}")
            if len(self.int_2_s[n_0]) > 1 and len(self.int_2_s[n_1]) > 1:
                merge_list_new = copy.deepcopy(self.merge_list)
                for node in self.merge_list:
                    for node2 in self.merge_list:
                        if node == self.int_2_s[n_0] and node2 == self.int_2_s[n_1]:
                            merge_list_new.remove(node)
                            merge_list_new.remove(node2)
                            merge_list_new.append(node + node2)
                self.merge_list = merge_list_new
            elif len(self.int_2_s[n_0]) > 1 or len(self.int_2_s[n_1]) > 1:
                merge_list_new = []
                for node in self.merge_list:
                    if node == self.int_2_s[n_0]:
                        merge_list_new.append(node + self.int_2_s[n_1])
                    elif node == self.int_2_s[n_1]:
                        merge_list_new.append(self.int_2_s[n_0] + node)
                    else:
                        merge_list_new.append(node)
                self.merge_list = merge_list_new
            elif len(self.int_2_s[n_0]) == 1 or len(self.int_2_s[n_1]) == 1:
                self.merge_list.append(self.int_2_s[n_0] + self.int_2_s[n_1])
            else:
                raise ValueError("merge error " + str(n_0) + str(n_1))
            self.calc_state_array(merge_list=self.merge_list)
        print("#" * 60)
        print(f"Converged")
        print(f"  Maximum r^2  : {r_square:.3f}")
        print(f"  Maximum rate : {max_rate:.3f}")
        return reason


def states_2_name(node):
    """
    Convert a list of states to a string
    """
    string = ""
    for s in node[:-1]:
        string += s
        string += "\n"
    string += node[-1]
    return string


def plot_net_T_matrix(ax, msm, cut_off, edge_cutoff, net_t_matrix, iterations=6,k=15,
                      colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
                      text_bbox={"boxstyle": "round", "ec": (1.0, 1.0, 1.0), "fc": (1.0, 1.0, 1.0)}
                      ):
    """
    Wrap up function to plot net transition matrix
    """
    G = nx.DiGraph()
    # cutoff at certain density
    for node_num in range(len(msm.int_2_s)):
        if msm.distribution[node_num] < cut_off:
            break

    # prepare node with name and color
    node_colors = []
    for i in range(node_num):
        name = states_2_name(msm.int_2_s[i])
        G.add_node(i, label=name)
        K_number = msm.int_2_s[i][0].count("K") + msm.int_2_s[i][0].count("C")
        node_colors.append(colors[K_number])

    # prepare edge
    e_list = []
    for i in range(node_num):
        for j in range(node_num):
            if net_t_matrix[i, j] > edge_cutoff:
                G.add_edge(i, j, weight=net_t_matrix[i, j])
                e_list.append((i, j))

    # prepare initial position of each node
    node_positions = {}
    for i in range(node_num):
        node_positions[i] = list(computer_pos(msm.int_2_s[i]))
    node_sizes = []
    counter = msm.node_counter
    frame_sum = sum(counter.values())
    for i in range(node_num):
        node_sizes.append(counter[i] / frame_sum * 4000)
    node_positions = nx.spring_layout(G, pos=node_positions, iterations=iterations, k=k)

    # plt

    nx.draw_networkx_nodes(G, ax=ax, pos=node_positions, node_color=node_colors, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_labels(G, ax=ax, pos=node_positions, labels=nx.get_node_attributes(G, 'label'),
                            font_family='monospace')
    width = [nx.get_edge_attributes(G, 'weight')[i] / 100 for i in G.edges()]
    nx.draw_networkx_edges(G, ax=ax, pos=node_positions, width=width, connectionstyle='arc3,rad=0.05',
                           alpha=0.5, arrowsize=15, node_size=np.array(node_sizes) * 2.5)
    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax,
                                 bbox=text_bbox
                                 )

    return e_list


def letter_code_pos(string):
    """
    compute a position for a string
    The string can only contain "0", "K", "W", "C".
    The number of "K" and "C" is the x coordinate, and the center of mass of the string is the y coordinate.
    """
    weight_dict = {"K": 1,
                   "W": 0.5,
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


def computer_pos(strings):
    """
    compute a position for a list of strings
    The string can only contain "0", "K", "W", "C".
    The number of "K" and "C" is the x coordinate, and the center of mass of the string is the y coordinate.
    """
    if isinstance(strings, str):
        return letter_code_pos(strings)
    elif isinstance(strings, list):
        coord = np.array([letter_code_pos(s) for s in strings])
        x = np.mean(coord[:, 0])
        y = np.mean(coord[:, 1])
        return x, y


def get_transition_matrix(state_arrays, begin=0, lag_time=1):
    state_num = max([max(traj) for traj in state_arrays])
    tran_matrix = np.zeros((state_num + 1, state_num + 1), dtype=np.int64)
    for traj in state_arrays:
        state_start = traj[begin:-lag_time]
        state_end = traj[begin + lag_time:]
        for m_step in np.array([state_start, state_end]).T:
            tran_matrix[m_step[0], m_step[1]] += 1
    return tran_matrix


def get_distribution(state_arrays):
    flattened = [num for sublist in state_arrays for num in sublist]
    return Counter(flattened)


def get_rate_matrix(state_arrays, phy_time, begin=0, lag_time=1):
    tran_matrix = np.array(get_transition_matrix(state_arrays, begin, lag_time), dtype=np.float64)
    counter = get_distribution(state_arrays)
    for i in range(tran_matrix.shape[0]):
        tran_matrix[i, :] /= counter[i] * phy_time
    return tran_matrix
