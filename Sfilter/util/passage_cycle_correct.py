import warnings
import numpy as np
import pandas as pd
from .output_wrapper import read_k_cylinder


class Passage_cycle_correct:
    def __init__(self, traj_raw_alltraj, passage_time_length_alltraj_raw, passage_time_point_alltraj_raw, jump_array_alltraj, time_step):
        """
        :param traj_raw_alltraj:
            A list of np.array
        :param passage_time_length_alltraj_raw:
            A list of 2D list
                Each 2D list is a np.array for a simulation replica.
                    Each np.array is the passage time length of state i to j.
        :param passage_time_point_alltraj_raw:
            A list of 2D list
                Each 2D list is a np.array for a simulation replica.
                    Each np.array is the passage time finishing point of state i to j.
        :param jump_array_alltraj:
            A list of np.array
                Each np.array is for a simulation replica. the integer number is how many jumps from the previous frame
                to the current frame.

        :return: None
        """
        self.traj_raw_alltraj = traj_raw_alltraj
        self.passage_time_length_alltraj_raw = passage_time_length_alltraj_raw
        self.passage_time_point_alltraj_raw = passage_time_point_alltraj_raw
        self.jump_array_alltraj = jump_array_alltraj
        self.time_step = time_step
        self.passage_ij_dict = {}
        # this dictionary saves the rate_ij result, so that we don't need to calculate the same rate_ij again.
        # Only when a new rate_ij is needed, we calculate it.
        # key: (state_i, state_j)
        # value: passage_ij (a dictionary)
        #     key: number of jumps
        #     value: (length, start, end) in each replica

    def get_passage_ij(self, state_i, state_j):
        """
        Get all the passage time length from state i to j. Group the passage by the number of jumps.
        :param state_i: int
        :param state_j: int
        :return:
            A dictionary
                key: number of jumps
                value: (length, start, end)
                    length[rep][k] is the k-th passage time length in replica rep.
                    start[rep][k] is the k-th passage time starting point in replica rep.
                    end[rep][k] is the k-th passage time ending point in replica rep.
        """
        if (state_i, state_j) in self.passage_ij_dict:
            return self.passage_ij_dict[(state_i, state_j)]
        else:
            rep_n = len(self.passage_time_length_alltraj_raw)
            p_length_alltraj = [self.passage_time_length_alltraj_raw[rep][state_i][state_j] for rep in range(rep_n)]
            p_end_alltraj = [self.passage_time_point_alltraj_raw[rep][state_i][state_j] for rep in range(rep_n)]
            p_start_alltraj = [p_end_alltraj[rep] - p_length_alltraj[rep]   for rep in range(rep_n)]
            # the jump of a passage is sum(jump_array_alltraj[rep][start:end])
            p_jump_alltraj = [[] for i in self.jump_array_alltraj]
            for rep, (jump_array, p_start, p_end) in enumerate(zip(self.jump_array_alltraj, p_start_alltraj, p_end_alltraj)):
                for start, end in zip(p_start, p_end):
                    p_jump_alltraj[rep].append(sum(jump_array[start+1:end+1]))
            # group the passage by the number of jumps
            p_result_group = {}
            for rep in range(rep_n):
                for k, (length, start, end, jump) in enumerate(zip(p_length_alltraj[rep],
                                                                   p_start_alltraj[rep],
                                                                   p_end_alltraj[rep],
                                                                   p_jump_alltraj[rep])):
                    if jump not in p_result_group:
                        p_result_group[jump] = ([[] for i in range(rep_n)],
                                                [[] for i in range(rep_n)],
                                                [[] for i in range(rep_n)], )
                    p_result_group[jump][0][rep].append(length)
                    p_result_group[jump][1][rep].append(start)
                    p_result_group[jump][2][rep].append(end)
            self.passage_ij_dict[(state_i, state_j)] = p_result_group
            return p_result_group

    def data_to_df(self):
        """
        Covert the input data to a pd.DataFrame
        | index | replica | traj | jump |
        Every column is a np.array in int.
        """
        replica_list = [ [rep] * len(self.jump_array_alltraj[rep]) for rep in range(len(self.jump_array_alltraj)) ]
        replica = np.concatenate(replica_list)
        traj = np.concatenate(self.traj_raw_alltraj)
        jump = np.concatenate(self.jump_array_alltraj)

        return pd.DataFrame({"replica": replica, "traj": traj, "jump": jump})



    def result_to_df(self, output_dir):
        """
        Convert self.passage_ij_dict to a pd.DataFrame
        """
        pass





    def get_mfpt_ij(self, state_i, state_j):
        """

        :param state_i:
        :param state_j:
        :return:
        """
        pass



