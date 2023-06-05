#!/usr/bin/env python3

import argparse


class PermEvent:
    def __init__(self, at_index: int, time, up=True):
        self.at_index = at_index
        self.time = time  # 1 number or a list of 2 numbers
        self.resident_time = None
        self.up = up

    def __str__(self):
        s = "Time " + str(self.time)
        s += " %d " % self.at_index
        if self.up:
            s += "up"
        else:
            s += "down"
        return s

    def __repr__(self):
        return self.__str__()

    def set_resident_time(self, resident_time):
        self.resident_time = resident_time


def read_xtck_perm_up_down(perm_up, perm_down):
    """

    :param perm_up: file
    :param perm_down: file
    :return: a list of PermEvent
    """
    perm_list = []
    with open(perm_up) as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            e = PermEvent(int(words[1]), float(words[0]), True)
            perm_list.append(e)

    with open(perm_down) as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            e = PermEvent(int(words[1]), float(words[0]), False)
            perm_list.append(e)
    perm_list = sorted(perm_list, key=lambda x: x.time)
    return perm_list


def read_cylinder(cylinder_file):
    """
    :param cylinder_file:
    :return: a list of PermEvent
    """
    perm_list = []
    with open(cylinder_file) as f:
        for line in f:
            if ("Perm: [4 1 3]" in line):
                line = line.rstrip()
                words = line.split()
                at_index = int(words[4])
                time = [float(words[10]), float(words[11])]
                perm_list.append(PermEvent(at_index, time))
            elif ("Perm: [5 1 3]" in line):
                line = line.rstrip()
                words = line.split()
                at_index = int(words[4])
                time = [0, float(words[11])]
                perm_list.append(PermEvent(at_index, time))
            elif ("Perm: [4 5 1]" in line):
                line = line.rstrip()
                words = line.split()
                at_index = int(words[4])
                time = [float(words[10]), float(words[12]) * 1.05]
                perm_list.append(PermEvent(at_index, time))

    perm_list = sorted(perm_list, key=lambda x: x.time[0])
    return perm_list


def read_cylinder_Sfilter(file):
    """
        :param cylinder_file:
        :return: a list of PermEvent
        """
    perm_list = []
    with open(file) as f:
        count_line = -1
        for line in f:
            if "Permeation up 4 -> 1 -> 3" in line:
                count_line = 0
                perm_up = True
            elif "Permeation up 3 -> 1 -> 4" in line:
                count_line = 0
                perm_up = False
            elif line == "\n" or "None" in line:
                count_line = -1

            if count_line >= 2:
                words = line.split(",")
                event = PermEvent(int(words[0]),
                                  float(words[2]),
                                  perm_up
                                  )
                event.set_resident_time(float(words[3]))
                perm_list.append(event)
            if count_line >= 0:
                count_line += 1
    return perm_list


def match_xtck_cylinder_sfilter(xtck: PermEvent, cyl: PermEvent):
    """
    test if the xtck/cylinder_count permeation event is the same event
    :param xtck:
    :param cyl:
    :return: bool
    """
    if xtck.at_index - 1 == cyl.at_index:
        if xtck.up == cyl.up:
            if cyl.time > xtck.time and cyl.time - cyl.resident_time < xtck.time:
                return True
    return False


class time_seq_list:
    def __init__(self, element):
        self.index = tuple(range(len(element)))
        self.element = element

    def pop(self):
        return time_seq_list(self.element[:-1])

    def __len__(self):
        return len(self.index)


def longest_common_subsequence(seq1: time_seq_list, seq2: time_seq_list, dp_dict, compare):
    """
    get the longest common subsequence using dynamic programming
    :param seq1: time_seq_list
    :param seq2: time_seq_list
    :param compare:
    :return: length: int, lcs_list
    """
    if (seq1.index, seq2.index) in dp_dict:
        return dp_dict[(seq1.index, seq2.index)]
    elif len(seq1) == 0:
        LCS_list = []
        for i in seq2.element:
            LCS_list.append(["None", i])
        dp_dict[(seq1.index, seq2.index)] = (0, LCS_list)
        return (0, LCS_list)
    elif len(seq2) == 0:
        LCS_list = []
        for i in seq1.element:
            LCS_list.append([i, "None"])
        dp_dict[(seq1.index, seq2.index)] = (0, LCS_list)
        return (0, LCS_list)
    elif compare(seq1.element[-1], seq2.element[-1]):
        lcs_len, LCS_list = longest_common_subsequence(seq1.pop(), seq2.pop(), dp_dict, compare)
        lcs_len += 1
        LCS_list.append([seq1.element[-1], seq2.element[-1]])
        dp_dict[(seq1.index, seq2.index)] = (lcs_len, LCS_list)
        return lcs_len, LCS_list
    else:
        if (seq1.pop().index, seq2.index) in dp_dict:
            lcs_len1, LCS_list1 = dp_dict[(seq1.pop().index, seq2.index)]
        else:
            lcs_len1, LCS_list1 = longest_common_subsequence(seq1.pop(), seq2, dp_dict, compare)
            dp_dict[(seq1.pop().index, seq2.index)] = (lcs_len1, LCS_list1)

        if (seq1.index, seq2.pop().index) in dp_dict:
            lcs_len2, LCS_list2 = dp_dict[(seq1.index, seq2.pop().index)]
        else:
            lcs_len2, LCS_list2 = longest_common_subsequence(seq1, seq2.pop(), dp_dict, compare)
            dp_dict[(seq1.index, seq2.pop().index)] = (lcs_len2, LCS_list2)

        if lcs_len1 > lcs_len2:
            lcs_len = lcs_len1
            LCS_list = LCS_list1 + [[seq1.element[-1], "None"]]
        else:
            lcs_len = lcs_len2
            LCS_list = LCS_list2 + [["None", seq2.element[-1]]]
        dp_dict[(seq1.index, seq2.index)] = (lcs_len, LCS_list)
        return lcs_len, LCS_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-perm_up",
                        dest="perm_up",
                        help="perm_up.dat from xtck",
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument("-perm_down",
                        dest="perm_down",
                        help="perm_down.dat from xtck",
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument("-cylinderS",
                        dest="cylinder",
                        help="K_perm_event from Chenggong",
                        type=argparse.FileType('r'),
                        required=True)
    args = parser.parse_args()
    xtck_list = read_xtck_perm_up_down(args.perm_up.name, args.perm_down.name)
    cyl_list = read_cylinder_Sfilter(args.cylinder.name)
    s1 = time_seq_list(xtck_list)
    s2 = time_seq_list(cyl_list)
    l, s_list = longest_common_subsequence(s1, s2, {}, compare=match_xtck_cylinder_sfilter)
    print(f"Number of matched event found {l}")
    print("# xtck   ,           ,          , cylinder")
    print("# index_1,   time(ps), direction,   index_0,  enter(ps),  leave(ps), direction,")
    direction_dict = {True: "up", False: "down"}
    for s_pair in s_list:
        xtck = s_pair[0]
        cyli = s_pair[1]
        if xtck == "None":
            print(f"     None                       , {cyli.at_index:9}, {cyli.time - cyli.resident_time:10}, {cyli.time: 10}, {direction_dict[cyli.up]}")
        elif cyli == "None":
            print(f"{xtck.at_index:9}, {xtck.time:10}, {direction_dict[xtck.up]:9}, None")
        else:
            print(
                f"{xtck.at_index:9}, {xtck.time:10}, {direction_dict[xtck.up]:9}, {cyli.at_index:9}, {cyli.time - cyli.resident_time:10}, {cyli.time:10}, {direction_dict[cyli.up]}")

