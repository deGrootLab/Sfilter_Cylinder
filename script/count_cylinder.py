#!/usr/bin/env python3

# command line tool to run permeation counting
import argparse
import warnings

from Sfilter import Sfilter
import numpy as np
import MDAnalysis as mda
import sys
import datetime


class PermeationEvent:
    def __init__(self, ind):
        self.index = ind
        self.state_memory = np.zeros((len(ind), 3), dtype=int)
        self.resident_time = np.zeros((len(ind), 3), dtype=int)
        self.frame = 0
        self.up_1_count   = []  # permeation up across 1
        self.down_1_count = []  # permeation down across 1
        self.up_3_count   = []
        self.down_3_count = []
        self.up_4_count   = []
        self.down_4_count = []
        self.safe_1_2_count = []  # Safety check
        self.safe_2_1_count = []
        self.safe_3_2_count = []
        self.safe_2_3_count = []
        self.safe_4_2_count = []
        self.safe_2_4_count = []
        self.atom_in_1 = np.zeros((len(ind),), dtype=bool)
        self.atom_in_5 = np.zeros((len(ind),), dtype=bool)

    def update(self, state_array):
        """
        :param state_array: state from 1 to 5
        :return: None
        """
        self.atom_in_1 = state_array == 1
        self.atom_in_5 = state_array == 5
        if self.frame == 0:
            state_array[self.atom_in_1] = 3
            state_array[self.atom_in_5] = 4
            self.state_memory[:, 2] = state_array
            self.resident_time[:, 2] = 1
            self.frame += 1
        else:
            state_array[state_array == 5] = 1
            # for repeating state, add resident time
            mask_repeat = self.state_memory[:, 2] == state_array
            self.resident_time[mask_repeat, 2] += 1
            # for non-repeating state, update state_memory/resident_time, check permeation
            self.state_memory[~mask_repeat, 0:2] = self.state_memory[~mask_repeat, 1:]
            self.resident_time[~mask_repeat, 0:2] = self.resident_time[~mask_repeat, 1:]
            self.state_memory[~mask_repeat, 2] = state_array[~mask_repeat]
            self.resident_time[~mask_repeat, 2] = 1

            for seq, count in ((np.array([[4, 1, 3]]), self.up_1_count),
                               (np.array([[3, 1, 4]]), self.down_1_count),
                               (np.array([[1, 3, 4]]), self.up_3_count),
                               (np.array([[3, 1, 4]]), self.down_3_count),
                               (np.array([[3, 4, 1]]), self.up_4_count),
                               (np.array([[1, 4, 3]]), self.down_4_count),
                               ):
                mask_event = np.logical_and( np.all(self.state_memory == seq, axis=1), ~mask_repeat)
                for i, res_time in zip(self.index[mask_event], self.resident_time[mask_event, 1]):
                    count.append([i, self.frame, res_time])
            for seq, count in ((np.array([[1, 2]]), self.safe_1_2_count),
                               (np.array([[2, 1]]), self.safe_2_1_count),
                               (np.array([[3, 2]]), self.safe_3_2_count),
                               (np.array([[2, 3]]), self.safe_2_3_count),
                               (np.array([[4, 2]]), self.safe_4_2_count),
                               (np.array([[2, 4]]), self.safe_2_4_count),
                               ):
                mask_event = np.logical_and(np.all(self.state_memory[:, 1:] == seq, axis=1), ~mask_repeat)
                for i, res_time in zip(self.index[mask_event], self.resident_time[mask_event, 1:]):
                    count.append([i, self.frame, res_time])
            self.frame += 1

    def final_frame_check(self):
        # check the atom the end in 1 or 5
        mask_1 = self.atom_in_1
        self.state_memory[mask_1, 0:2] = self.state_memory[mask_1, 1:]
        self.resident_time[mask_1, 0:2] = self.resident_time[mask_1, 1:]
        self.state_memory[mask_1, 2] = 3
        self.resident_time[mask_1, 2] = 1
        mask_5 = self.atom_in_5
        self.state_memory[mask_5, 0:2] = self.state_memory[mask_5, 1:]
        self.resident_time[mask_5, 0:2] = self.resident_time[mask_5, 1:]
        self.state_memory[mask_5, 2] = 4
        self.resident_time[mask_5, 2] = 1
        mask_cylinder = np.logical_or(mask_1, mask_5)
        for seq, count in ((np.array([[4, 1, 3]]), self.up_1_count),
                           (np.array([[3, 1, 4]]), self.down_1_count),
                           ):
            mask_event = np.logical_and(np.all(self.state_memory == seq, axis=1), mask_cylinder)
            for i, res_time in zip(self.index[mask_event], self.resident_time[mask_event, 1]):
                count.append([i, self.frame-1, res_time])

    def write_result(self, file, charge, voltage, time_step):
        lines = []
        lines.append("# Permeation up 4 -> 1 -> 3 #######\n")
        lines.append("# index (0 base),  frame,  time(ps), resident_time_in_the_cylinder\n")
        for index, frame, resident_time in self.up_1_count:
            lines.append(f"  {index:<14d}, {frame:6d}, {frame * time_step:9}, {resident_time * time_step}\n")
        if len(self.up_1_count) == 0:
            lines.append("None\n")
        lines.append("# Permeation up 3 -> 1 -> 4 #######\n")
        lines.append("# index (0 base),  frame,  time(ps), resident_time_in_the_cylinder\n")
        for index, frame, resident_time in self.down_1_count:
            lines.append(f"  {index:<14d}, {frame:6d}, {frame * time_step:9}, {resident_time * time_step}\n")
        if len(self.down_1_count) == 0:
            lines.append("None\n")
        up_count_sum = len(self.up_1_count) - len(self.down_1_count)
        current = (up_count_sum) * 1.602176634 / (time_step * self.frame) * 100000.0 * charge  # pA
        conductance = current * 1000 / voltage  # pS
        lines.append("\n#################################\n")
        lines.append(f"Assumed voltage (mV)   : {voltage}\n" )
        lines.append(f"Simulation time (ns)   : {time_step * (self.frame - 1) / 1000}\n")
        lines.append(f"Permeation events up   : {len(self.up_1_count)}\n")
        lines.append(f"Permeation events down : {len(self.down_1_count)}\n")
        lines.append(f"Ave current (pA)       : {current:.5f}\n")
        lines.append(f"Ave conductance (pS)   : {conductance:.5f}\n")
        if len(self.up_1_count) != 0:
            lines.append(f"Minimum resident time up crossing   : {time_step * min(x[2] for x in self.up_1_count)}\n")
        if len(self.down_1_count) != 0:
            lines.append(f"Minimum resident time down crossing : {time_step * min(x[2] for x in self.down_1_count)}\n")
        lines.append("###############################\n")
        lines.append("\n")
        lines.append("Safety check\n")
        up_4_count = len(self.up_4_count) - len(self.down_4_count)
        lines.append(f"Net permeation events up crossing 4    : {up_4_count}\n")
        if len(self.up_4_count) != 0:
            min_time = time_step * min(x[2] for x in self.up_4_count)
            lines.append(f"Minimum up   crossing resident time    : {min_time}\n")
        if len(self.down_4_count) != 0:
            min_time = time_step * min(x[2] for x in self.down_4_count)
            lines.append(f"Minimum down crossing resident time    : {min_time}\n")
        up_3_count = len(self.up_3_count) - len(self.down_3_count)
        lines.append(f"Net permeation events up crossing 3    : {up_3_count}\n")
        if len(self.up_3_count) != 0:
            min_time = time_step * min(x[2] for x in self.up_3_count)
            lines.append(f"Minimum up   crossing resident time    : {min_time}\n")
        if len(self.down_3_count) != 0:
            min_time = time_step * min(x[2] for x in self.down_3_count)
            lines.append(f"Minimum down crossing resident time    : {min_time}\n")
        for name, i in (("1_2, SF inside out    ", self.safe_1_2_count),
                        ("2_1, SF outside in    ", self.safe_2_1_count)):
            lines.append(f"Number of {name} event : {len(i)}\n")
            for index, frame, res_time in i:
                lines.append(f"{index} , frame {frame}, resident time (ps) : {res_time[0] * time_step},  {res_time[1] * time_step}\n")

        for name, i in (("3_2, upper to membrane", self.safe_3_2_count),
                        ("2_3, membrane to upper", self.safe_2_3_count),
                        ("4_2, lower to membrane", self.safe_4_2_count),
                        ("2_4, membrane to lower", self.safe_2_4_count)):
            lines.append(f"Number of {name} event : {len(i)}\n")
        lines.append("Done\n")
        with open(file, "w") as f:
            f.writelines(lines)


def state_label_convert(at_state_a):
    at_state_a[at_state_a == 7] = 13
    at_state_a[at_state_a == 0] = 13
    at_state_a[at_state_a == 1] = 11
    at_state_a[at_state_a == 2] = 11
    at_state_a[at_state_a == 3] = 15
    at_state_a[at_state_a == 4] = 15
    at_state_a[at_state_a == 5] = 14
    at_state_a[at_state_a == 6] = 14
    at_state_a[at_state_a == 8] = 12
    at_state_a = at_state_a - 10
    return at_state_a



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdb",
                        dest="top",
                        help="Ideally This file should be generated from the same trjconv command as xtc. gro and tpr "
                             "are also acceptable",
                        metavar="top.pdb",
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument("-xtc",
                        dest="traj",
                        metavar="fix_c.xtc",
                        help="This traj should have SF centered.",
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument("-K",
                        dest="K_name",
                        metavar="atom_name",
                        help="Atom name that will be checked for permeation. It can be K, POT, Na, SOD..",
                        type=str,
                        nargs="+",
                        default=["K"])
    parser.add_argument("-volt",
                        dest="volt",
                        metavar="float",
                        help="Voltage in mV, default 300.",
                        type=float,
                        default=300.)
    parser.add_argument("-cylinderRad",
                        dest="cylRAD",
                        metavar="float",
                        help="Radius of the cylinder in Å. Default 2.5",
                        type=float,
                        default=2.5)
    parser.add_argument("-S0_Rad",
                        dest="s0_rad",
                        metavar="float",
                        help="Radius of the S0 in Å. Default 4",
                        type=float,
                        default=4)
    parser.add_argument("-S5_cutoff",
                        dest="s5_cutoff",
                        metavar="float",
                        help="Z cutoff of the S5 in Å. Default value is 4. You might see double occupation",
                        type=float,
                        default=4)
    parser.add_argument("-SF_seq",
                        dest="SF_seq",
                        metavar="list of string",
                        help="THR VAL GLY TYR GLY",
                        type=str,
                        nargs="+")
    parser.add_argument("-SF_seq2",
                        dest="SF_seq2",
                        metavar="list of string",
                        help="THR VAL GLY PHE GLY",
                        type=str,
                        default=[],
                        nargs="+")
    parser.add_argument("-o",
                        dest="output_postfix",
                        metavar="file name",
                        type=str,
                        default="perm_event.out",
                        help="post fix for output, default perm_event.out",)
    args = parser.parse_args()
    now = datetime.datetime.now()
    print("#################################################################################")
    print("Time :", now.strftime("%Y-%m-%d %H:%M:%S"))
    print("The command you used is :")
    print(" ".join(sys.argv))
    print("PDB top file :", args.top.name)
    print("xtc traj file:", args.traj.name)
    print("Ion name(s) in this pdb should be:", args.K_name)
    print(f"The Voltage in this simulation is: {args.volt} mV")
    print("The sequence of the SF is:", args.SF_seq, args.SF_seq2)
    print(f"The cylinder radius is   : {args.cylRAD} Å")
    print(f"Radius cutoff for S0 is  : {args.s0_rad} Å")
    print(f"Z cutoff for S5 is       : {args.s5_cutoff} Å")
    print("#################################################################################")

    u = mda.Universe(args.top.name, args.traj.name)
    sf = Sfilter(u)
    sf.detect_SF_sequence(args.SF_seq, args.SF_seq2)
    for site, atoms in zip(("S00", "S01", "S12", "S23", "S34", "S45"),
                           sf.sf_oxygen):
        print(site)
        for atom in atoms:
            print(f'{atom.resname} {atom.name}, Index (0 base): {atom.index}')
    event_count_dict = {}
    atom_selection_dict = {}
    for atom in args.K_name:
        selection = sf.u.select_atoms('name ' + atom)
        print(f"Number of {atom} found is {len(selection.ix)}")
        print(selection.ix)
        event_count_dict[atom] = PermeationEvent(selection.ix)
        atom_selection_dict[atom] = selection
    wat_selection = sf.u.select_atoms('resname SOL and name OW')
    event_count_dict["Wat"] = PermeationEvent(wat_selection.ix)
    atom_selection_dict["Wat"] = wat_selection

    print("# Loop Over Traj ################################################################################")
    for ts in u.trajectory:
        state_ts_dict = {}
        for at_name in event_count_dict:
            at_selection = atom_selection_dict[at_name]
            at_state_a = sf.state_detect(at_selection)  # state array with the label for every atom
            at_state_l = sf.state_2_list(at_state_a, at_selection)  # state list with the index of atoms in every binding site
            state_ts_dict[at_name] = at_state_l
            at_state_a = state_label_convert(at_state_a)
            event_count_dict[at_name].update(at_state_a)

        # if K Water system print as K_priority
        if args.K_name == ["K"]:
            k_state = state_ts_dict["K"]
            o_state = state_ts_dict["Wat"]
            state_string = sf.state_2_string([k_state, o_state], method="K_priority")
            for k_index, site in zip(k_state, ["0", "1", "2", "3", "4", "5"]):
                sumK = len(k_index)
                if sumK >= 2:
                    warnings.warn(f"Number of K in site { site } is {sumK} in frame {ts.frame}")
        elif args.K_name == ["POT"]:
            k_state = state_ts_dict["POT"]
            o_state = state_ts_dict["Wat"]
            state_string = sf.state_2_string([k_state, o_state], method="K_priority")
            for k_index, site in zip(k_state, ["0", "1", "2", "3", "4", "5"]):
                sumK = len(k_index)
                if sumK >= 2:
                    warnings.warn(f"Number of K in site { site } is {sumK} in frame {ts.frame}")
        else:
            method = "Everything"
            state_string = sf.state_2_string(state_ts_dict, method="Everything")
        print("# S6l", ts.frame, state_string)
        for i in range(6):
            for at_name in state_ts_dict:
                print(f" {at_name} : ", end="")
                for index in state_ts_dict[at_name][i]:
                    print(index, end=" ")
                print(end=",")
            print()
    print("#################################################################################")
    knows_charge_table = {"POT": 1, "K": 1,
                          "SOD": 1, "Na": 1,
                          "LIT": 1, "Li": 1,
                          "Wat": 0,
                          "CLA": -1, "CL": -1,
                          "OH": -1,
                          "CAL": 2, "CA": 2,
                          "RUB": 1, "RB": 1,
                          "CS": 1,
                          }
    for at_name in event_count_dict:
        event_count_dict[at_name].final_frame_check()
        if at_name in knows_charge_table:
            charge = abs(knows_charge_table[at_name])
        else:
            charge = 1
        event_count_dict[at_name].write_result(file=f"{at_name}_{args.output_postfix}",
                                               charge=charge,
                                               voltage=args.volt,
                                               time_step=ts.dt)
