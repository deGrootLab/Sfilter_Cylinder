#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis import transformations
import argparse
import datetime
import json

try:
    import Sfilter as S_package

    S_version = S_package.__version__
except ImportError:
    S_version = "X.X.X-not_found"


class Perm_Event_Pool:
    def __init__(self, ind, state_array):
        """

        :param ind: a list of index. You can provide it as mda.Universe.select_atoms("name P").ix
        :param state_array: The state in the first frame. No even number is allowed.
        """
        # state_array should only contain odd number
        if np.any(state_array % 2 == 0):
            raise ValueError("state_array should only contain odd number")
        if not isinstance(ind, np.ndarray):
            ind = np.array(ind)
        self.ind = ind
        self.frame = 0

        self.previous_state = np.zeros(len(ind), dtype=int)
        self.previous_state[:] = state_array
        self.previous_odd = np.zeros(len(ind), dtype=int)
        self.previous_odd[:] = state_array
        self.leaving_time = np.zeros(len(ind), dtype=int)  # the frame when the atom leaves the previous odd state
        self.odd_mask_old = np.ones(len(ind), dtype=bool)  # the mask for the previous odd state
        self.permeation_event = []

    def update(self, state_array):
        """
        give the state for a new frame, count the permeation. The first frame should be given in the __init__. The final
        frame should be added by this function and later by update_final_frame.
        :param state_array:
        :return: None
        """
        # frame 0 should be given in the __init__
        self.frame += 1

        diff_mask = self.previous_state != state_array
        odd_mask = np.mod(state_array, 2) == 1

        # update the leaving time
        self.leaving_time[diff_mask & self.odd_mask_old] = self.frame - 1

        # get permeation event, current odd that does not equal to previous odd
        perm_mask = odd_mask & (state_array != self.previous_odd)
        self.append_permeation_event(perm_mask, state_array)

        # update the previous state
        self.previous_state[:] = state_array
        self.previous_odd[odd_mask] = state_array[odd_mask]
        self.odd_mask_old = odd_mask

    def append_permeation_event(self, perm_mask, state_array):
        for a_index, frame_leave, s0, s1 in zip(self.ind[perm_mask],
                                                self.leaving_time[perm_mask],
                                                self.previous_odd[perm_mask],
                                                state_array[perm_mask]):
            self.permeation_event.append([a_index, frame_leave, self.frame, (s1 - s0) // 2])

    def update_final_frame(self, state_array):
        """
        give the state for the final frame again so that atoms that stays in the membrane (even) can be counted,
        only odd number is allowed.
        :param state_array:
        :return: None
        """
        # previously even, and now odd, and different from the previous odd
        diff_mask = self.previous_state != state_array
        odd_diff_mask = self.previous_odd != state_array
        perm_mask = diff_mask & odd_diff_mask
        self.append_permeation_event(perm_mask, state_array)

    def write_result(self, file_name):
        """
        write the permeation to a file
        | index0 | frame_leave | frame_enter | state_change |
        index0 is the 0 base index which can be read by mda.Universe.select_atoms("XXX").ix
        frame_leave is the frame when the atom leaves the previous odd state (water compartment)
        frame_enter is the frame when the atom enters the current odd state (water compartment)
        state_change is the state change, 1 for moving up 1 compartment, -1 for moving down 1 compartment
        :param file_name:
        :return: None
        """
        # prepare a dataframe ["index", "frame_leave", "frame_enter"]
        df = pd.DataFrame(self.permeation_event, columns=["index0", "frame_leave", "frame_enter", "state_change"])
        df.to_csv(file_name)


def args_dict_safty_check(args_dict):
    """check the args_dict to make sure all the necessary keys are there"""
    for k in ["pdb", "xtc", "selection_dict", ]:
        if k not in args_dict:
            raise ValueError(f"{k} is not found in input json file.")
    if "mem_str" not in args_dict and "mem_layer" not in args_dict:
        args_dict["mem_str"] = "name P"
    elif "mem_str" in args_dict and "mem_layer" in args_dict:
        raise ValueError("mem_str and mem_layer cannot be used together")
    if "output_post_fix" not in args_dict:
        args_dict["output_post_fix"] = "perm_event_nj.csv"
    # output_post_fix need to be csv
    if not args_dict["output_post_fix"].endswith(".csv"):
        args_dict["output_post_fix"] += ".csv"
    return args_dict


def print_args_dict(args_dict):
    """print the args_dict"""
    print("The following arguments are used:")
    print(json.dumps(args_dict, indent=4))


def set_up_boundary_mem_str(u, mem_str):
    """
    set up the boundary using mem_str, normally you can give it as "name P". This function will decide which P is in the
    upper/lower leaflet
    :param u: mda.Universe
    :param mem_str: a string that can be used to select the membrane
    :return: selection_upper, selection_lower
    """
    selection_all = u.select_atoms(mem_str)
    center_z = selection_all.positions[:, 2].mean()
    upper_selection = selection_all.select_atoms(f"prop z > {center_z}")
    lower_selection = selection_all.select_atoms(f"prop z < {center_z}")
    return upper_selection, lower_selection


def set_up_boundary_mem_layer(u, mem_layer):
    """
    check, order, and group the mem_layer. This is suitable for multiple layers of membrane.
    :param u: mdanalysis.Universe
    :param mem_layer: a dictionary of selection_string pair, e.g.
    {
        mem1: ["name P_Up1", "name P_Down1"],
        mem2: ["name P_Up2", "name P_Down2"]
    }
    :return: selection_upper_list, selection_lower_list
    """
    boundary_dict = {}
    for name, (upper, lower) in mem_layer.items():
        selection_upper = u.select_atoms(upper)
        selection_lower = u.select_atoms(lower)
        if len(selection_upper) == 0:
            raise ValueError(f"selection {upper} does not select any atom")
        if len(selection_lower) == 0:
            raise ValueError(f"selection {lower} does not select any atom")
        # check if the two selection overlap in Z
        if np.any(np.abs(selection_upper.positions[:, 2] - selection_lower.positions[:, 2]) < 0.01):
            raise ValueError(f"selection {upper} and {lower} overlap in Z")
        boundary_dict[name] = [selection_upper, selection_lower]
    return boundary_dict


def div_mod_coordinate_z(atom_z, upper, lower, box_z):
    """
    If an atom is in membrane layer, assign an even number, and depending on the PBC position, assign -2, 0, 2
    If an atom is in water layer, assign an odd number, and depending on the PBC position, assign -3, -1, 1, 3
    3 water
    2 membrane
    1 water
    0 membrane
    -1 water
    -2 membrane
    -3 water
    :param atom_z: the z coordinate of the atom
    :param upper: the z coordinate of the upper boundary
    :param lower: the z coordinate of the lower boundary
    :param box_z: the box size in z
    :return: state_array
    """
    coord_z = atom_z - (upper + lower) / 2
    mem_thickness = upper - lower
    div, mod = np.divmod(coord_z, box_z)
    state_array = np.zeros(atom_z.shape, dtype=int)
    state_array[:] = div * 2
    state_array[mod > mem_thickness / 2] += 1
    state_array[mod > box_z - mem_thickness / 2] += 1
    return state_array


def calc_state_array(permeable_selection, upper_selection, lower_selection):
    """
    according to the Z coordinate, calculate the state array, which encode which compartment the atom is in

    3 water
    2 membrane
    1 water
    0 membrane
    -1 water
    -2 membrane
    -3 water

    :param permeable_selection: MDAnalysis selection
    :param upper_selection: MDAnalysis selection
    :param lower_selection: MDAnalysis selection
    :return: state_array
    """
    # this will work on nojump trajectory
    z_upper = np.mean(upper_selection.positions[:, 2])
    z_lower = np.mean(lower_selection.positions[:, 2])
    box_z = permeable_selection.dimensions[2]
    return div_mod_coordinate_z(permeable_selection.positions[:, 2], z_upper, z_lower, box_z)


def calc_state_array_odd_only(permeable_selection, upper_selection, lower_selection):
    """

    :param permeable_selection:
    :param upper_selection:
    :param lower_selection:
    :return:
    """
    z_center = (np.mean(upper_selection.positions[:, 2]) + np.mean(lower_selection.positions[:, 2])) / 2
    box_z = permeable_selection.dimensions[2]
    coord_z = permeable_selection.positions[:, 2] - z_center
    div, mod = np.divmod(coord_z, box_z)
    state_array = np.zeros(coord_z.shape[0], dtype=int)
    state_array[:] = div * 2 + 1
    return state_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     f"""Version {S_version}. This is the script to run counting. It should works for 
                                     anything that crosses the membrane. This script can work alone outside Sfilter 
                                     package""", )
    parser.add_argument("-i",
                        dest="json_file",
                        type=argparse.FileType('r'),
                        nargs=1,
                        required=True)

    args = parser.parse_args()
    now = datetime.datetime.now()
    print("#################################################################################")
    print(f"Sfilter Version {S_version}")
    print("Time :", now.strftime("%Y-%m-%d %H:%M:%S"))
    print("The command you used is :")
    print(" ".join(sys.argv))

    args_dict = json.load(args.json_file[0])
    args_dict = args_dict_safty_check(args_dict)
    print_args_dict(args_dict)

    # build selection
    u = mda.Universe(args_dict["pdb"], args_dict["xtc"])

    permeable_dict = {}
    for k, v in args_dict["selection_dict"].items():
        permeable_dict[k] = u.select_atoms(v)
    print("#################################################################################")
    print("The permeable atoms are:")
    for k, v in permeable_dict.items():
        print(f"{k} : {v.ix}")
    if "mem_str" in args_dict:
        selection_upper, selection_lower = set_up_boundary_mem_str(u, args_dict["mem_str"])
        boundary_dict = {"mem": [selection_upper, selection_lower]}
    else:
        boundary_dict = set_up_boundary_mem_layer(u, args_dict["mem_layer"])
    print("#################################################################################")
    print("The membrane layers' index are:")
    for k, v in boundary_dict.items():
        print(f"{k} :")
        print(f"upper {v[0].ix}")
        print(f"lower {v[1].ix}")

    # Every permeable atom with every membrane layer should have a detector.
    # A detector consists of a selection of permeable atoms, and 2 selections of membrane layer (upper and lower)
    detector_dict = {}
    for permeable_name, permeable_selection in permeable_dict.items():
        for mem_name, (upper_selection, lower_selection) in boundary_dict.items():
            detector_dict[permeable_name + "_" + mem_name] = [permeable_selection, upper_selection, lower_selection]
    # init Perm_Event_Pool
    for name, detector in detector_dict.items():
        # print(name)
        permeable_selection, upper_selection, lower_selection = detector
        state_array = calc_state_array_odd_only(permeable_selection, upper_selection, lower_selection)
        p_event = Perm_Event_Pool(permeable_selection.ix, state_array)
        detector.append(p_event)

    # loop through the trajectory
    traj_len = len(u.trajectory)
    # u.trajectory.add_transformations(transformations.nojump.NoJump())
    for ts in u.trajectory:
        if ts.frame == 0:
            print(f"The time step of this trajectory (ps): {ts.dt}")
            continue
        elif ts.frame < traj_len - 1:
            for name, detector in detector_dict.items():
                permeable_selection, upper_selection, lower_selection, p_event = detector
                p_event.update(calc_state_array(permeable_selection, upper_selection, lower_selection))
        else:
            for name, detector in detector_dict.items():
                permeable_selection, upper_selection, lower_selection, p_event = detector
                p_event.update_final_frame(
                    calc_state_array_odd_only(permeable_selection, upper_selection, lower_selection))
            print(f"The number of frame                  : {ts.frame}")
            print(f"The length of the trajectory (ns)    : {ts.frame * ts.dt / 1000}")

    # write result to file
    for name, detector in detector_dict.items():
        permeable_selection, upper_selection, lower_selection, p_event = detector
        p_event.write_result(name + "_" + args_dict["output_post_fix"])
        up_event = sum(i[3] for i in p_event.permeation_event if i[3] > 0)
        down_event = sum(i[3] for i in p_event.permeation_event if i[3] < 0)
        print(f"{name} sum  : {up_event + down_event}")
        print(f"{name} up   : {up_event}")
        print(f"{name} down : {down_event}")
        print(f"{name} sum  per ns : {(up_event + down_event) / u.trajectory.totaltime * 1000}")
        print(f"{name} up   per ns : {up_event / u.trajectory.totaltime * 1000}")
        print(f"{name} down per ns : {down_event / u.trajectory.totaltime * 1000}")

    print("Time :", now.strftime("%Y-%m-%d %H:%M:%S"))
