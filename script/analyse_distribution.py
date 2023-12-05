#!/usr/bin/env python3

# command line tool to get SF state distribution
import argparse
import numpy as np
import Sfilter
from Sfilter import Cylinder_output
import pandas as pd
import json
import datetime
import os
import sys


def state_distribution_2_dict(state_distribution):
    # state_set = set([s[0] for traj in state_distribution for s in traj])
    state_set = set([s[0] for s in state_distribution[0]])
    state_dict = {s: {"ave": [], "low": [], "up": []} for s in state_set}
    for traj in state_distribution:
        # make sure every state appears in this traj
        for s in state_set:
            if s not in [i[0] for i in traj]:
                raise ValueError(f"State {s} does not appear in this traj.")
        # append the data to state_dict
        for s, ave, low, up in traj:
            state_dict[s]["ave"].append(ave)
            state_dict[s]["low"].append(ave - low)
            state_dict[s]["up"].append(up - ave)
    # convert to np.array
    for s in state_dict:
        state_dict[s]["ave"] = np.array(state_dict[s]["ave"])
        state_dict[s]["low"] = np.array(state_dict[s]["low"])
        state_dict[s]["up"] = np.array(state_dict[s]["up"])

    return state_dict


def load_hRE_bootstrap(ci, n_resamples=999):
    """load a set of HRE output and get the proportion of each state"""
    state_distribution = ci.get_state_distribution_CI_bootstrap_frame(n_resamples=n_resamples)
    return state_distribution_2_dict(state_distribution)


def save_distribution(distribution, x, file_name):
    # flat distribution
    dis_flat = {}
    dis_flat["x"] = x
    for state in distribution:
        dis_flat[state + "_ave"] = distribution[state]["ave"]
        dis_flat[state + "_low"] = distribution[state]["low"]
        dis_flat[state + "_up"] = distribution[state]["up"]

    # convert to pd.DataFrame and save
    df = pd.DataFrame(dis_flat)
    df.to_csv(file_name, index=None)


def print_sample_time(ci):
    """print the sampling time for each traj"""
    print("sim_length time_step (after slicing)")
    for fname, meta_data, traj in zip(ci.files, ci.meta_data, ci.state_str):
        time_step = meta_data["time_step"]
        sim_length = time_step * (len(traj) - 1) / 1000
        print(f"{sim_length:8.1f}ns {time_step:7.1f}ps  {fname}")
    return sim_length, time_step

def load_conduntance_bootstrap(base_list, n_resamples=999, method="K_priority"):
    state_list = []
    for i_dict in base_list:
        fi = i_dict["files"]
        if "end_frame" in i_dict:
            end_frame = i_dict["end_frame"]
        else:
            end_frame = None
        if "step" in i_dict:
            step_frame = i_dict["step"]
        else:
            step_frame = 1
        ci = Cylinder_output(fi, method=method, end=end_frame + 1, step=step_frame)
        # print the sampling
        print_sample_time(ci)
        # bootstrap
        state_distribution = ci.get_state_distribution_CI_bootstrap_traj(n_resamples=n_resamples)
        state_list.append(state_distribution)
    state_set = set([s[0] for sim in state_list for s in sim])
    state_dict = {s: {"ave": [], "low": [], "up": []} for s in state_set}
    for sim in state_list:
        sim_dict = {}
        for s, ave, low, up in sim:
            sim_dict[s] = ave, low, up
        for s in state_set:
            if s in sim_dict:
                ave, low, up = sim_dict[s]
                state_dict[s]["ave"].append(ave)
                state_dict[s]["low"].append(ave - low)
                state_dict[s]["up"].append(up - ave)
            else:
                state_dict[s]["ave"].append(np.nan)
                state_dict[s]["low"].append(np.nan)
                state_dict[s]["up"].append(np.nan)
    # convert to np.array
    for s in state_dict:
        state_dict[s]["ave"] = np.array(state_dict[s]["ave"])
        state_dict[s]["low"] = np.array(state_dict[s]["low"])
        state_dict[s]["up"] = np.array(state_dict[s]["up"])
    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     f"""Version {Sfilter.__version__}. This program read a list of HRE results. The 
                                     input file should be the std_out from count_cylinder.py. This program will load 6
                                     letter code and compute the proportion of each state. The Error estimation is the
                                     95% confidence interval of from bootstrap. The result will be saved in a csv file.
                                     """)
    parser.add_argument('-i',
                        dest="files",
                        metavar='files',
                        type=str,
                        nargs='+',
                        required=True,
                        help='std_out from count_cylinder.py, or a json file')
    parser.add_argument('-o',
                        dest="output",
                        metavar='output',
                        type=str,
                        default="output.csv",
                        help='output file name. Default: output.csv')
    parser.add_argument('-resample',
                        dest="resample",
                        metavar='resample',
                        type=int,
                        default=999,
                        help='number of resamples for bootstrap. Default: 999')
    parser.add_argument('-b',
                        dest="begin_step",
                        metavar='begin_step',
                        type=int,
                        default=0,
                        help='begin step. Default: 0, Will not take effect for multi_replica run.')
    parser.add_argument('-e',
                        dest="end_step",
                        metavar='end_step',
                        type=int,
                        default=-1,
                        help='end step. Default: -1. Will not take effect for multi_replica run.')
    parser.add_argument('-method',
                        dest="method",
                        metavar='method',
                        type=str,
                        default="K_priority",
                        choices=['K_priority', 'Co-occupy'],
                        help='method. Default: K_priority')

    # Parse arguments
    args = parser.parse_args()
    print("#################################################################################")
    print(f"Sfilter Version {Sfilter.__version__}")
    print("Time :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("The command you used is :")
    print(" ".join(sys.argv))
    # if args.files is a json file, read the filename from the json file
    if args.files[0].endswith(".json"):
        if len(args.files) == 1:
            print(f"You provided a json file {args.files[0]}. Read file names from the json file.")
            with open(args.files[0]) as f:
                file_list = json.load(f)
        else:
            raise ValueError("You provided a json file and other files. Please provide only one json file.")
    else:
        file_list = args.files
    # if file_list is a list of file names, This is a RE run. Print the file names
    # if file_list is a list of list of file names, This is a multi_replica run. Print the file names.
    if isinstance(file_list[0], dict):
        print("There are multiple files in each replica. We will bootstrap each traj. The input files are :")
        print("#" * 80)
        run_type = "multi_replica"
        for i_dict in file_list:
            fi = i_dict["files"]
            end_frame = i_dict["end_frame"]
            step_frame = i_dict["step"]
            for f in fi:
                print(f)
            print("#" * 80)
    elif isinstance(file_list[0], str):
        print("You provide a list of files. We will bootstrap frames. The input files are :")
        run_type = "RE"
        for f in file_list:
            print(f)
        print("#" * 80)
    else:
        raise ValueError("The input file list should be a list of file names or a list of list of file names.")

    print(f"The number of resamples for bootstrap is {args.resample}.")

    start = args.begin_step
    if args.end_step == -1:
        end = None
    else:
        end = args.end_step

    # if output file exists, rename it so that the name is covered by #, such as output.csv -> #output.csv.1#
    if os.path.isfile(args.output):
        i = 1
        while os.path.isfile(f"#{args.output}.{i}#"):
            i += 1
        print(f"{args.output} exists. Rename it to #{args.output}.{i}#")
        os.rename(args.output, f"#{args.output}.{i}#")

    if run_type == "RE":
        c_tmp = Cylinder_output(file_list, method=args.method, start=start, end=end)
        print_sample_time(c_tmp)
        res = load_hRE_bootstrap(c_tmp, n_resamples=args.resample)
        save_distribution(res, range(0, len(c_tmp.state_str)), args.output)
    elif run_type == "multi_replica":
        res = load_conduntance_bootstrap(file_list, n_resamples=args.resample, method=args.method)
        save_distribution(res, range(0, len(file_list)), args.output)
