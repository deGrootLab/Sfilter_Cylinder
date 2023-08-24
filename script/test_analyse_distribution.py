import unittest
import analyse_distribution
import numpy as np
from Sfilter import Cylinder_output
import json


class MyTestCase(unittest.TestCase):

    # test each function
    def test_state_distribution_2_dict(self):
        print("TEST: convert state_distribution to dict")
        state_distribution = [
            [["A", 0.5, 0.4, 0.6], ["B", 0.3, 0.2, 0.4], ["C", 0.2, 0.1, 0.3]],  # traj 1
            [["A", 0.1, 0.0, 0.2], ["B", 0.5, 0.4, 0.6], ["C", 0.4, 0.3, 0.6]],  # traj 2
        ]
        state_dict = analyse_distribution.state_distribution_2_dict(state_distribution)
        answer_dict = {
            "A": {"ave": [0.5, 0.1],
                  "low": [0.1, 0.1],
                  "up": [0.1, 0.1]
                  },
            "B": {"ave": [0.3, 0.5],
                  "low": [0.1, 0.1],
                  "up": [0.1, 0.1]
                  },
            "C": {"ave": [0.2, 0.4],
                  "low": [0.1, 0.1],
                  "up": [0.1, 0.2]
                  }
        }
        for state in answer_dict:
            for key in answer_dict[state]:
                self.assertTrue(np.allclose(answer_dict[state][key], state_dict[state][key]))

    def test_load_hRE_bootstrap(self):
        print("TEST: load HRE output, and get the proportion of each state")
        f_list = ["../test/05-HRE/01-charmm-charge/0/k_cylinder/08-02-1.8A/k_cylinder.log",
                  "../test/05-HRE/01-charmm-charge/1/k_cylinder/08-02-1.8A/k_cylinder.log"]
        ci = Cylinder_output(f_list, method="K_priority", end=None, step=1)
        state_dict = analyse_distribution.load_hRE_bootstrap(ci, n_resamples=99)

        # make sure every traj in the output has the correct length
        for s in state_dict:
            self.assertEqual(len(state_dict[s]["ave"]), 2)
            self.assertEqual(len(state_dict[s]["low"]), 2)
            self.assertEqual(len(state_dict[s]["up"]), 2)

        answer_dict = {"WKK0K0":
                           {"ave": [1479 / 4001, 1709 / 4001], },
                       "WK0KK0":
                           {"ave": [895 / 4001, 1045 / 4001], },
                       }
        for state in answer_dict:
            for key in answer_dict[state]:
                self.assertTrue(np.allclose(answer_dict[state][key], state_dict[state][key]))

    def test_print_sample_time(self):
        print("TEST: print the sampling time and frame interval for each traj")
        f_list = ["../test/05-HRE/01-charmm-charge/0/k_cylinder/08-02-1.8A/k_cylinder.log",
                  "../test/05-HRE/01-charmm-charge/1/k_cylinder/08-02-1.8A/k_cylinder.log"]
        ci = Cylinder_output(f_list, method="K_priority", end=None, step=1)
        sim_length, time_step = analyse_distribution.print_sample_time(ci)
        self.assertEqual(sim_length, 80)  # ns
        self.assertEqual(time_step, 20)  # ps
        ci = Cylinder_output(f_list, method="K_priority", end=None, step=2)
        sim_length, time_step = analyse_distribution.print_sample_time(ci)
        self.assertEqual(sim_length, 80)
        self.assertEqual(time_step, 40)

    def test_load_conduntance_bootstrap(self):
        print("TEST: load voltage simulation output, and get the proportion of each state")
        with open("./NaK2K_C.json") as f:
            file_list = json.load(f)
        state_dict = analyse_distribution.load_conduntance_bootstrap(file_list, n_resamples=29, method="K_priority")
        # check length
        for s in state_dict:
            self.assertEqual(len(state_dict[s]["ave"]), 3)
            self.assertEqual(len(state_dict[s]["low"]), 3)
            self.assertEqual(len(state_dict[s]["up"]), 3)
        answer_dict = {"WKK0K0":
                           {"ave": [(11219 + 11600) / (25001 * 2),
                                    (2479 + 8223) / (25001 * 2),
                                    (1040 + 398) / (25001 * 2)], },
                       "WK0KK0":
                           {"ave": [(7734 + 8182) / (25001 * 2),
                                    (592 + 6865) / (25001 * 2),
                                    (1146 + 378) / (25001 * 2)], },
                       }
        for state in answer_dict:
            for key in answer_dict[state]:
                self.assertTrue(np.allclose(answer_dict[state][key], state_dict[state][key]))

    # test the whole program


if __name__ == '__main__':
    unittest.main()
