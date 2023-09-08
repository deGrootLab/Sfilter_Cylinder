import unittest
import numpy as np
from Sfilter import kinetics
from pathlib import Path
from Sfilter import read_k_cylinder
from numpy.testing import assert_allclose

class MyTestCase(unittest.TestCase):
    def test_init(self):
        print("#TESTING: init. Init the Sf_model class in the normal way.")
        base = Path("04-output_wrapper/C_0.75_2ps/05-2ps")
        file_list = [base / f"{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        k_model = kinetics.Sf_model(file_list)
        self.assertAlmostEqual(k_model.time_step, 2.0)
        self.assertAlmostEqual(k_model.state_distribution["WKK0KW"], 322 / 1002)
        self.assertAlmostEqual(k_model.state_distribution["KK0KKW"], 207 / 1002)
        self.assertAlmostEqual(k_model.state_distribution["WK0KKW"], 146 / 1002)
        self.assertAlmostEqual(k_model.state_distribution["KKK0KW"], 89 / 1002)
        self.assertAlmostEqual(k_model.state_distribution["WKK0KK"], 74 / 1002)
        self.assertEqual(k_model.total_frame, 1002)
        self.assertListEqual(k_model.traj_raw[0][:5].tolist(), [2, 2, 2, 0, 0])
        self.assertListEqual(k_model.traj_raw[1][:5].tolist(), [3, 1, 1, 1, 1])

        # lumped traj
        self.assertAlmostEqual(k_model.node_distribution[0], 322 / 1002)
        self.assertAlmostEqual(k_model.node_distribution[1], 207 / 1002)
        self.assertAlmostEqual(k_model.node_distribution[2], 146 / 1002)
        self.assertAlmostEqual(k_model.node_distribution[3], 89 / 1002)
        self.assertListEqual(k_model.traj_node[0][:5].tolist(), [2, 2, 2, 0, 0])
        self.assertListEqual(k_model.traj_node[1][:5].tolist(), [3, 1, 1, 1, 1])
        self.assertAlmostEqual(k_model.node_distribution_str[("WKK0KW",)], 322 / 1002)
        self.assertAlmostEqual(k_model.node_distribution_str[("KK0KKW",)], 207 / 1002)
        self.assertAlmostEqual(k_model.node_distribution_str[("WK0KKW",)], 146 / 1002)
        self.assertAlmostEqual(k_model.node_distribution_str[("KKK0KW",)], 89 / 1002)
        self.assertAlmostEqual(k_model.node_distribution_str[("WKK0KK",)], 74 / 1002)

        k_model = kinetics.Sf_model(file_list, step=2)
        self.assertAlmostEqual(k_model.time_step, 4.0)
        self.assertEqual(k_model.total_frame, 502)
        self.assertListEqual(k_model.traj_raw[0][:3].tolist(), [2, 2, 0])
        self.assertListEqual(k_model.traj_raw[1][:3].tolist(), [3, 1, 1])

    def test_set_traj_from_str(self):
        print("#TESTING: set_traj_from_str. Init an empty Sf_model class and set the traj from lists of string.")
        base = Path("04-output_wrapper/C_0.75_2ps/05-2ps")
        traj_l = [read_k_cylinder(base / f"{i:02}/analysis/04-state-code/k_cylinder.log")[0] for i in range(2)]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=2.0)

        self.assertAlmostEqual(k_model.time_step, 2.0)

    def test_get_traj_in_string(self):
        print("#TESTING: get_traj_in_string, convert the int traj back to str")
        base = Path("04-output_wrapper/C_0.75_2ps/05-2ps")
        file_list = [base / f"{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        k_model = kinetics.Sf_model(file_list)
        traj_str = k_model.get_traj_in_string()
        self.assertListEqual(traj_str[0][:4], ["WK0KKW", "WK0KKW", "WK0KKW", "WKK0KW"])
        self.assertListEqual(traj_str[1][:4], ["KKK0KW", "KK0KKW", "KK0KKW", "KK0KKW"])
        self.assertEqual(len(traj_str), 2)
        self.assertEqual(len(traj_str[0]), 501)
        self.assertEqual(len(traj_str[1]), 501)

    def test_set_lumping_from_str(self):
        print("TESTING: set_lumping_from_str. Can we lump the states and still get the correct distribution?")
        base = Path("04-output_wrapper/C_0.75_2ps/05-2ps")
        file_list = [base / f"{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        k_model = kinetics.Sf_model(file_list)
        self.assertEqual(len(k_model.state_map_int_2_s), 16)
        self.assertEqual(len(k_model.state_map_s_2_int), 16)

        # lump the 2nd and 3rd most popular state, so that they become the most popular state
        def check_answer1():
            self.assertEqual(len(k_model.state_map_int_2_s), 16)
            self.assertEqual(len(k_model.state_map_s_2_int), 16)
            self.assertEqual(len(k_model.node_map_int_2_s), 15)
            self.assertEqual(len(k_model.node_map_s_2_int), 16)
            self.assertEqual(len(k_model.node_distribution_str), 15)
            self.assertEqual(len(k_model.node_distribution), 15)
            self.assertAlmostEqual(sum(k_model.node_distribution_str.values()), 1.0)
            self.assertAlmostEqual(sum(k_model.node_distribution.values()), 1.0)
            self.assertAlmostEqual(k_model.node_distribution_str[("KK0KKW", "WK0KKW")], 353 / 1002)
            self.assertAlmostEqual(k_model.node_distribution_str[("WKK0KW",)], 322 / 1002)
            self.assertAlmostEqual(k_model.node_distribution_str[("KKK0KW",)], 89 / 1002)
            self.assertAlmostEqual(k_model.node_distribution_str[("WKK0KK",)], 74 / 1002)
            self.assertAlmostEqual(k_model.node_distribution[0], 353 / 1002)
            self.assertAlmostEqual(k_model.node_distribution[1], 322 / 1002)
            self.assertAlmostEqual(k_model.node_distribution[2], 89 / 1002)
            self.assertAlmostEqual(k_model.node_distribution[3], 74 / 1002)
            self.assertListEqual(k_model.traj_node[0][:5].tolist(), [0, 0, 0, 1, 1])
            self.assertListEqual(k_model.traj_node[1][:5].tolist(), [2, 0, 0, 0, 0])
            self.assertListEqual(k_model.node_map_int_2_s[0], ["KK0KKW", "WK0KKW"])
            self.assertListEqual(k_model.node_map_int_2_s[1], ["WKK0KW"])
            self.assertListEqual(k_model.node_map_int_2_s[2], ["KKK0KW"])
            self.assertEqual(k_model.node_map_s_2_int["KK0KKW"], 0)
            self.assertEqual(k_model.node_map_s_2_int["WK0KKW"], 0)
            self.assertEqual(k_model.node_map_s_2_int["WKK0KW"], 1)
            self.assertEqual(k_model.node_map_s_2_int["KKK0KW"], 2)

        k_model.set_lumping_from_str([("KK0KKW", "WK0KKW")])
        check_answer1()
        k_model.set_lumping_from_int([(1, 2)])
        check_answer1()

        # more lumping
        def check_answer2():
            self.assertEqual(len(k_model.state_map_int_2_s), 16)
            self.assertEqual(len(k_model.state_map_s_2_int), 16)
            self.assertEqual(len(k_model.node_map_int_2_s), 14)
            self.assertEqual(len(k_model.node_map_s_2_int), 16)
            self.assertEqual(len(k_model.node_distribution_str), 14)
            self.assertEqual(len(k_model.node_distribution), 14)
            self.assertAlmostEqual(sum(k_model.node_distribution_str.values()), 1.0)
            self.assertAlmostEqual(sum(k_model.node_distribution.values()), 1.0)
            self.assertAlmostEqual(k_model.node_distribution_str[("WKK0KW", "KK0KKW")], 529 / 1002)
            self.assertAlmostEqual(k_model.node_distribution_str[("WK0KKW", "KKK0KW")], 235 / 1002)
            self.assertAlmostEqual(k_model.node_distribution_str[("WKK0KK",)], 74 / 1002)
            self.assertAlmostEqual(k_model.node_distribution[0], 529 / 1002)
            self.assertAlmostEqual(k_model.node_distribution[1], 235 / 1002)
            self.assertAlmostEqual(k_model.node_distribution[2], 74 / 1002)
            self.assertListEqual(k_model.traj_node[0][:5].tolist(), [1, 1, 1, 0, 0])
            self.assertListEqual(k_model.traj_node[1][:5].tolist(), [1, 0, 0, 0, 0])
            self.assertListEqual(k_model.node_map_int_2_s[0], ["WKK0KW", "KK0KKW"])
            self.assertListEqual(k_model.node_map_int_2_s[1], ["WK0KKW", "KKK0KW"])
            self.assertListEqual(k_model.node_map_int_2_s[2], ["WKK0KK"])
            self.assertEqual(k_model.node_map_s_2_int["WKK0KW"], 0)
            self.assertEqual(k_model.node_map_s_2_int["KK0KKW"], 0)
            self.assertEqual(k_model.node_map_s_2_int["WK0KKW"], 1)
            self.assertEqual(k_model.node_map_s_2_int["KKK0KW"], 1)
            self.assertEqual(k_model.node_map_s_2_int["WKK0KK"], 2)

        k_model.set_lumping_from_str([("WK0KKW", "KKK0KW"), ("WKK0KW", "KK0KKW")])
        check_answer2()
        k_model.set_lumping_from_int([(0, 1), (2, 3)])
        check_answer2()
        k_model.set_lumping_from_int([[1, 2]])
        check_answer1()

    def test_calc_properties(self):
        print("TESTING: flux, net_flux, transition_probability")
        traj_l = ["A A B A B A B C C".split(),
                  "A B A B A B A C C".split(),
                  "A A A A A A B C C".split()]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=1.0)
        self.assertListEqual(k_model.flux_matrix.tolist(), [[6, 7, 1],
                                                            [5, 0, 2],
                                                            [0, 0, 3]])
        self.assertListEqual(k_model.flux_matrix_every_traj[0].tolist(), [[1, 3, 0],
                                                                          [2, 0, 1],
                                                                          [0, 0, 1]])
        self.assertListEqual(k_model.transition_probability_matrix.tolist(), [[6 / 14, 7 / 14, 1 / 14],
                                                                              [5 / 7, 0, 2 / 7],
                                                                              [0, 0, 1]])
        self.assertListEqual(k_model.transition_probability_matrix_every_traj[0].tolist(), [[1 / 4, 3 / 4, 0],
                                                                                            [2 / 3, 0, 1 / 3],
                                                                                            [0, 0, 1]])
        self.assertListEqual(k_model.net_flux_matrix.tolist(), [[0, 2, 1],
                                                                [-2, 0, 2],
                                                                [-1, -2, 0]])
        self.assertListEqual(k_model.net_flux_matrix_every_traj[0].tolist(), [[0, 1, 0],
                                                                              [-1, 0, 1],
                                                                              [0, -1, 0]])
        k_model.set_lumping_from_str([("A", "B")])
        self.assertListEqual(k_model.flux_matrix.tolist(), [[18, 3],
                                                            [0, 3]])
        self.assertListEqual(k_model.transition_probability_matrix.tolist(), [[18 / 21, 3 / 21],
                                                                              [0, 1]])
        self.assertListEqual(k_model.net_flux_matrix.tolist(), [[0, 3],
                                                                [-3, 0]])

    def test_calc_passage_time(self):
        print("TESTING: calc_passage_time.")
        traj_l = ["A A A A A B B B A B A".split(),
                  "A A B C A B C A B C A B".split(),
                  "A A B C A A B C A B C A B".split()]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=1)
        k_model.calc_passage_time()
        passage_track_alltraj = k_model.passage_time_length_every_traj
        self.assertListEqual(passage_track_alltraj[0], [
            [[],     [5, 1], []],
            [[3, 1],     [], []],
            [[],         [], []]
        ])
        self.assertListEqual(passage_track_alltraj[1], [
            [[],     [2,1,1,1], [3,2,2]],
            [[2,2,2],       [], [1,1,1]],
            [[1,1,1],  [2,2,2],       []]
        ])
        self.assertListEqual(passage_track_alltraj[2], [
            [[],     [2,2,1,1], [3,3,2]],
            [[2,2,2],       [], [1,1,1]],
            [[1,1,1],  [3,2,2],       []]
        ])
        mfpt, mfpt_every_traj = k_model.get_mfpt()
        self.assertListEqual(mfpt.tolist(), [[0, 1.7, 2.5],
                                             [2, 0, 1],
                                             [1, 13/6, 0]])
        assert_allclose(mfpt_every_traj[0], [[0., 3., np.nan],
                                             [2., 0., np.nan],
                                             [np.nan, np.nan, 0.]], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
