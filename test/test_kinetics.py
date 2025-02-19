import time
import unittest
import numpy as np
from Sfilter import kinetics
from pathlib import Path
from Sfilter import read_k_cylinder
from numpy.testing import assert_allclose
from Sfilter import Perm_event_output


def assert_2d_int_arrays_equal(arrays1, arrays2):
    """
    Assert that two 2D lists of NumPy arrays are equal element-wise.

    Parameters:
    - arrays1: First 2D list of NumPy arrays.
    - arrays2: Second 2D list of NumPy arrays.

    Raises:
    - AssertionError: If the arrays are not equal element-wise.
    """
    assert len(arrays1) == len(arrays2), "Number of rows must be the same."
    for row1, row2 in zip(arrays1, arrays2):
        assert len(row1) == len(row2), "Number of columns must be the same in each row."
        for arr1, arr2 in zip(row1, row2):
            np.testing.assert_array_equal(arr1, arr2)


class MyTestCase(unittest.TestCase):
    def test_assert_2d_int_arrays_equal(self):
        print("#TESTING: assert_2d_int_arrays_equal, a helper function to assert 2d arrays equal")
        arr1 = [[np.array([1, 2, 3]), np.array([4, 5, 6])],
                [np.array([7, 8, 9]), np.array([10, 11, 12])]]
        arr2 = [[np.array([1, 2, 3]), np.array([4, 5, 6])],
                [np.array([7, 8, 9]), np.array([10, 11, 12])]]
        arr3 = [[np.array([1, 2, 3]), np.array([4, 5, 6])],
                [np.array([7, 8, 9]), np.array([10, 11, 13])]]
        assert_2d_int_arrays_equal(arr1, arr2)
        self.assertRaises(AssertionError, assert_2d_int_arrays_equal, arr1, arr3)

    def test_init(self):
        print("#TESTING: init. Init the Sf_model class in the normal way.")
        base = Path("04-output_wrapper/C_0.75_2ps/05-2ps")
        file_list = [base / f"{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        k_model = kinetics.Sf_model(file_list)
        self.assertAlmostEqual(k_model.time_step, 2.0)
        self.assertAlmostEqual(k_model.state_distribution["WKK0KW"], 322 / 1002)
        self.assertListEqual(k_model.state_distribution_alltraj["WKK0KW"].tolist(), [201/501, 121/501])
        self.assertAlmostEqual(k_model.state_distribution["KK0KKW"], 207 / 1002)
        self.assertListEqual(k_model.state_distribution_alltraj["KK0KKW"].tolist(), [41 / 501, 166 / 501])
        self.assertAlmostEqual(k_model.state_distribution["WK0KKW"], 146 / 1002)
        self.assertAlmostEqual(k_model.state_distribution["KKK0KW"], 89 / 1002)
        self.assertAlmostEqual(k_model.state_distribution["WKK0KK"], 74 / 1002)
        self.assertEqual(k_model.total_frame, 1002)
        self.assertListEqual(k_model.traj_raw_alltraj[0][:5].tolist(), [2, 2, 2, 0, 0])
        self.assertListEqual(k_model.traj_raw_alltraj[1][:5].tolist(), [3, 1, 1, 1, 1])

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
        self.assertListEqual(k_model.traj_raw_alltraj[0][:3].tolist(), [2, 2, 0])
        self.assertListEqual(k_model.traj_raw_alltraj[1][:3].tolist(), [3, 1, 1])

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
        print("#TESTING: set_lumping_from_str. Can we lump the states and still get the correct distribution?")
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

    def test_calc_properties(self):
        print("#TESTING: flux, net_flux, transition_probability")
        traj_l = ["A A B A B A B C C".split(),
                  "A B A B A B A C C".split(),
                  "A A A A A A B C C".split()]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=1.0)
        self.assertListEqual(k_model.flux_matrix.tolist(), [[6, 7, 1],
                                                            [5, 0, 2],
                                                            [0, 0, 3]])
        self.assertListEqual(k_model.flux_matrix_alltraj[0].tolist(), [[1, 3, 0],
                                                                       [2, 0, 1],
                                                                       [0, 0, 1]])
        self.assertListEqual(k_model.transition_probability_matrix.tolist(), [[6 / 14, 7 / 14, 1 / 14],
                                                                              [5 / 7, 0, 2 / 7],
                                                                              [0, 0, 1]])
        self.assertListEqual(k_model.transition_probability_matrix_alltraj[0].tolist(), [[1 / 4, 3 / 4, 0],
                                                                                         [2 / 3, 0, 1 / 3],
                                                                                         [0, 0, 1]])
        self.assertListEqual(k_model.net_flux_matrix.tolist(), [[0, 2, 1],
                                                                [-2, 0, 2],
                                                                [-1, -2, 0]])
        self.assertListEqual(k_model.net_flux_matrix_alltraj[0].tolist(), [[0, 1, 0],
                                                                           [-1, 0, 1],
                                                                           [0, -1, 0]])
        k_model.set_lumping_from_str([("A", "B")])
        self.assertListEqual(k_model.flux_matrix.tolist(), [[18, 3],
                                                            [0, 3]])
        self.assertListEqual(k_model.transition_probability_matrix.tolist(), [[18 / 21, 3 / 21],
                                                                              [0, 1]])
        self.assertListEqual(k_model.net_flux_matrix.tolist(), [[0, 3],
                                                                [-3, 0]])

    def test_count_passage(self):
        print("#TESTING: count_passage_single_traj/count_passage. This function count passage time from a list of trajectory.")
        #                  0  1  2  3  4  5  6  7  8  9 10
        traj_s = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
        traj_l = [traj_s]

        ptime_len_s, ptime_point_s = kinetics.count_passage_single_traj(traj_s, 2)
        ptime_len_l, ptime_point_l = kinetics.count_passage(traj_l, 2)

        for ptime_len, ptime_point in ((ptime_len_l, ptime_point_l),
                                       ([ptime_len_s], [ptime_point_s])):
            answer = [
                [[], [5, 1]],
                [[3, 1], []],
            ]
            answer = [[[np.array(i) for i in j] for j in k] for k in answer]
            assert_2d_int_arrays_equal(ptime_len[0], answer)
            answer = [
                [[], [5, 9]],
                [[8, 10], []],
            ]
            answer = [[[np.array(i) for i in j] for j in k] for k in answer]
            assert_2d_int_arrays_equal(ptime_point[0], answer)

    def test_calc_passage_time(self):
        print("#TESTING: calc_passage_time.")
        traj_l = ["A A A A A B B B A B A".split(),
                  "A A B C A B C A B C A B".split(),
                  "A A B C A A B C A B C A B".split()]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=1)
        k_model.calc_passage_time()
        passage_track_alltraj = k_model.passage_time_length_alltraj
        answer = [
            [[], [5, 1], []],
            [[3, 1], [], []],
            [[], [], []]
        ]
        answer = [[[np.array(i) for i in j] for j in k] for k in answer]
        assert_2d_int_arrays_equal(passage_track_alltraj[0], answer)

        answer = [
            [[], [2, 1, 1, 1], [3, 2, 2]],
            [[2, 2, 2], [], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2], []]
        ]
        answer = [[[np.array(i) for i in j] for j in k] for k in answer]
        assert_2d_int_arrays_equal(passage_track_alltraj[1], answer)

        answer = [
            [[], [2, 2, 1, 1], [3, 3, 2]],
            [[2, 2, 2], [], [1, 1, 1]],
            [[1, 1, 1], [3, 2, 2], []]
        ]
        answer = [[[np.array(i) for i in j] for j in k] for k in answer]
        assert_2d_int_arrays_equal(passage_track_alltraj[2], answer)

        mfpt, mfpt_every_traj = k_model.get_mfpt()
        self.assertListEqual(mfpt.tolist(), [[0, 1.7, 2.5],
                                             [2, 0, 1],
                                             [1, 13 / 6, 0]])
        assert_allclose(mfpt_every_traj[0], [[0., 3., np.nan],
                                             [2., 0., np.nan],
                                             [np.nan, np.nan, 0.]], equal_nan=True)

    def test_calc_flux_matrix(self):
        print("#TESTING: calc_flux_matrix.")
        traj_l = ["A A A A A B B B A B A".split(),
                  "A A B C A B C A B C A B".split()]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=1)
        self.assertListEqual(k_model.flux_matrix.tolist(),
                             [[5, 6, 0], [2, 2, 3], [3, 0, 0]])
        self.assertListEqual(k_model.flux_matrix_alltraj[0].tolist(), [[4, 2, 0], [2, 2, 0], [0, 0, 0]])
        self.assertListEqual(k_model.flux_matrix_alltraj[1].tolist(), [[1, 4, 0], [0, 0, 3], [3, 0, 0]])

    def test_get_rate_inverse_mfpt(self):
        print("#TESTING: get_rate_inverse_mfpt. This function calculate the rate from every node to every node")
        traj_l = ["A A B A A B A A B A".split(),
                  "A A B A A B A A B".split()]
        for traj_type in ["lumped", "raw"]:
            k_model = kinetics.Sf_model()
            k_model.set_traj_from_str(traj_l, time_step=1)
            k_model.calc_passage_time()
            rate, rate_every_traj = k_model.get_rate_inverse_mfpt(traj_type)
            for i in [rate, rate_every_traj[0], rate_every_traj[1]]:
                self.assertListEqual(i.tolist(), [
                    [0, 1 / 2],
                    [1, 0]])

        traj_l = ["A A A B C A A A C B A".split(),
                  "A A A C B A A A B B".split()]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=1)
        k_model.set_lumping_from_str([("B", "C")], calc_passage_time=True)
        rate, rate_every_traj = k_model.get_rate_inverse_mfpt("lumped")
        self.assertListEqual(rate.tolist(), [
            [0, 4 / 12],
            [3 / 6, 0]])
        self.assertListEqual(rate_every_traj[0].tolist(), [
            [0, 2 / 6],
            [2 / 4, 0]])
        self.assertListEqual(rate_every_traj[1].tolist(), [
            [0, 2 / 6],
            [1 / 2, 0]])
        rate, rate_every_traj = k_model.get_rate_inverse_mfpt("raw")
        self.assertListEqual(rate.tolist(), [
            [0, 4 / 14, 3 / 10],
            [3 / 4, 0, 1 / 1],
            [3 / 5, 2 / 6, 0]])
        self.assertListEqual(rate_every_traj[0].tolist(), [
            [0, 2 / 7, 2 / 7],
            [2 / 3, 0, 1 / 1],
            [2 / 3, 1 / 5, 0]])

    def test_get_rate_passage_time(self):
        print("#TESTING: get_rate_passage_time. This function calculate the rate from everything to everything")
        traj_l = ["A A B A A B A A B A".split(),
                  "A A B A A B A A B".split()]
        for traj_type in ["lumped", "raw"]:
            k_model = kinetics.Sf_model()
            k_model.set_traj_from_str(traj_l, time_step=1)
            k_model.calc_passage_time()
            rate, rate_every_traj = k_model.get_rate_passage_time(traj_type)
            self.assertListEqual(rate.tolist(),
                                 [[0, 6 / 13],
                                  [5 / 6, 0]])
            self.assertListEqual(rate_every_traj[0].tolist(),
                                 [[0, 3 / 7],
                                  [3 / 3, 0]])
            self.assertListEqual(rate_every_traj[1].tolist(),
                                 [[0, 3 / 6],
                                  [2 / 3, 0]])

        traj_l = ["A A A B C A A A C B A".split(),
                  "A A A C B A A A B B".split()]
        k_model = kinetics.Sf_model()
        k_model.set_traj_from_str(traj_l, time_step=1)
        k_model.set_lumping_from_str([("B", "C")], calc_passage_time=True)
        rate, rate_every_traj = k_model.get_rate_passage_time("lumped")
        self.assertListEqual(rate.tolist(), [
            [0, 4 / 13],
            [3 / 8, 0]])
        self.assertListEqual(rate_every_traj[0].tolist(), [
            [0, 2 / 7],
            [2 / 4, 0]])
        self.assertListEqual(rate_every_traj[1].tolist(), [
            [0, 2 / 6],
            [1 / 4, 0]])
        rate, rate_every_traj = k_model.get_rate_passage_time("raw")
        self.assertListEqual(rate.tolist(), [
            [0, 4 / 13, 3 / 13],
            [3 / 5, 0, 1 / 5],
            [3 / 3, 2 / 3, 0]])
        self.assertListEqual(rate_every_traj[0].tolist(), [
            [0, 2 / 7, 2 / 7],
            [2 / 2, 0, 1 / 2],
            [2 / 2, 1 / 2, 0]])
        self.assertListEqual(rate_every_traj[1].tolist(), [
            [0, 2 / 6, 1 / 6],
            [1 / 3, 0, 0 / 3],
            [1 / 1, 1 / 1, 0]])

    def test_k_distance(self):
        a_string = "K0KK"
        b_string = "KK0K"
        self.assertAlmostEqual(kinetics.k_distance(a_string, b_string)[0], 1 / 5)
        self.assertAlmostEqual(kinetics.k_distance(a_string, a_string)[0], 0)

        a_string = "WK0KK"
        b_string = "WKK0K"
        self.assertAlmostEqual(kinetics.k_distance(a_string, b_string)[0], 1 / 6)

        a_string = "KK0KK"
        b_string = "0KK0K"
        self.assertAlmostEqual(kinetics.k_distance(a_string, b_string)[0], 2 / 6)
        self.assertAlmostEqual(kinetics.k_distance(b_string, a_string)[0], - 2 / 6)

        a_string = "K0KK0"
        b_string = "KK0KK"
        self.assertAlmostEqual(kinetics.k_distance(a_string, b_string)[0], 2 / 6)
        self.assertAlmostEqual(kinetics.k_distance(b_string, a_string)[0], - 2 / 6)

        a_string = "WKK0K0"
        b_string = "K0KKK0"
        self.assertAlmostEqual(kinetics.k_distance(a_string, b_string)[0], -3 / 7)
        self.assertAlmostEqual(kinetics.k_distance(b_string, a_string)[0], 3 / 7)


    def test_performance(self):
        print("# TEST Speed")
        base = Path.home()/"E29Project-2023-04-11/076-MthK/20-C-POT-iso_radius/22-0.78/07_+150_0.2ps/"
        f_list = [base/f"{i:02d}/analysis/08-02-1.8A/k_cylinder.log"  for i in range(2)]

        if f_list[0].is_file() and f_list[1].is_file():
            print("Run speed test")
            # clear cache file
            cache_list = [base / f"{i:02d}/analysis/08-02-1.8A/k_cylinder.jump_np_array.npy" for i in range(2)]
            # for f in cache_list:
            #     if f.is_file():
            #         f.unlink()
            t0 = time.time()
            k_model = kinetics.Sf_model(f_list, print_time=True)
            t1 = time.time()
            print(f"{t1 - t0:.2f} s")

            # redo, cache will be used. It can be 40 times faster
            k_model = kinetics.Sf_model(f_list, print_time=True)
            t2 = time.time()
            print(f"{t2 - t1:.2f} s")
        else:
            print("Speed test cannot be done. Input files are not found.")

    def test_rate_cycle_correct(self):
        print("# TEST rate_cycle_correct")
        base = Path("04-output_wrapper/C_0.75_2ps/05-2ps")
        file_list = [base / f"{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        k_model = kinetics.Sf_model(file_list)
        self.assertEqual(k_model.get_state_index("WKK0KW"), 0)
        self.assertEqual(k_model.get_state_index("KK0KKW"), 1)
        answer_dict = {
            -2: (
                [[152, 71], [4, 11]],
                [[3, 204], [103, 158]],
                [[155, 275], [107, 169]]
            ),
            5: (
                [[], [69]],
                [[], [172]],
                [[], [241]]
            )
        }
        for s_i, s_j in ((0,        1),
                         ("WKK0KW", "KK0KKW")):
            res_dict = k_model.get_passage_AB(s_i, s_j)
            self.assertDictEqual(res_dict, answer_dict)
            res_short, jump = k_model.get_passage_AB_shortest(s_i, s_j)
            self.assertEqual(res_short, answer_dict[-2])
            self.assertEqual(jump, -2)
            mfpt, mfpt_alltraj, jump = k_model.get_mfpt_AB_shortest_passage(s_i, s_j)
            self.assertAlmostEqual(mfpt, 119.0)
            self.assertListEqual(mfpt_alltraj, [223.0, 15.0])
            self.assertEqual(jump, -2)



if __name__ == '__main__':
    unittest.main()
