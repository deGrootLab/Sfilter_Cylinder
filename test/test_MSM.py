import unittest
from Sfilter import MSM
import numpy as np
from Sfilter import kinetics


class MyTestCase(unittest.TestCase):

    def test_SF_msm_init(self):
        file_list = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        msm = MSM.SF_msm(file_list)
        self.assertListEqual(msm.time_step, [2.0, 2.0])
        self.assertListEqual(msm.state_str[0][:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
        self.assertListEqual(msm.state_str[1][:3], ["KKK0KW", "KK0KKW", "KK0KKW"])

        msm = MSM.SF_msm(file_list, step=2)
        self.assertListEqual(msm.time_step, [4.0, 4.0])  # jump 1 frame
        self.assertListEqual(msm.state_str[0][:5], ["WK0KKW", "WK0KKW", "WKK0KW", "WK0KKW", "WKK0KW"])
        self.assertListEqual(msm.state_str[1][:5], ["KKK0KW", "KK0KKW", "KK0KKW", "KK0KKW", "KK0KKW"])

    def test_SF_msm_set_state_str(self):
        msm = MSM.SF_msm([])
        msm.set_state_str(["A B A B A B C".split(),
                           "B A B A B A C".split(),
                           "A A A A A B C".split()])
        msm.calc_state_array()
        self.assertDictEqual(msm.s_2_int, {"A": 0, "B": 1, "C": 2})
        self.assertDictEqual(msm.int_2_s, {0: ["A"], 1: ["B"], 2: ["C"]})
        self.assertListEqual(msm.state_array[0].tolist(), [0, 1, 0, 1, 0, 1, 2])
        self.assertAlmostEqual(msm.distribution[0], 11 / 21)
        self.assertAlmostEqual(msm.distribution[1], 7 / 21)
        self.assertAlmostEqual(msm.distribution[2], 3 / 21)
        self.assertDictEqual(msm.state_counter, {"A": 11, "B": 7, "C": 3})
        self.assertDictEqual(msm.node_counter, {0: 11, 1: 7, 2: 3})

        msm.calc_state_array(merge_list=[["A", "B"]])
        self.assertDictEqual(msm.s_2_int, {"A": 0, "B": 0, "C": 1})
        self.assertDictEqual(msm.int_2_s, {0: ["A", "B"], 1: ["C"]})
        self.assertListEqual(msm.state_array[0].tolist(), [0, 0, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(msm.distribution[0], 6 / 7)
        self.assertAlmostEqual(msm.distribution[1], 1 / 7)
        self.assertDictEqual(msm.state_counter, {"A": 11, "B": 7, "C": 3})
        self.assertDictEqual(msm.node_counter, {0: 18, 1: 3})

    def test_set_state_array(self):
        msm = MSM.SF_msm([])
        msm.set_state_array([np.array("1 0 1 0 1".split()),
                             np.array("1 0 1 0 0".split())])
        self.assertListEqual(msm.state_str[0], ["1", "0", "1", "0", "1"])
        self.assertListEqual(msm.state_str[1], ["1", "0", "1", "0", "0"])
        self.assertDictEqual(msm.state_counter, {"1": 5, "0": 5})
        self.assertDictEqual(msm.s_2_int, {"1": 1, "0": 0})
        self.assertDictEqual(msm.int_2_s, {0: ["0"], 1: ["1"]})

    def test_SF_msm_get_transition_matrix(self):
        msm = MSM.SF_msm([])
        msm.set_state_str(["A A B C A B C A B C A B C A B".split()])
        msm.calc_state_array()
        f_matrix_1 = msm.get_transition_matrix(lag_step=1)
        f_matrix_2 = msm.get_transition_matrix(lag_step=2)
        f_matrix_3 = msm.get_transition_matrix(lag_step=3)
        self.assertListEqual(f_matrix_1.tolist(), [[1, 5, 0], [0, 0, 4], [4, 0, 0]])
        self.assertListEqual(f_matrix_2.tolist(), [[0, 1, 4], [4, 0, 0], [0, 4, 0]])
        self.assertListEqual(f_matrix_3.tolist(), [[4, 0, 1], [0, 4, 0], [0, 0, 3]])
        msm.time_step = [1]
        r_1 = msm.get_rate_matrix(lag_step=1)
        r_2 = msm.get_rate_matrix(lag_step=2)
        r_3 = msm.get_rate_matrix(lag_step=3)
        self.assertListEqual(r_1.tolist(), [[0, 5/6, 0], [0, 0, 4/5], [4/4, 0, 0]])
        self.assertListEqual(r_2.tolist(), [[0, 1/6, 4/6], [4/5, 0, 0], [0, 4/4, 0]])
        self.assertListEqual(r_3.tolist(), [[0, 0, 1/6], [0, 0, 0], [0, 0, 0]])



    def test_SF_msm_get_matrix(self):
        msm = MSM.SF_msm([])
        msm.set_state_str(["A B A B A B C D".split(),
                           "B A B A B A D C".split(),
                           "A A A A A B C D".split()])
        msm.calc_state_array(merge_list=[["A", "B"], ["C", "D"]])
        t_matrix = msm.get_transition_matrix()
        self.assertListEqual(t_matrix.tolist(), [[15, 3],
                                                 [ 0, 3]])
        rate_matrix = msm.get_rate_matrix(lag_step=1, physical_time=2)
        self.assertListEqual(rate_matrix.tolist(), [[0, 3/(18*2)],
                                                    [0, 0]])
        p_matrix = msm.get_transition_probability(lag_step=1)
        self.assertListEqual(p_matrix.tolist(), [[15/18, 1/6],
                                                 [0, 1]])

        t_matrix, net_t_matrix, rate_matrix, p_matrix = msm.get_matrix(lag_step=1, physical_time=2)
        self.assertListEqual(t_matrix.tolist(), [[15, 3],
                                                 [0, 3]])
        self.assertListEqual(net_t_matrix.tolist(), [[0, 3],
                                                     [-3, 0]])
        self.assertListEqual(rate_matrix.tolist(), [[0, 3/(18*2)],
                                                    [0, 0]])
        self.assertListEqual(p_matrix.tolist(), [[15/18, 1/6],
                                                 [0, 1]])
    def test_get_CK_test(self):
        msm = MSM.SF_msm([])
        msm.set_state_str(["A B C A B C A B C A B C A".split(),
                           "B C A B C A B C A B C A".split(),
                           "A B C A B C A B C A B".split()])
        msm.calc_state_array(merge_list=[])
        reality, prediction = msm.get_CK_test(lag_step=1, test_time=[2])
        p0 = [[0., 1., 0.],
              [0., 0., 1.],
              [1., 0., 0.]]
        self.assertListEqual(reality[0].tolist(), p0)
        self.assertListEqual(prediction[0].tolist(), p0)
        self.assertListEqual(reality[1].tolist(), np.dot(p0, p0).tolist())
        self.assertListEqual(prediction[1].tolist(), np.dot(p0, p0).tolist())

        reality, prediction = msm.get_CK_test(lag_step=2, test_time=[2])
        self.assertListEqual(reality[0].tolist(), np.dot(p0, p0).tolist())
        self.assertListEqual(prediction[0].tolist(), np.dot(p0, p0).tolist())
        self.assertListEqual(reality[1].tolist(), np.linalg.matrix_power(p0, 4).tolist())
        self.assertListEqual(prediction[1].tolist(), np.linalg.matrix_power(p0, 4).tolist())





    def test_MFPT_A_to_B(self):
        traj_list = [(np.array("B A A C A A C C C B B".split()), [8]),
                     #            ^---------------^
                     (np.array("B C A A C A A C C C B B".split()), [8]),
                     #              ^---------------^
                     (np.array("A C A".split()), []),
                     (np.array("A B A A C A A C C C B B".split()), [1, 8]),
                     #          ^-^ ^---------------^
                     (np.array("A C A B C X A B A".split()), [3, 1]),
                     #          ^-----^     ^-^
                     (np.array("C C A C A B C X A B A".split()), [3, 1]),
                     #              ^-----^     ^-^
                     (np.array("C B A C A B C X A B A".split()), [3, 1]),
                     #              ^-----^     ^-^
                     ]
        for traj, answer in traj_list:
            mfpt_list = MSM.MFPT_A_to_B(traj, "A", "B")
            self.assertListEqual(mfpt_list, answer)
            model = kinetics.Sf_model()
            model.set_traj_from_str([traj.tolist()], time_step=1)
            passage_track_alltraj = model.calc_passage_time()
            if "B" in model.node_map_s_2_int:
                passage_track_A_B = passage_track_alltraj[0] [model.node_map_s_2_int['A']] [model.node_map_s_2_int['B']]
                self.assertListEqual(passage_track_A_B, answer)
            else:
                self.assertListEqual(answer, [])

    def test_MFPT_pair(self):
        file_list = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in
                     range(2)]
        msm = MSM.SF_msm(file_list)
        msm.calc_state_array()
        mfpt_01 = msm.get_MFPT_pair(0, 1)
        self.assertListEqual(mfpt_01[:2], [152, 71])

    def test_mfpt_kinetic_model(self):
        file_list = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in
                     range(2)]
        msm = MSM.SF_msm(file_list)
        msm.calc_state_array()
        mfpt_01 = msm.get_MFPT_pair(0, 1)
        model = kinetics.Sf_model(file_list)
        mfpt, mfpt_every_traj = model.get_mfpt()
        self.assertAlmostEqual(mfpt[0,1], np.mean(mfpt_01) * msm.time_step[0])
        self.assertListEqual(model.passage_time_length_every_traj[0][0][1][:2], mfpt_01[:2])
        self.assertListEqual(model.passage_time_length_every_traj[1][0][1], mfpt_01[2:])

    def test_MFPT_matrix(self):
        file_list = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in
                     range(2)]
        msm = MSM.SF_msm(file_list)
        msm.calc_state_array()
        mfpt_01 = msm.get_MFPT_pair(0, 1)
        MFPT_matrix, FPT_list1 = msm.get_MFPT_matrix()
        rate, FPT_list2 = msm.get_nfp_rate_matrix()
        self.assertListEqual(FPT_list1, FPT_list2)

    def test_population_cutoff(self):
        file_list = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in
                     range(2)]
        msm = MSM.SF_msm(file_list)
        msm.calc_state_array()
        n = msm.population_cutoff(0.01)
        self.assertEqual(msm.population_cutoff(0.01), 10)
        self.assertEqual(msm.population_cutoff(1/1002/2), 16)


    # plot
    def test_computer_pos(self):
        node = "WKKKWK"
        x, y = MSM.computer_pos(node, end=5)
        print(x, y)
        node = "K0KKW0"
        x, y = MSM.computer_pos(node, end=5)
        print(x, y)











if __name__ == '__main__':
    unittest.main()
