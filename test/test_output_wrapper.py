import unittest
import Sfilter
from Sfilter import Perm_event_output
from Sfilter import Cylinder_output
from Sfilter.util.output_wrapper import read_k_cylinder
from Sfilter.util.output_wrapper import read_k_cylinder_list

class MyTestCase(unittest.TestCase):
    def test_line_to_state(self):
        lines = [" POT : , Wat : 49610 58583 ,",
                 " POT : 5962 , Wat : ,\n",
                 " POT : , Wat : ,\n",
                 " POT : 5961 , Wat : ,\n",
                 " POT : 5960 , Wat : ,\n",
                 " POT : 5959 , Wat : 29864 38630 40004 40109 42965 43178 44273 44456 44522 46910 47849 48983 55532 60080 60440 ,\n",]
        for l, state_str_ref, n_pot_ref, n_wat_ref  in zip(lines, ["W", "K", "0", "K", "K", "C"], [0, 1, 0, 1, 1, 1], [2, 0, 0, 0, 0, 15]):
            state_str, n_pot, n_wat = Sfilter.util.output_wrapper.line_to_state(l)
            self.assertEqual(state_str, state_str_ref)
            self.assertEqual(n_pot, n_pot_ref)
            self.assertEqual(n_wat, n_wat_ref)

    def test_Perm_event_output_read_file(self):
        files = ["03-longest_common_sequence/03-state-code/POT_perm_event.out",
                 "04-output_wrapper/POT_event.out"]
        p_list = Perm_event_output(files)
        self.assertListEqual(p_list.perm_list[1]["time"].tolist()[-3:], [2000., 2040., 2100.])
        self.assertListEqual(p_list.perm_list[1]["up"].tolist()[-3:], [True, False, True])


    def test_cut_event(self):
        files = "03-longest_common_sequence/03-state-code/POT_perm_event.out"
        p_list = Perm_event_output(files)
        self.assertEqual(len(p_list.perm_list[0]), 94)
        self.assertAlmostEqual(p_list.meta_data[0]["time"], 5e5)
        p_list.cut_event(0, 1e5)
        self.assertEqual(len(p_list.perm_list[0]), 21)
        self.assertAlmostEqual(p_list.meta_data[0]["time"], 1e5)

    def test_Perm_event_output_get_conductance(self):
        files = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02d}/analysis/04-state-code/POT_perm_event.out" for i in range(2)]
        p_list = Perm_event_output(files)
        conductance, cond_SEM, c_list = p_list.get_conductance()


    def test_cylinder_output_init(self):
        file = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        cylinder_list = Cylinder_output(file)
        self.assertEqual(len(cylinder_list.state_str[0]), 501)
        self.assertAlmostEqual(cylinder_list.meta_data[0]["time_step"], 2)
        self.assertListEqual(cylinder_list.state_str[0][:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
        self.assertListEqual(cylinder_list.state_str[1][:3], ["KKK0KW", "KK0KKW", "KK0KKW"])
        cylinder_list = Cylinder_output(file, step=2)
        self.assertListEqual(cylinder_list.state_str[0][:2], ["WK0KKW", "WK0KKW"])
        self.assertListEqual(cylinder_list.state_str[1][:2], ["KKK0KW", "KK0KKW"])
        self.assertEqual(len(cylinder_list.state_str[0]), 251)
        self.assertAlmostEqual(cylinder_list.meta_data[0]["time_step"], 4)


    def test_cylinder_output_init(self):
        file = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        cylinder_list = Cylinder_output(file)
        propotion, counter_list, counter_all = cylinder_list.get_state_distribution()
        #print(propotion[0])
        #print(propotion[1])
        propotion, counter_list, counter_all = cylinder_list.get_state_distribution("A")
        self.assertListEqual(propotion[0], [["A", 0]])

    def test_get_state_distribution_CI_bootstrap(self):
        file = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        cylinder_list = Cylinder_output(file)
        state_distribution = cylinder_list.get_state_distribution_CI_bootstrap_frame(n_resamples=200)
        self.assertListEqual(state_distribution[0][0][:2], ["WKK0KW", 201 / 501])
        self.assertListEqual(state_distribution[1][0][:2], ["WKK0KW", 121 / 501])
        self.assertListEqual(state_distribution[0][1][:2], ["KK0KKW", 41  / 501])
        self.assertListEqual(state_distribution[1][1][:2], ["KK0KKW", 166 / 501])

    def test_get_state_distribution_CI_bootstrap_traj(self):
        file = [f"04-output_wrapper/C_0.75_2ps/05-2ps/{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        cylinder_list = Cylinder_output(file)
        state_distribution = cylinder_list.get_state_distribution_CI_bootstrap_traj(n_resamples=200)
        #print(state_distribution)
        self.assertListEqual(state_distribution[0][:2], ["WKK0KW", (201 + 121) / 1002])
        self.assertListEqual(state_distribution[1][:2], ["KK0KKW", (41 + 166) / 1002])

    def test_s6l_function1_original(self):
        l = "# S6l 0 WKKK0K\n"
        self.assertEqual(Sfilter.util.output_wrapper._s6l_function1_original(l), "WKKK0K")
        l = "# S6l 1 K0KKKW"
        self.assertEqual(Sfilter.util.output_wrapper._s6l_function1_original(l), "K0KKKW")

    def test_s6l_function2_nonK(self):
        l = "# S6l 0 Wat , SOD , SOD , SOD , , SOD Wat ,\n"
        self.assertEqual(Sfilter.util.output_wrapper._s6l_function2_nonK(l, ion="SOD"), "WKKK0K")
        l = "# S6l 1 SOD Wat , , SOD , SOD , SOD , Wat ,"
        self.assertEqual(Sfilter.util.output_wrapper._s6l_function2_nonK(l, ion="SOD"), "K0KKKW")

    def test_read_k_cylinder(self):
        print("# TESTING read_k_cylinder, see if we can read the correct 6 letter code and meta data")
        for get_occu in [True, False]:
            s_list, meta_data, K_occupency, W_occupency = read_k_cylinder("04-output_wrapper/C_0.75_2ps/05-2ps/00"
                                                                          "/analysis/04-state-code/k_cylinder.log",
                                                                          get_occu=get_occu)
            self.assertListEqual(s_list[:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
            # meta_data
            self.assertDictEqual(meta_data,
                                 {"time_step": 2.0, })

    def test_read_k_cylinder_SOD(self):
        print("# TESTING read_k_cylinder, SOD")
        s_list1, mdata1, K_occup1, W_occup1 = read_k_cylinder("01-NaK2K/1-Charmm/SOD/k_cylinder_POT.log", get_occu=True)
        s_list2, mdata2, K_occup2, W_occup2 = read_k_cylinder("01-NaK2K/1-Charmm/SOD/k_cylinder_SOD.log", get_occu=True)
        self.assertListEqual(s_list1, s_list2)
        self.assertListEqual(s_list1[:5], ["WKKK0K", "K0KKKW", "K0KKKW", "K0KKKW", "K0KKKW"])
        s_list1, mdata1, K_occup1, W_occup1 = read_k_cylinder("01-NaK2K/1-Charmm/SOD/k_cylinder_POT.log", get_occu=False)
        s_list2, mdata2, K_occup2, W_occup2 = read_k_cylinder("01-NaK2K/1-Charmm/SOD/k_cylinder_SOD.log", get_occu=False)
        self.assertListEqual(s_list1, s_list2)
        self.assertListEqual(s_list1[:5], ["WKKK0K", "K0KKKW", "K0KKKW", "K0KKKW", "K0KKKW"])

    def test_read_state_file_co_occupy(self):
        print("# TESTING read_k_cylinder, Co-occupy. 6 letter code contains K, W, and C")
        for get_occu in [True, False]:
            s_list, meta_data, K_occupency, W_occupency = read_k_cylinder("04-output_wrapper/C_0.75_2ps/05-2ps/00"
                                                                          "/analysis/04-state-code/k_cylinder.log",
                                                                          method="Co-occupy",
                                                                          get_occu=get_occu)
            self.assertListEqual(s_list[:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
            self.assertListEqual(s_list[405:408], ["WK0KKC", "WK0KKC", "WKK0KW"])
            self.assertDictEqual(meta_data,
                                 {"time_step": 2.0, })

    def test_read_k_cylinder_list(self):
        print("# TESTING read_k_cylinder_list, read a sequence of k_cylinder.log files")
        f_list = ["04-output_wrapper/C_0.75_2ps/05-2ps/00/analysis/04-state-code/k_cylinder.log",
                  "04-output_wrapper/C_0.75_2ps/05-2ps/01/analysis/04-state-code/k_cylinder.log"
                  ]
        for get_occu in [False, True]:
            s_list, meta_data, K_occupency, W_occupency = read_k_cylinder_list(f_list, get_occu=get_occu)
            self.assertListEqual(s_list[:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
            self.assertEqual(len(s_list), 1001)
            self.assertDictEqual(meta_data,
                                 {"time_step": 2.0, })
        self.assertEqual(len(K_occupency), 1001)
        self.assertListEqual(K_occupency[:3].tolist(),
                             [
                                 [0, 1, 0, 1, 1, 0],
                                 [0, 1, 0, 1, 1, 0],
                                 [0, 1, 0, 1, 1, 0]
                             ])
        self.assertEqual(len(W_occupency), 1001)
        self.assertListEqual(W_occupency[:3].tolist(),
                             [
                                 [2, 0, 0, 0, 0, 15],
                                 [2, 0, 0, 0, 0, 14],
                                 [2, 0, 0, 0, 0, 12]
                             ])

        self.assertListEqual(s_list[500:503], ["WKK0KK", "KK0KKW", "KK0KKW"])
        self.assertListEqual(K_occupency[500:503].tolist(),
                             [
                                 [0, 1, 1, 0, 1, 1],
                                 [1, 1, 0, 1, 1, 0],
                                 [1, 1, 0, 1, 1, 0]
                             ])
        self.assertListEqual(W_occupency[500:503].tolist(),
                             [
                                 [2, 0, 0, 0, 0, 11],
                                 [0, 0, 0, 0, 0, 14],
                                 [0, 0, 0, 0, 0, 9]
                             ])




if __name__ == '__main__':
    unittest.main()
