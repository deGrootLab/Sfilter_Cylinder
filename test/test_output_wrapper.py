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

    def test_read_k_cylinder(self):
        print("# TESTING read_k_cylinder, see if we can read the correct 6 letter code and meta data")
        s_list, meta_data, K_occupency, W_occupency = read_k_cylinder("04-output_wrapper/C_0.75_2ps/05-2ps/00"
                                                                      "/analysis/04-state-code/k_cylinder.log")
        self.assertListEqual(s_list[:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
        # meta_data
        self.assertDictEqual(meta_data,
                             {"time_step": 2.0, })

    def test_read_state_file_co_occupy(self):
        print("# TESTING read_k_cylinder, Co-occupy. 6 letter code contains K, W, and C")
        s_list, meta_data, K_occupency, W_occupency = read_k_cylinder("04-output_wrapper/C_0.75_2ps/05-2ps/00"
                                                                      "/analysis/04-state-code/k_cylinder.log",
                                                                      method="Co-occupy")
        self.assertListEqual(s_list[:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
        self.assertListEqual(s_list[405:408], ["WK0KKC", "WK0KKC", "WKK0KW"])
        self.assertDictEqual(meta_data,
                             {"time_step": 2.0, })

    def test_read_k_cylinder_list(self):
        print("# TESTING read_k_cylinder_list, read a sequence of k_cylinder.log files")
        f_list = ["04-output_wrapper/C_0.75_2ps/05-2ps/00/analysis/04-state-code/k_cylinder.log",
                  "04-output_wrapper/C_0.75_2ps/05-2ps/01/analysis/04-state-code/k_cylinder.log"
                  ]
        s_list, meta_data, K_occupency, W_occupency = read_k_cylinder_list(f_list)
        self.assertListEqual(s_list[:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
        self.assertEqual(len(s_list), 1001)
        self.assertDictEqual(meta_data,
                             {"time_step": 2.0, })
        self.assertEqual(len(K_occupency), 1001)
        self.assertListEqual(K_occupency[:3],
                             [
                                 [0, 1, 0, 1, 1, 0],
                                 [0, 1, 0, 1, 1, 0],
                                 [0, 1, 0, 1, 1, 0]
                             ])
        self.assertEqual(len(W_occupency), 1001)
        self.assertListEqual(W_occupency[:3],
                             [
                                 [2, 0, 0, 0, 0, 15],
                                 [2, 0, 0, 0, 0, 14],
                                 [2, 0, 0, 0, 0, 12]
                             ])

        self.assertListEqual(s_list[500:503], ["WKK0KK", "KK0KKW", "KK0KKW"])
        self.assertListEqual(K_occupency[500:503],
                             [
                                 [0, 1, 1, 0, 1, 1],
                                 [1, 1, 0, 1, 1, 0],
                                 [1, 1, 0, 1, 1, 0]
                             ])
        self.assertListEqual(W_occupency[500:503],
                             [
                                 [2, 0, 0, 0, 0, 11],
                                 [0, 0, 0, 0, 0, 14],
                                 [0, 0, 0, 0, 0, 9]
                             ])




if __name__ == '__main__':
    unittest.main()
