import unittest
import Sfilter
from Sfilter import Perm_event_output
from Sfilter import Cylinder_output
from Sfilter.util.output_wrapper import read_k_cylinder
from Sfilter.util.output_wrapper import read_k_cylinder_list
import time
from pathlib import Path

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
            self.assertDictEqual(
                meta_data,
                {"time_step": 2.0, "ion_name":"POT", "num_ion" : 160,
                 "ion_index": [5960, 5961, 5962, 5963, 5964, 5965, 5966, 5967, 5968, 5969, 5970, 5971, 5972, 5973,
                               5974, 5975, 5976, 5977, 5978, 5979, 5980, 5981, 5982, 5983, 5984, 5985, 5986, 5987,
                               5988, 5989, 5990, 5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999, 6000, 6001,
                               6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010, 6011, 6012, 6013, 6014, 6015,
                               6016, 6017, 6018, 6019, 6020, 6021, 6022, 6023, 6024, 6025, 6026, 6027, 6028, 6029,
                               6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6038, 6039, 6040, 6041, 6042, 6043,
                               6044, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6052, 6053, 6054, 6055, 6056, 6057,
                               6058, 6059, 6060, 6061, 6062, 6063, 6064, 6065, 6066, 6067, 6068, 6069, 6070, 6071,
                               6072, 6073, 6074, 6075, 6076, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6084, 6085,
                               6086, 6087, 6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098, 6099,
                               6100, 6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108, 6109, 6110, 6111, 6112, 6113,
                               6114, 6115, 6116, 6117, 6118, 6119]
                 })

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
            self.assertDictEqual(
                meta_data,
                {'ion_name': 'POT', 'num_ion': 160, 'time_step': 2.0,
                 "ion_index": [5960, 5961, 5962, 5963, 5964, 5965, 5966, 5967, 5968, 5969, 5970, 5971, 5972, 5973,
                               5974, 5975, 5976, 5977, 5978, 5979, 5980, 5981, 5982, 5983, 5984, 5985, 5986, 5987,
                               5988, 5989, 5990, 5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999, 6000, 6001,
                               6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010, 6011, 6012, 6013, 6014, 6015,
                               6016, 6017, 6018, 6019, 6020, 6021, 6022, 6023, 6024, 6025, 6026, 6027, 6028, 6029,
                               6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6038, 6039, 6040, 6041, 6042, 6043,
                               6044, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6052, 6053, 6054, 6055, 6056, 6057,
                               6058, 6059, 6060, 6061, 6062, 6063, 6064, 6065, 6066, 6067, 6068, 6069, 6070, 6071,
                               6072, 6073, 6074, 6075, 6076, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6084, 6085,
                               6086, 6087, 6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098, 6099,
                               6100, 6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108, 6109, 6110, 6111, 6112, 6113,
                               6114, 6115, 6116, 6117, 6118, 6119]
                 })

    def test_read_k_cylinder_list(self):
        print("# TESTING read_k_cylinder_list, read a sequence of k_cylinder.log files")
        f_list = ["04-output_wrapper/C_0.75_2ps/05-2ps/00/analysis/04-state-code/k_cylinder.log",
                  "04-output_wrapper/C_0.75_2ps/05-2ps/01/analysis/04-state-code/k_cylinder.log"
                  ]
        for get_occu in [False, True]:
            s_list, meta_data, K_occupency, W_occupency = read_k_cylinder_list(f_list, get_occu=get_occu)
            self.assertListEqual(s_list[:3], ["WK0KKW", "WK0KKW", "WK0KKW"])
            self.assertEqual(len(s_list), 1001)
            self.assertDictEqual(
                meta_data,
                {'ion_name': 'POT', 'num_ion': 160, 'time_step': 2.0,
                 "ion_index":[5960,5961,5962,5963,5964,5965,5966,5967,5968,5969,5970,5971,5972,5973,
                              5974,5975,5976,5977,5978,5979,5980,5981,5982,5983,5984,5985,5986,5987,
                              5988,5989,5990,5991,5992,5993,5994,5995,5996,5997,5998,5999,6000,6001,
                              6002,6003,6004,6005,6006,6007,6008,6009,6010,6011,6012,6013,6014,6015,
                              6016,6017,6018,6019,6020,6021,6022,6023,6024,6025,6026,6027,6028,6029,
                              6030,6031,6032,6033,6034,6035,6036,6037,6038,6039,6040,6041,6042,6043,
                              6044,6045,6046,6047,6048,6049,6050,6051,6052,6053,6054,6055,6056,6057,
                              6058,6059,6060,6061,6062,6063,6064,6065,6066,6067,6068,6069,6070,6071,
                              6072,6073,6074,6075,6076,6077,6078,6079,6080,6081,6082,6083,6084,6085,
                              6086,6087,6088,6089,6090,6091,6092,6093,6094,6095,6096,6097,6098,6099,
                              6100,6101,6102,6103,6104,6105,6106,6107,6108,6109,6110,6111,6112,6113,
                              6114,6115,6116,6117,6118,6119]})
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

    def test_read_k_cylinder_broken_line(self):
        print("# TESTING read_k_cylinder, broken line")
        for i in range(6):
            s_list, meta_data, K_occupency, W_occupency = read_k_cylinder(f"01-NaK2K/1-Charmm/broken_last_frame/k_cylinder_{i}.out")
            self.assertListEqual(s_list, ["WKKKKW", "KK0KKW", "KK0KKW", "WKKKKW", "WKKKKW",
                                      "WKKKKW", "KK0KKW", "WKK0KW", "0K0KKW", "WK0KKW"])


    def test_read_k_cylinder_speed(self):
        test_file = Path.home()/"E29Project-2023-04-11/076-MthK/20-C-POT-iso_radius/22-0.78/07_+150_0.2ps/00/analysis/08-02-1.8A/k_cylinder.log"
        if test_file.is_file():
            print("# TESTING read_k_cylinder, speed test... ")
            for name, flag_o, flag_j in (("get_occu=F, get_jump=F", False, False),
                                         ("get_occu=T, get_jump=F",  True, False),
                                         ("get_occu=T, get_jump=T",  True, True),
                                         ):
                t0 = time.time()
                if not flag_j:
                    s_list, meta_data, K_occ, W_occ = read_k_cylinder(test_file, get_occu=flag_o, get_jump=flag_j)
                else:
                    s_list, meta_data, K_occ, W_occ, jump_array = read_k_cylinder(test_file, get_occu=flag_o,  get_jump=flag_j)
                print(f"{name} : {time.time() - t0}")
                self.assertListEqual(s_list[:3], ["WKK0K0", "WKK0K0", "WKK0KW"])
                self.assertListEqual(s_list[-9:], ["WKK0KW", "WKK0K0", "WKK0K0", "WKK0K0", "WKK0K0",
                                                   "WKK0K0", "WKK0K0", "WKK0K0", "WKK0K0",])
                self.assertEqual(len(s_list), 2500001)
                if flag_o:
                    self.assertListEqual(K_occ[:2, :].tolist(),
                                         [[0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0]])
                    self.assertListEqual(W_occ[:3, :].tolist(),
                                         [[2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 1]])
                    self.assertEqual(len(K_occ), 2500001)
                    self.assertEqual(len(W_occ), 2500001)
                if flag_j:
                    self.assertListEqual(jump_array[:52].tolist(),
                                         [0] * 35 + [-1, +1] + [0]*10 + [-1] + [0]*3 + [1])
                    self.assertEqual(len(jump_array), 2500001)
        else :
            print("# TESTING read_k_cylinder speed test cannot be done, input file not found")

    def test_read_k_cylinder_get_jump(self):
        s_list, meta_data, K_occ, W_occ, jump_array = read_k_cylinder("01-NaK2K/1-Charmm/with_water/k_cylinder.out",
                                                                      get_occu=True, get_jump=True)
        self.assertListEqual(s_list[:3], ["WKKKKW", "KK0KKW", "KK0KKW"])
        self.assertListEqual(K_occ[:2, :].tolist(),
                             [[0, 1, 1, 1, 1, 0], [1, 1, 0, 1, 1, 0]])
        self.assertListEqual(W_occ[:2, :].tolist(),
                             [[1, 0, 0, 0, 0, 10], [0, 0, 0, 0, 0, 11]])
        self.assertListEqual(jump_array[:9].tolist(),
                             [0, 2, 0, -2, 0, 0, 2, 2, -1])

if __name__ == '__main__':
    unittest.main()
