import unittest
from Sfilter import Perm_event_output


class MyTestCase(unittest.TestCase):
    def test_Perm_event_output_read_file(self):
        files = ["03-longest_common_sequence/03-state-code/POT_perm_event.out",
                 "04-output_wrapper/POT_event.out"]
        p_list = Perm_event_output(files)
        self.assertListEqual(p_list.perm_list[1]["time"].tolist()[-3:], [2000., 2040., 2100.])
        self.assertListEqual(p_list.perm_list[1]["up"].tolist()[-3:], [True, False, True])
        print(p_list.get_conductance())
        print(p_list.get_conductance(begin=0, end=500000.0, voltage=300))



if __name__ == '__main__':
    unittest.main()
