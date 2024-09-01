import unittest
from pathlib import Path
from Sfilter import kinetics


class MyTestCase(unittest.TestCase):
    def test_init(self):
        print("#TESTING: init. Init the Sf_model class in the normal way.")
        base = Path("04-output_wrapper/C_0.75_2ps/05-2ps")
        file_list = [base / f"{i:02}/analysis/04-state-code/k_cylinder.log" for i in range(2)]
        k_model = kinetics.Sf_model(file_list)
        self.assertTrue(isinstance(k_model.passage_cycle_correct, kinetics.Passage_cycle_correct))
        self.assertEqual(len(k_model.passage_cycle_correct.passage_time_length_alltraj_raw), 2)
        self.assertEqual(len(k_model.passage_cycle_correct.passage_time_point_alltraj_raw),  2)
        self.assertEqual(len(k_model.passage_cycle_correct.jump_array_alltraj),              2)

        res_dict = k_model.passage_cycle_correct.get_passage_ij(0, 1)
        #        WKK0KW | KK0KKW
        # | rep | start | end | p_jump |
        # | 0   | 3     | 155 | -2     |
        # | 0   | 204   | 275 | -2     |
        # | 1   | 103   | 107 | -2     |
        # | 1   | 158   | 169 | -2     |
        # | 1   | 172   | 241 | +5     |
        answer_dict = {
            -2: (
                [[152,  71], [4,    11]],
                [[3,   204], [103, 158]],
                [[155, 275], [107, 169]]
                ),
            5: (
                [[], [69]],
                [[], [172]],
                [[], [241]]
                )
            }
        self.assertDictEqual(res_dict, answer_dict)
        res_dict = k_model.passage_cycle_correct.get_passage_ij(2, 0)
        #        WK0KKW | WKK0KW
        # | rep | start | end | p_jump |
        # | 0   | 0     | 3   | 1      |
        # | 0   | 6     | 7   | 1      |
        #                             jump  property  rep
        self.assertListEqual(res_dict[   1][       0][  0][0:2], [3, 1]) #
        self.assertListEqual(res_dict[   1][       1][  0][0:2], [0, 6]) # start
        self.assertListEqual(res_dict[   1][       2][  0][0:2], [3, 7]) # end




if __name__ == '__main__':
    unittest.main()
