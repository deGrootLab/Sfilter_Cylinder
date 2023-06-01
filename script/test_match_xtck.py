import unittest
from match_xtck import *


class MyTestCase(unittest.TestCase):
    def test_read_xtck(self):
        print("# TEST : read xtck perm_up.dat and perm_down.dat")
        up_f = "../test/01-NaK2K/1-Charmm/dry/xtck/perm_up.dat"
        down_f = "../test/01-NaK2K/1-Charmm/dry/xtck/perm_down.dat"
        p_list = read_xtck_perm_up_down(up_f, down_f)
        self.assertEqual(len(p_list), 2)

    def test_read_cylinder_Sfilter(self):
        print("# TEST : read cylinder count")
        file = "../test/01-NaK2K/1-Charmm/dry/POT_perm_event.out"
        p_list = read_cylinder_Sfilter(file)
        self.assertEqual(len(p_list), 0)

        file = "../test/03-longest_common_sequence/03-state-code/POT_perm_event.out"
        p_list = read_cylinder_Sfilter(file)
        self.assertEqual(len(p_list), 94)

    def test_match_xtck_cylinder_sfilter(self):
        print("# TEST : match xtck event with cylinder count event.")
        f_up = "../test/03-longest_common_sequence/01-xtck/perm_up.dat"
        f_down = "../test/03-longest_common_sequence/01-xtck/perm_down.dat"
        xtck_list = read_xtck_perm_up_down(f_up, f_down)
        self.assertEqual(len(xtck_list), 93)

        f_cylinder = "../test/03-longest_common_sequence/03-state-code/POT_perm_event.out"
        cyl_list = read_cylinder_Sfilter(f_cylinder)
        self.assertEqual(len(cyl_list), 94)

        for i in range(6):
            self.assertTrue(match_xtck_cylinder_sfilter(xtck_list[i], cyl_list[i + 1]))
        for i in range(7, 10):
            self.assertFalse(match_xtck_cylinder_sfilter(xtck_list[i], cyl_list[i + 1]))
    def test_time_seq_list(self):
        print("# TEST : class time_seq_list.pop")
        s1 = time_seq_list([1, 2, 3, 4, 5])
        s2 = s1.pop()
        self.assertEqual(len(s1), 5)
        self.assertEqual(len(s2), 4)

    def test_LCS(self):
        print("# TEST : LCS on toy sequence")
        s1 = time_seq_list([1, 2, 3, 4, 5])
        s2 = time_seq_list([1, 2, 3, 5])
        l, s_list = longest_common_subsequence(s1, s2, {}, compare=lambda x, y: x == y)
        self.assertEqual(l, 4)
        self.assertListEqual(s_list, [[1, 1], [2, 2], [3, 3], [4, 'None'], [5, 5]])

    def test_LCS_2(self):
        print("# TEST : LCS on real sequence")
        f_up = "../test/03-longest_common_sequence/01-xtck/perm_up.dat"
        f_down = "../test/03-longest_common_sequence/01-xtck/perm_down.dat"
        xtck_list = read_xtck_perm_up_down(f_up, f_down)
        self.assertEqual(len(xtck_list), 93)

        f_cylinder = "../test/03-longest_common_sequence/03-state-code/POT_perm_event.out"
        cyl_list = read_cylinder_Sfilter(f_cylinder)
        self.assertEqual(len(cyl_list), 94)

        s1 = time_seq_list(xtck_list)
        s2 = time_seq_list(cyl_list)
        l, s_list = longest_common_subsequence(s1, s2, {}, compare=match_xtck_cylinder_sfilter)
        self.assertEqual([i[1] for i in s_list].count("None"), 2)
        self.assertEqual([i[0] for i in s_list].count("None"), 3)


if __name__ == '__main__':
    unittest.main()
