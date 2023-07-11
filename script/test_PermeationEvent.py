import unittest
import sys
from count_cylinder import PermeationEvent
import numpy as np
import subprocess
import os

class MyTestCase(unittest.TestCase):
    def test_up(self):
        seq = [np.array([5, 5]),
               np.array([1, 1]),
               np.array([3, 3]),
               np.array([4, 4]),
               np.array([5, 5]),
               np.array([1, 1]),
               np.array([3, 3]),
               np.array([4, 4]),
               np.array([5, 5]),
               np.array([5, 1]),
               ]
        p = PermeationEvent(np.array([5961, 5962]))
        for s in seq:
            p.update(s)
        p.final_frame_check()
        self.assertListEqual(p.up_1_count[0], [5961, 2, 1])
        self.assertListEqual(p.up_1_count[1], [5962, 2, 1])
        self.assertListEqual(p.up_1_count[2], [5961, 6, 2])
        self.assertListEqual(p.up_1_count[3], [5962, 6, 2])
        self.assertListEqual(p.up_1_count[4], [5962, 9, 2])


        self.assertEqual(len(p.up_1_count), 5)
        p.write_result("file", charge=1, voltage=300, time_step=20)
        with open("file", "r") as f:
            lines = f.readlines()
        self.assertEqual(lines[2], "  5961          ,      2,        40, 20\n")
        self.assertEqual(lines[3], "  5962          ,      2,        40, 20\n")
        self.assertEqual(lines[4], "  5961          ,      6,       120, 40\n")
        self.assertEqual(lines[5], "  5962          ,      6,       120, 40\n")
        self.assertEqual(lines[6], "  5962          ,      9,       180, 40\n")
        self.assertTrue("Permeation events up   : 5\n" in lines)

    def test_down(self):
        seq = [np.array([1, 5]),
               np.array([5, 4]),
               np.array([4, 3]),
               np.array([3, 1]),
               np.array([1, 5]),
               np.array([5, 4]),
               np.array([4, 3]),
               np.array([3, 1]),
               np.array([1, 5]),
               np.array([5, 4]),
               ]
        p = PermeationEvent(np.array([5961, 5962]))
        for s in seq:
            p.update(s)
        p.final_frame_check()
        self.assertListEqual(p.down_1_count[0], [5961, 2, 1])
        self.assertListEqual(p.down_1_count[1], [5962, 5, 2])
        self.assertListEqual(p.down_1_count[2], [5961, 6, 2])
        self.assertListEqual(p.down_1_count[3], [5962, 9, 2])
        self.assertListEqual(p.down_1_count[4], [5961, 9, 2])

        self.assertEqual(len(p.up_1_count), 0)
        self.assertEqual(len(p.down_1_count), 5)

    def test_count_cylinder_01(self):
        print("\n# Test count_cylinder.py and see if we can get the consistent 6 letter code")
        args = "-pdb ../test/01-NaK2K/1-Charmm/em.pdb -xtc ../test/01-NaK2K/1-Charmm/with_water/fix_atom_c_100ps.xtc " \
               "-K POT -SF_seq THR VAL GLY TYR GLY"
        command = "count_cylinder.py " + args
        results = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res_text = results.stdout.decode('utf-8').split("\n")
        letter_codes = []
        for line in res_text:
            if "# S6l" in line:
                letter_codes.append(line.split()[-1])
        self.assertListEqual(letter_codes, ["WKKKKW", "KK0KKW", "KK0KKW", "WKKKKW", "WKKKKW",
                                            "WKKKKW", "KK0KKW", "WKK0KW", "0K0KKW", "WK0KKW",
                                            "K0KKKW", ])
        #  remove the output file
        os.remove("POT_perm_event.out")
        os.remove("Wat_perm_event.out")
    def test_count_cylinder_02(self):
        print("\n# Test count_cylinder.py and see if we can count the permeation event")
        args = "-pdb ../test/01-NaK2K/1-Charmm/dry/em_dry.gro -xtc ../test/01-NaK2K/1-Charmm/dry/fix_atom_c_100ps_dry.xtc " \
               "-K K -SF_seq THR VAL GLY TYR GLY"
        command = "count_cylinder.py " + args
        results = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res_text = results.stdout.decode('utf-8').split("\n")
        letter_codes = []
        for line in res_text:
            if "# S6l" in line:
                letter_codes.append(line.split()[-1])
        self.assertListEqual(letter_codes[:10], ["0KKKK0", "KK0KK0", "KK0KK0", "0KKKK0", "0KKKK0",
                                                 "0KKKK0", "KK0KK0", "0KK0K0", "0K0KK0", "0K0KK0"])
        # read the K_perm_event.out file
        perm_number = 0
        with open("K_perm_event.out", "r") as f:
            lines = f.readlines()
            for l in lines:
                if "Permeation events up" in l:
                    perm_number = int(l.split()[-1])
        self.assertEqual(perm_number, 2)
        os.remove("K_perm_event.out")
        os.remove("Wat_perm_event.out")





if __name__ == '__main__':
    unittest.main()
