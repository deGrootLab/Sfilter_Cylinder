import unittest
import sys
from count_cylinder import PermeationEvent, prepare_non_water_SF
from count_cylinder import get_closest_water
from count_cylinder import prepare_non_water
from Sfilter import sfilter
import MDAnalysis as mda
import numpy as np
import subprocess
import os


class MyTestCase(unittest.TestCase):
    # unit test, test each function/class
    def test_up(self):
        print("\n# Test PermeationEvent with output writing")
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
        os.remove("file")

    def test_down(self):
        print("\n# Test PermeationEvent with down permeation event")
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

    def test_get_closest_water(self):
        print("\n# Test if we can iteratively find the closest water")
        u = mda.Universe("../test/01-NaK2K/1-Charmm/em.pdb",
                         "../test/01-NaK2K/1-Charmm/with_water/fix_atom_c_100ps.xtc")
        sf = sfilter(u)
        sf.detect_SF_sequence(SF_seq1="THR VAL GLY TYR GLY".split())
        s45 = sf.sf_oxygen[-1]
        water_O_selection = u.select_atoms('resname SOL and name OW')
        distance_array = np.zeros((1, water_O_selection.n_atoms))
        waters_0 = []
        for ts in u.trajectory:
            waters = get_closest_water(s45[[0]], water_O_selection, 2, distance_array)
            # print(waters.ix)  # only 0 and 3 are oxygen atoms
            waters_0.append(waters.ix[[0, 3]])
            self.assertTrue(np.all((waters.ix[[1, 4]] - waters.ix[[0, 3]]) == 1))
            self.assertTrue(np.all((waters.ix[[2, 5]] - waters.ix[[0, 3]]) == 2))
        waters_1 = []
        for ts in u.trajectory:
            waters = get_closest_water(s45, water_O_selection, 2, distance_array)
            waters_1.append(waters.ix[[0, 3]])
            self.assertTrue(np.all((waters.ix[[1, 4]] - waters.ix[[0, 3]]) == 1))
            self.assertTrue(np.all((waters.ix[[2, 5]] - waters.ix[[0, 3]]) == 2))

        self.assertListEqual(waters_0[0].tolist(), [40145, 44588])
        self.assertListEqual(waters_0[1].tolist(), [36476, 61961])
        self.assertListEqual(waters_0[2].tolist(), [49097, 55148])

        self.assertTrue(40145 in waters_1[0])
        self.assertTrue(61961 in waters_1[1])
        self.assertTrue(51449 in waters_1[2])
        self.assertTrue(43874 in waters_1[3] and 58355 in waters_1[3])
        self.assertTrue(49463 in waters_1[4])

    def test_prepare_non_water_SF(self):
        print("\n# Test if we can prepare the non-water (SF) atoms (using TRAAK)")
        pdb = "../test/02-TRAAK/1-Charmm/em.pdb"
        u = mda.Universe(pdb)
        sf = sfilter(u)
        sf.detect_SF_sequence("THR ILE GLY TYR GLY".split(), "THR VAL GLY PHE GLY".split())
        non_wat = prepare_non_water_SF(sf, ["POT"])
        SF_index = (list(range(1541, 1609))  # SF1 68
                    + list(range(3278, 3342))  # SF2 64
                    + list(range(5577, 5645))  # SF1 68
                    + list(range(7314, 7378))  # SF2 64
                    + list(range(8072, 8475))  # POT
                    )  # the index should be exactly as this in the correct order
        self.assertListEqual(non_wat.ix.tolist(), SF_index)

    def test_prepare_non_water(self):
        print("\n# Test if we can prepare the non-water (all non water) atoms (using TRAAK)")
        pdb = "../test/02-TRAAK/1-Charmm/em.pdb"
        u = mda.Universe(pdb)
        sf = sfilter(u)
        sf.detect_SF_sequence("THR ILE GLY TYR GLY".split(), "THR VAL GLY PHE GLY".split())
        non_wat = prepare_non_water(sf, ["POT"], non_wat="nWat")
        all_index = list(range(0, 41828))  # the index should be exactly as this in the correct order
        self.assertListEqual(non_wat.ix.tolist(), all_index)

    def test_prepare_non_water_2(self):
        print("\n# Test if we can prepare the non-water (all non water) atoms (using NaK2K)")
        pdb = "../test/01-NaK2K/1-Charmm/em.pdb"
        u = mda.Universe(pdb)
        sf = sfilter(u)
        sf.detect_SF_sequence("THR VAL GLY TYR GLY".split())
        non_wat = prepare_non_water(sf, ["POT"], non_wat="nWat")
        all_index = list(range(0, 28784))
        self.assertListEqual(non_wat.ix.tolist(), all_index)


    # test the whole command line tool, regression test
    def test_count_cylinder_01(self):
        print("\n# Test count_cylinder.py and see if we can get the consistent 6 letter code")
        args = "-pdb ../test/01-NaK2K/1-Charmm/em.pdb -xtc ../test/01-NaK2K/1-Charmm/with_water/fix_atom_c_100ps.xtc " \
               "-K POT -SF_seq THR VAL GLY TYR GLY "
        xtc_out = "../test/01-NaK2K/1-Charmm/with_water/fix_10water.xtc"
        commands = ["count_cylinder.py " + args,
                    "count_cylinder.py " + args + " -n_water 10 -reduced_xtc " + xtc_out + " -non_wat nWat",
                    "count_cylinder.py " + args + " -n_water 10 -reduced_xtc " + xtc_out + " -non_wat SF",
                    ]
        for command in commands:
            if os.path.isfile(xtc_out):
                os.remove(xtc_out)
            results = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res_text = results.stdout.decode('utf-8').split("\n")
            res_err = results.stderr.decode('utf-8').split("\n")
            print(res_err)
            letter_codes = []
            POT_index = []
            for line in res_text:
                if "# S6l" in line:
                    letter_codes.append(line.split()[-1])
                if " POT :" in line:
                    POT_line = line.split(",")[0]
                    POT_words = POT_line.split(":")[1].split()
                    if POT_words != []:
                        index = int(POT_words[0])
                        POT_index.append(index)
            self.assertListEqual(letter_codes, ["WKKKKW", "KK0KKW", "KK0KKW", "WKKKKW", "WKKKKW",
                                                "WKKKKW", "KK0KKW", "WKK0KW", "0K0KKW", "WK0KKW",
                                                "K0KKKW", ])
            self.assertListEqual(POT_index[:12], [5963, 5962, 5961, 5960, 5963, 5962, 5961, 5960, 5963, 5962, 5961, 5960])
            #  remove the output file
            os.remove("POT_perm_event.out")
            os.remove("Wat_perm_event.out")
        os.remove(xtc_out)
        os.remove(xtc_out[:-4] + "_SF.pdb")
        os.remove(xtc_out[:-4] + "_nWat.pdb")

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

    def test_count_cylinder_03(self):
        print("\n# Test count_cylinder.py and see if we can rerun the script on a reduced water output")
        xtc_out1 = "../test/01-NaK2K/1-Charmm/with_water/fix_100water.xtc"
        args1 = "-pdb ../test/01-NaK2K/1-Charmm/em.pdb " \
                "-xtc ../test/01-NaK2K/1-Charmm/with_water/fix_atom_c_100ps.xtc " \
                "-K POT -SF_seq THR VAL GLY TYR GLY -n_water 100 " \
                "-reduced_xtc " + xtc_out1 + " -non_wat nWat"
        xtc_out2 = "../test/01-NaK2K/1-Charmm/with_water/fix_25water_SF.xtc"
        args2 = "-pdb ../test/01-NaK2K/1-Charmm/with_water/fix_100water_nWat.pdb " \
                "-xtc ../test/01-NaK2K/1-Charmm/with_water/fix_100water.xtc " \
                "-K POT -SF_seq THR VAL GLY TYR GLY -n_water 25 " \
                "-reduced_xtc " + xtc_out2 + " -non_wat SF"
        commands = ["count_cylinder.py " + args1,
                    "count_cylinder.py " + args2,
                    ]
        for command, xtc in zip(commands, [xtc_out1, xtc_out2]):
            print(command)
            if os.path.isfile(xtc):
                os.remove(xtc)
            results = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res_text = results.stdout.decode('utf-8').split("\n")
            res_err = results.stderr.decode('utf-8').split("\n")
            print(res_err)
            letter_codes = []
            POT_index = []
            for line in res_text:
                if "# S6l" in line:
                    letter_codes.append(line.split()[-1])
                if " POT :" in line:
                    POT_line = line.split(",")[0]
                    POT_words = POT_line.split(":")[1].split()
                    if POT_words != []:
                        index = int(POT_words[0])
                        POT_index.append(index)
            self.assertListEqual(letter_codes, ["WKKKKW", "KK0KKW", "KK0KKW", "WKKKKW", "WKKKKW",
                                                "WKKKKW", "KK0KKW", "WKK0KW", "0K0KKW", "WK0KKW",
                                                "K0KKKW", ])
            self.assertListEqual(POT_index[:12], [5963, 5962, 5961, 5960, 5963, 5962, 5961, 5960, 5963, 5962, 5961, 5960])
            #  remove the output file
            os.remove("POT_perm_event.out")
            os.remove("Wat_perm_event.out")
        os.remove(xtc_out1)
        os.remove(xtc_out2)

    def test_count_cylinder_04(self):
        print("\n# Test count_cylinder.py. Prepare Water reduced traj on a pdb that has water before the POT")
        xtc_out = "../test/01-NaK2K/1-Charmm/free_energy/Reaction1/19/k_cylinder/07-200wat/fix_100water.xtc"
        args1 = "-pdb ../test/01-NaK2K/1-Charmm/free_energy/Reaction1/19/start_hard.pdb " \
                "-xtc ../test/01-NaK2K/1-Charmm/free_energy/Reaction1/19/fix_c_10frame.xtc " \
                "-K POT -SF_seq THR VAL GLY TYR GLY -n_water 100 " \
                "-cylinderRad 3.5 -S5_z_cutoff 4 -S5_r_cutoff 9 " \
                "-reduced_xtc " + xtc_out + " -non_wat SF"
        args2 = "-pdb ../test/01-NaK2K/1-Charmm/free_energy/Reaction1/19/k_cylinder/07-200wat/fix_100water_SF.pdb " \
                "-xtc ../test/01-NaK2K/1-Charmm/free_energy/Reaction1/19/k_cylinder/07-200wat/fix_100water.xtc " \
                "-K POT -SF_seq THR VAL GLY TYR GLY " \
                "-cylinderRad 3.5 -S5_z_cutoff 4 -S5_r_cutoff 9 "
        if os.path.isfile(xtc_out):
            os.remove(xtc_out)
        commands = ["count_cylinder.py " + args1,
                    "count_cylinder.py " + args2,
                    ]
        for command in commands:
            results = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res_text = results.stdout.decode('utf-8').split("\n")
            res_err = results.stderr.decode('utf-8').split("\n")
            # print(res_err)
            letter_codes = []
            POT_index = []
            for line in res_text:
                if "# S6l" in line:
                    letter_codes.append(line.split()[-1])
                if " POT :" in line:
                    POT_line = line.split(",")[0]
                    POT_words = POT_line.split(":")[1].split()
                    if POT_words != []:
                        index = int(POT_words[0])
                        POT_index.append(index)
            self.assertListEqual(letter_codes, ["WK0K0K", "WK0K0K", "WK0K0K", "WK0K0K", "WK0K0K",
                                                "WK0KWW", "WK0K0K", "WK0KKW", "WK0K0K", "WK0KKW",
                                                "WK0KWK",
                                                ])
        if os.path.isfile(xtc_out):
            os.remove(xtc_out)



if __name__ == '__main__':
    unittest.main()
