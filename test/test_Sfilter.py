import unittest
import Sfilter
import MDAnalysis as mda
import time
import warnings


class MyTestCase(unittest.TestCase):
    def test_detect_SF_sequence_NaK2K_charmm(self):
        print("# TEST detect_SF_sequence NaK2K Charmm")
        for file in ["01-NaK2K/1-Charmm/em.gro",
                     "01-NaK2K/1-Charmm/em_K.gro",
                     "01-NaK2K/1-Charmm/dry/em_dry.gro"
                     ]:
            u = mda.Universe(file)
            sf = Sfilter.sfilter(u)
            sf.detect_SF_sequence(['THR', 'VAL', 'GLY', 'TYR', 'GLY'])
            s00, s01, s12, s23, s34, s45 = sf.sf_oxygen
            self.assertListEqual(s00.ix.tolist(), [int(i) - 1 for i in "754 2244 3734 5224".split()])
            self.assertListEqual(s01.ix.tolist(), [int(i) - 1 for i in "747 2237 3727 5217".split()])
            self.assertListEqual(s12.ix.tolist(), [int(i) - 1 for i in "726 2216 3706 5196".split()])
            self.assertListEqual(s23.ix.tolist(), [int(i) - 1 for i in "719 2209 3699 5189".split()])
            self.assertListEqual(s34.ix.tolist(), [int(i) - 1 for i in "703 2193 3683 5173".split()])
            self.assertListEqual(s45.ix.tolist(), [int(i) - 1 for i in "696 2186 3676 5166".split()])

    def test_detect_SF_sequence_NaK2K_amber(self):
        print("# TEST detect_SF_sequence NaK2K Amber")
        u = mda.Universe("01-NaK2K/2-Amber/em.pdb")
        sf = Sfilter.sfilter(u)
        sf.detect_SF_sequence(['THR', 'VAL', 'GLY', 'TYR', 'GLY'])
        s00, s01, s12, s23, s34, s45 = sf.sf_oxygen
        self.assertListEqual(s00.ix.tolist(), [int(i) - 1 for i in "754 2244 3734 5224".split()])
        self.assertListEqual(s01.ix.tolist(), [int(i) - 1 for i in "747 2237 3727 5217".split()])
        self.assertListEqual(s12.ix.tolist(), [int(i) - 1 for i in "726 2216 3706 5196".split()])
        self.assertListEqual(s23.ix.tolist(), [int(i) - 1 for i in "719 2209 3699 5189".split()])
        self.assertListEqual(s34.ix.tolist(), [int(i) - 1 for i in "703 2193 3683 5173".split()])
        self.assertListEqual(s45.ix.tolist(), [int(i) - 1 for i in "700 2190 3680 5170".split()])

    def test_detect_SF_sequence_TRAAK_charmm(self):
        print("# TEST detect_SF_sequence TRAAK Charmm")
        u = mda.Universe("02-TRAAK/1-Charmm/em.pdb")
        sf = Sfilter.sfilter(u)
        sf.detect_SF_sequence(['THR', 'ILE', 'GLY', 'TYR', 'GLY'], ['THR', 'VAL', 'GLY', 'PHE', 'GLY'])
        s00, s01, s12, s23, s34, s45 = sf.sf_oxygen
        self.assertListEqual(s00.ix.tolist(), [int(i) - 1 for i in "1609 3342 5645 7378".split()])
        self.assertListEqual(s01.ix.tolist(), [int(i) - 1 for i in "1602 3335 5638 7371".split()])
        self.assertListEqual(s12.ix.tolist(), [int(i) - 1 for i in "1581 3315 5617 7351".split()])
        self.assertListEqual(s23.ix.tolist(), [int(i) - 1 for i in "1574 3308 5610 7344".split()])
        self.assertListEqual(s34.ix.tolist(), [int(i) - 1 for i in "1555 3292 5591 7328".split()])
        self.assertListEqual(s45.ix.tolist(), [int(i) - 1 for i in "1548 3285 5584 7321".split()])

    def test_state_detect(self):
        print("# TEST state_detect")
        u = mda.Universe("01-NaK2K/1-Charmm/em.pdb", "01-NaK2K/1-Charmm/with_water/fix_atom_c_100ps.xtc")
        sf = Sfilter.sfilter(u)
        sf.detect_SF_sequence(['THR', 'VAL', 'GLY', 'TYR', 'GLY'])
        K = sf.u.select_atoms('name POT or name K')
        K_state_list = []
        tick = time.time()
        for ts in u.trajectory:
            k_state = sf.state_detect(K)
            if sum(k_state == 8) != 0:
                warnings.warn("There are K+ atom found in the membrane. Number :", sum(k_state == 8))
            K_state_list.append(sf.state_2_list(k_state, K))
        tock = time.time()
        print("    10 frames take ", tock - tick)
        # frame 0
        self.assertListEqual(K_state_list[0][0].tolist(), [])
        self.assertListEqual(K_state_list[0][1].tolist(), [5963])
        self.assertListEqual(K_state_list[0][2].tolist(), [5962])
        self.assertListEqual(K_state_list[0][3].tolist(), [5961])
        self.assertListEqual(K_state_list[0][4].tolist(), [5960])
        self.assertListEqual(K_state_list[0][5].tolist(), [])
        # frame 1
        self.assertListEqual(K_state_list[1][0].tolist(), [5963])
        self.assertListEqual(K_state_list[1][1].tolist(), [5962])
        self.assertListEqual(K_state_list[1][2].tolist(), [])
        self.assertListEqual(K_state_list[1][3].tolist(), [5961])
        self.assertListEqual(K_state_list[1][4].tolist(), [5960])
        self.assertListEqual(K_state_list[1][5].tolist(), [])

    def test_state_2_string(self):
        print("# TEST state_2_string")
        u = mda.Universe("01-NaK2K/1-Charmm/em.pdb", "01-NaK2K/1-Charmm/with_water/fix_atom_c_100ps.xtc")
        sf = Sfilter.sfilter(u)
        sf.detect_SF_sequence(['THR', 'VAL', 'GLY', 'TYR', 'GLY'])
        K = sf.u.select_atoms('name POT or name K')
        O = sf.u.select_atoms('resname SOL and name OW')

        state_str_k_list = []
        state_str_c_list = []
        tick = time.time()
        for ts in u.trajectory:
            o_state = sf.state_2_list(sf.state_detect(O), O)
            k_state = sf.state_2_list(sf.state_detect(K), K)
            state_string = sf.state_2_string([k_state, o_state], method="K_priority")
            state_str_k_list.append(state_string)
            state_string = sf.state_2_string([k_state, o_state], method="Co-occupy")
            state_str_c_list.append(state_string)
            #print(sf.state_2_string({"K": k_state, "W": o_state}, method="Everything"))
        tock = time.time()
        print("    10 frames take ", tock - tick)
        answer = ["KKKK", "K0KK", "K0KK", "KKKK", "KKKK",
                  "KKKK", "K0KK", "KK0K", "K0KK", "K0KK",
                  "0KKK"]  # xtck answer
        for state, xtck in zip(state_str_k_list, answer):
            self.assertEqual(state[1:5], xtck)
        for state, xtck in zip(state_str_c_list, answer):
            self.assertEqual(state[1:5], xtck)


if __name__ == '__main__':
    unittest.main()
