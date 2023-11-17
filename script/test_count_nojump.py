import unittest


import count_nojump
import os
from pathlib import Path
import subprocess
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations




basedir = Path(__file__).parent
class MyTestCase(unittest.TestCase):
    def setUp(self):
        os.chdir(basedir)

    def test_Perm_Event_Pool(self):
        print("TEST: class Perm_Event_Pool")
        p_event = count_nojump.Perm_Event_Pool(np.array([0, 1]), np.array([1, 1]))
        sequence = np.array([[2, 1],
                             [3, 3],
                             [4, 5],
                             [5, 3],
                             [6, 5],
                             [7, 4],
                             [8, 3],])
        for s in sequence:
            p_event.update(s)
        p_event.update_final_frame(np.array([5, 3]))
        self.assertListEqual(p_event.permeation_event, [[0, 0, 2, 1],  # 0
                                                        [1, 1, 2, 1],  # 1
                                                        [1, 2, 3, 1],  # 2
                                                        [0, 2, 4, 1],  # 3
                                                        [1, 3, 4, -1],  # 4
                                                        [1, 4, 5, 1],  # 5
                                                        [0, 4, 6, 1],
                                                        [1, 5, 7, -1],
                                                        [0, 6, 7, -1]
                                                        ])

    def test_set_up_boundary_mem_layer(self):
        print("TEST: set_up_boundary_mem_layer, give selection str for upper and lower layer")
        pdb = "../test/01-NaK2K/1-Charmm/em.pdb"
        u = mda.Universe(pdb)
        mem_layer = {
        "SF":["resid 67 and resname GLY and name O",
              "resid 63 and resname THR and name OG1"]
        }
        boundary_dict = count_nojump.set_up_boundary_mem_layer(u, mem_layer)
        sele_upper, sele_lower = boundary_dict["SF"]
        self.assertListEqual(sele_upper.ix.tolist(), [753, 2243, 3733, 5223,])
        self.assertListEqual(sele_lower.ix.tolist(), [695, 2185, 3675, 5165])

    def test_set_up_boundary_mem_str(self):
        print("TEST: set_up_boundary_mem_str, give selection str and automatically find the upper and lower layer, SF")
        pdb = "../test/01-NaK2K/1-Charmm/em.pdb"
        u = mda.Universe(pdb)
        mem_str = "(resid 67 and resname GLY and name O) or (resid 63 and resname THR and name OG1)"
        sele_upper, sele_lower = count_nojump.set_up_boundary_mem_str(u, mem_str)
        self.assertListEqual(sele_upper.ix.tolist(), [753, 2243, 3733, 5223, ])
        self.assertListEqual(sele_lower.ix.tolist(), [695, 2185, 3675, 5165])

    def test_set_up_boundary_mem_str_nameP(self):
        print("TEST: set_up_boundary_mem_str, give selection str and automatically find the upper and lower layer, name P")
        pdb = "../test/01-NaK2K/1-Charmm/em.pdb"
        u = mda.Universe(pdb)
        mem_str = "name P"
        sele_upper, sele_lower = count_nojump.set_up_boundary_mem_str(u, mem_str)
        self.assertEqual(len(sele_upper.ix), 83)
        self.assertEqual(len(sele_lower.ix), 85)
        self.assertTrue(max(sele_upper.ix) < min(sele_lower.ix))


    def test_div_mod_coordinate_z(self):
        print("TEST: div_mod_coordinate_z, give atom_z, decide which compartment it belongs to")
        atom_z = np.array([-10, -5, -1, 1.1, 2.5, 3.0,  8.5, 9, 15])
        upper = 2.0
        lower = -2.0
        box_z = 10
        state_array = count_nojump.div_mod_coordinate_z(atom_z, upper, lower, box_z)
        self.assertListEqual(state_array.tolist(), [-2, -1, 0, 0, 1, 1, 2, 2, 3])

    def test_calc_state_array(self):
        print("TEST: calc_state_array, give atom selection, decide which compartment it belongs to")
        u = mda.Universe("../test/01-NaK2K/1-Charmm/em.pdb")
        permeable_selection = u.select_atoms("name POT")
        upper_selection = u.select_atoms("resid 67 and resname GLY and name O")
        lower_selection = u.select_atoms("resid 63 and resname THR and name OG1")
        box_z = u.dimensions[2]
        state_array = count_nojump.calc_state_array(permeable_selection, upper_selection, lower_selection)
        self.assertListEqual(state_array.tolist()[:8], [0, 0, 0, 0,
                                                    1, -1, 1, -1
                                                    ])





    # regression test start from here
    def test_unit1(self):
        print("TEST: unit test 1")
        os.chdir(basedir/"../test/01-NaK2K/1-Charmm/dry/count_nojump")
        command = "../../../../../script/count_nojump.py -i in.json"
        results = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res_text = results.stdout.decode('utf-8')
        res_err = results.stderr.decode('utf-8')
        print(res_text)
        # print(res_err)


if __name__ == '__main__':
    unittest.main()
