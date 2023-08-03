#!/usr/bin/env python3
import MDAnalysis as mda
import warnings
import numpy as np


def detect_SF(u, SF_seq1, SF_seq2=None):
    """
    :param u: MDAnalysis.Universe
    :param SF_seq1: A list of 5 string, each string is a 3 letter amino acid name.
        such as ['THR', 'VAL', 'GLY', 'TYR', 'GLY']
    :param SF_seq2: In case of TRAAK/TREK2 channel, provide the sequence of the other half of SF
    :return: (S00, S01, S12, S23, S34, S45), each element is a MDAnalysis selection
    """
    if SF_seq2 is None:
        SF_seq2 = []
    if len(SF_seq1) != 5:
        raise ValueError("Length of SF sequence should be 5")
    elif len(SF_seq2) != 5 and len(SF_seq2) != 0:
        raise ValueError("Length of SF_seq2 should be 5 or 0")
    seq_length = 5
    S00 = mda.core.groups.AtomGroup([], u)  # empty selection. We will append more atoms on the fly
    S01 = mda.core.groups.AtomGroup([], u)
    S12 = mda.core.groups.AtomGroup([], u)
    S23 = mda.core.groups.AtomGroup([], u)
    S34 = mda.core.groups.AtomGroup([], u)
    S45 = mda.core.groups.AtomGroup([], u)
    for i in range(len(u.residues) - seq_length + 1):
        residue_sequence = [residue.resname for residue in u.residues[i:i + seq_length]]
        if residue_sequence == SF_seq1:
            S00 += u.residues[i + 4].atoms.select_atoms("name O")
            S01 += u.residues[i + 3].atoms.select_atoms("name O")
            S12 += u.residues[i + 2].atoms.select_atoms("name O")
            S23 += u.residues[i + 1].atoms.select_atoms("name O")
            S34 += u.residues[i + 0].atoms.select_atoms("name O")
            S45 += u.residues[i + 0].atoms.select_atoms("name OG1")
        if len(SF_seq2) == 5 and residue_sequence == SF_seq2:
            S00 += u.residues[i + 4].atoms.select_atoms("name O")
            S01 += u.residues[i + 3].atoms.select_atoms("name O")
            S12 += u.residues[i + 2].atoms.select_atoms("name O")
            S23 += u.residues[i + 1].atoms.select_atoms("name O")
            S34 += u.residues[i + 0].atoms.select_atoms("name O")
            S45 += u.residues[i + 0].atoms.select_atoms("name OG1")
    for name, s in zip(("S00", "S01", "S12", "S23", "S34", "S45"),
                       (S00, S01, S12, S23, S34, S45)):
        if len(s) != 4:
            print(name)
            print(s)
            raise ValueError("Number of O found for SF is not 4. Please check the input structure/sequence")
    return S00, S01, S12, S23, S34, S45


class sfilter:
    def __init__(self, u):
        """
        :type u: object MDAnalysis.Universe
        """
        self.u = u
        self.sf_oxygen = ()

    def detect_SF_sequence(self, SF_seq1, SF_seq2=None):
        """
        :param SF_seq1: A list of 5 string, each string is a 3 letter amino acid name.
            such as ['THR', 'VAL', 'GLY', 'TYR', 'GLY']
        :param SF_seq2: In case of TRAAK/TREK2 channel, provide the sequence of the other half of SF
        :return: (S00, S01, S12, S23, S34, S45), each element is a MDAnalysis selection
        """
        self.sf_oxygen = detect_SF(self.u, SF_seq1, SF_seq2)
        return self.sf_oxygen

    def set_sf_oxygen(self, s00: mda.Universe, s01, s12, s23, s34, s45):
        """
        :param s00: MDAnalysis Universe (atoms selection)
        :param s01:
        :param s12:
        :param s23:
        :param s34:
        :param s45:
        :return: None
        """
        for name, sf_oxygen in ("S00", s00), ("S01", s01), ("S12", s12), ("S23", s23), ("S34", s34), ("S45", s45):
            if len(sf_oxygen) != 4:
                warnings.warn("Number if atoms in " + name + "is not 4.", UserWarning)
        self.sf_oxygen = (s00, s01, s12, s23, s34, s45)

    def state_detect(self, K, s5_z_cutoff=4, s5_r_cutoff=8, r_cutoff=2.5, s0_r_cutoff=4, ):
        """
        :param K: MDAnalysis Universe (atoms selection)
        :param s5_z_cutoff: S5 z cutoff from THR oxygen
        :param s5_r_cutoff: S5 radius cutoff
        :param r_cutoff: radius cutoff for inside the selectivity filter
        :param s0_r_cutoff:
        :return: np.array( dtype=np.uint8), what state is each K atom in
             7
          ---------
          |   0   |   7
        --------------
            | 1 |
        8   | 2 |  8
            | 3 |
            | 4 |
        --------------
            | 5 |
           ------
                   6
        """
        s00, s01, s12, s23, s34, s45 = self.sf_oxygen
        z_00 = np.mean(s00.positions[:, 2])
        z_01 = np.mean(s01.positions[:, 2])
        z_12 = np.mean(s12.positions[:, 2])
        z_23 = np.mean(s23.positions[:, 2])
        z_34 = np.mean(s34.positions[:, 2])
        z_45 = np.mean(s45.positions[:, 2])
        xy = (s00 + s01 + s12 + s23 + s34 + s45).positions[:, :2]
        xy = np.mean(xy, axis=0)  # average across atoms

        z_K = K.positions[:, 2]
        x_K = K.positions[:, 0]
        y_K = K.positions[:, 1]
        z_state = np.ones(z_K.shape, dtype=np.uint8) * 7
        z_state[z_K < z_00] = 0
        z_state[z_K < z_01] = 1
        z_state[z_K < z_12] = 2
        z_state[z_K < z_23] = 3
        z_state[z_K < z_34] = 4
        z_state[z_K < z_45] = 5
        z_state[z_K < z_45 - s5_z_cutoff] = 6
        # print(xy.shape)
        r_2 = (x_K - xy[0]) ** 2 + (y_K - xy[1]) ** 2
        # mask = np.logical_and((z_state < 5), (z_state > 0))
        mask = (z_state < 5) & (z_state > 0) & (r_2 > r_cutoff ** 2)
        z_state[mask] = 8  # incase something is inside the membrance

        mask = (z_state == 0) & (r_2 > s0_r_cutoff ** 2)
        z_state[mask] = 7  # xy outside S0 would be assigned as 7

        mask = (z_state == 5) & (r_2 > s5_r_cutoff ** 2)
        z_state[mask] = 6  # xy outside S5 would be assigned as 6
        return z_state


    def state_2_string(self, state_list,  method="K_priority"):
        """
        :param state_list: A list of lists containing the index of what atoms in each binding site.
            such as :
            [[S0_K_index, S1_K_index, S2_K_index, S3_K_index, S4_K_index, S5_K_index],
             [S0_W_index, S1_W_index, S2_W_index, S3_W_index, S4_W_index, S5_W_index]]
        :param method:
            "K_priority",
                K : There is 1 K+ in this binding site, maybe there is water.
                W : There is at least one water molecule in the binding site and there is no K.
                0 : There is neither K nor water in this binging site. (number 0, not letter O)
            "Co-occupy"
                K : There is 1 K+ in this binding site, and there is no water.
                W : There is at least one water molecule in the binding site and there is no K.
                C : There is 1 K+ in this binding site, and there is at least 1 water.
                0 : There is neither K+ nor water in this binging site. (number 0, not letter O)
            "Everything"
                Binding sites are seperated by ',' .
                Elements are seperated by space.
        :return:
        """
        state_str = ""
        if method == "K_priority":
            for k_index, w_index, site in zip(state_list[0], state_list[1], ["0", "1", "2", "3", "4", "5"]):
                sumK = len(k_index)
                sumW = len(w_index)
                if sumK >= 1:
                    state_str += "K"
                elif sumW >= 1:
                    state_str += "W"
                else:
                    state_str += "0"
        elif method == "Co-occupy":
            for k_index, w_index, site in zip(state_list[0], state_list[1], ["0", "1", "2", "3", "4", "5"]):
                sumK = len(k_index)
                sumW = len(w_index)
                if sumK >= 1:
                    if sumK >= 2:
                        warnings.warn("Number of atom in site " + site + " is more than 1. "+str(sumK))
                    if sumW >= 1:
                        state_str += "C"
                    else:
                        state_str += "K"
                elif sumW >= 1:
                    state_str += "W"
                else:
                    state_str += "0"
        elif method == "Everything":
            key_list = list(state_list.keys())
            for site in range(6):
                for atom in key_list:
                    if len(state_list[atom][site]) >= 1:
                        state_str += atom
                        state_str += " "
                state_str += ", "
        else:
            raise NotImplementedError(method+" has not been implemented")

        return state_str


    def state_2_list(self, atom_state, atom_selection):
        """
        :param atom_state: A np.array with the state of every atom. shape should be (number_of_atom,)
        :param atom_selection: MDAnalysis atom selection. Atoms should match the state in atom_state
        :return: list of np.array. Each array contains the indices corresponding to atoms located at each binding site.
            [S0_atom_index,
             S1_atom_index,
             S2_atom_index,
             S3_atom_index,
             S4_atom_index,
             S5_atom_index]  # 0 base index
        """
        state_list = []
        for i in range(6):  # binding site 0 ~ 6
            state_list.append(atom_selection.ix[(atom_state == i)])
        return state_list

    def print_state_list(self, state_list):
        for s, ind in zip(("S0", "S1", "S2", "S3", "S4", "S5", ), state_list):
            print(s, ind)