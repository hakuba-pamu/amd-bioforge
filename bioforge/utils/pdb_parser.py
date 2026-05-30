"""PDB (Protein Data Bank) file parser.

Extracts atom coordinates, bonds, and topology
for force field assignment.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Atom:
    """PDB ATOM/HETATM record."""
    serial: int
    name: str
    alt_loc: str
    residue_name: str
    chain_id: str
    residue_seq: int
    position: np.ndarray  # [3] Angstrom
    occupancy: float
    temp_factor: float
    element: str
    mass: float = 12.011
    charge: float = 0.0
    sigma: float = 3.5
    epsilon: float = 0.1


class PDBParser:
    """Parse PDB format files."""

    ATOMIC_MASSES = {
        "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
        "S": 32.065, "P": 30.974, "F": 18.998, "CL": 35.453,
        "MG": 24.305, "ZN": 65.380, "CA": 40.078, "FE": 55.845,
    }

    def parse(self, pdb_path: Path) -> Tuple[
        List[Atom], List[Tuple[int, int]],
        List[Tuple[int, int, int]], List[Tuple[int, int, int, int]],
        np.ndarray,
    ]:
        """Parse a PDB file.

        Returns:
            atoms, bonds, angles, dihedrals, box_vectors
        """
        atoms = []

        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom = self._parse_atom_line(line)
                    atoms.append(atom)

        if not atoms:
            raise ValueError(f"No atoms found in {pdb_path}")

        print(f"[PDB] Found {len(atoms)} atoms")

        bonds = self._generate_bonds(atoms)
        print(f"[PDB] Generated {len(bonds)} bonds")

        angles = self._generate_angles(bonds)
        print(f"[PDB] Generated {len(angles)} angles")

        dihedrals = self._generate_dihedrals(angles)
        print(f"[PDB] Generated {len(dihedrals)} dihedrals")

        coords = np.array([a.position for a in atoms])
        box_size = np.max(coords.max(axis=0) - coords.min(axis=0)) + 20.0
        box = np.eye(3) * box_size

        return atoms, bonds, angles, dihedrals, box

    def _parse_atom_line(self, line: str) -> Atom:
        """Parse a single ATOM/HETATM line."""
        serial = int(line[6:11].strip())
        name = line[12:16].strip()
        alt_loc = line[16:17].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21:22].strip()
        residue_seq = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
        temp_factor = float(line[60:66]) if line[60:66].strip() else 0.0
        element = line[76:78].strip() or name[0]

        mass = self.ATOMIC_MASSES.get(element.upper(), 12.011)

        return Atom(
            serial=serial, name=name, alt_loc=alt_loc,
            residue_name=residue_name, chain_id=chain_id,
            residue_seq=residue_seq,
            position=np.array([x, y, z], dtype=np.float64),
            occupancy=occupancy, temp_factor=temp_factor,
            element=element, mass=mass,
        )

    def _generate_bonds(self, atoms: List[Atom], cutoff: float = 2.0) -> List[Tuple[int, int]]:
        """Generate bonds from interatomic distances."""
        bonds = []
        positions = np.array([a.position for a in atoms])

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    bonds.append((i, j))

        return bonds

    def _generate_angles(self, bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """Generate angles from bonded triplets."""
        angles = []
        bond_dict = {}

        for i, j in bonds:
            bond_dict.setdefault(i, []).append(j)
            bond_dict.setdefault(j, []).append(i)

        for i in bond_dict:
            partners = bond_dict[i]
            for a in range(len(partners)):
                for b in range(a + 1, len(partners)):
                    angles.append((partners[a], i, partners[b]))

        return angles

    def _generate_dihedrals(self, angles: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Generate dihedrals from angle pairs sharing a bond."""
        dihedrals = []
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                a1 = angles[i]
                a2 = angles[j]
                if a1[1] == a2[0] and a1[2] == a2[1]:
                    dihedrals.append((a1[0], a1[1], a2[1], a2[2]))
                elif a1[0] == a2[1] and a1[1] == a2[0]:
                    dihedrals.append((a1[2], a1[1], a2[1], a2[2]))
        return dihedrals
