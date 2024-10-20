#!/usr/bin/env python

import sys
from pathlib import Path

import numpy as np
from deepmd.infer import DeepPot


def symbol2num(symbol: str) -> int:
    table = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Th": 90,
        "Pa": 91,
        "U": 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Cn": 112,
        "Nh": 113,
        "Ut": 113,
        "Fl": 114,
        "Mc": 115,
        "Up": 115,
        "Lv": 116,
        "Ts": 117,
        "Us": 117,
        "Og": 118,
        "Uo": 118,
        "Un": 119,
        "Ux": 120,
    }
    return table[symbol]


def convert_raw(type_map_path: Path) -> dict[int, int]:
    with type_map_path.open() as f:
        type_map = f.read().splitlines()
    return {symbol2num(symbol): i for i, symbol in enumerate(type_map)}


def to_sysdata(inp_file: Path, type_map_path: Path) -> tuple[list, np.ndarray, int]:
    atomic_nums = []
    coord = []
    with inp_file.open(mode="r") as f:
        natom, derivative, charge, multiplicity = map(int, f.readline().strip().split())
        for line in f:
            atomic_num, x, y, z, _ = line.split()
            atomic_nums.append(int(atomic_num))
            coord.append([x, y, z])

    type_map = convert_raw(type_map_path)
    atom_type = [
        type_map[atomic_num] for atomic_num in atomic_nums
    ]  # TODO(hayashi): Check if this structure is correct
    coord = np.array(coord, dtype=np.float32).reshape([1, -1])  # TODO(hayashi): Check if this structure is correct
    return atom_type, coord, derivative


def write_out(output_file: Path, derivative: int, energy: float, forces: np.ndarray) -> None:
    with output_file.open(mode="w") as f:
        f.write(f"{energy:>20.12f}{0:>20.12f}{0:>20.12f}{0:>20.12f}\n")  # energy, dipole(x), dipole(y), dipole(z)
        if derivative >= 1:
            forces = forces.reshape([-1, 3])
            for force in forces:
                f.write(f"{force[0]:>20.12f}{force[1]:>20.12f}{force[2]:>20.12f}\n")  # force(x), force(y), force(z)


def main():
    # Parse the arguments
    model_path = Path(sys.argv[1])
    type_map_path = Path(sys.argv[2])
    _ = sys.argv[3]
    input_file = Path(sys.argv[4])
    output_file = Path(sys.argv[5])

    # Parse the input files
    atom_type, coord, derivative = to_sysdata(input_file, type_map_path)

    # Load the trained model
    model = DeepPot(model_path)
    energy, forces, _ = model.eval(coords=coord, cells=None, atom_types=atom_type)

    # Write the output files
    write_out(output_file, derivative, energy, forces)


if __name__ == "__main__":
    main()
    sys.exit(0)
