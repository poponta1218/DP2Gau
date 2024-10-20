# DP2Gau

This project provides an interface for using a trained DeepMD model in conjunction with Gaussian. It parses atomic numbers and atomic coordinates from Gaussian input files, uses a DeepPot-SE model to predict energies and forces, and writes the output back in a format readable by Gaussian.

## Requirements

To run the script, you will need the following:

- Python: 3.10 or higher
- `deepmd-kit`: Tested with version 2.2.11
- `numpy`: Tested with version 1.26.4

## Installation

1. Clone the repository

    ```bash
    git clone git@github.com:poponta1218/DP2Gau.git
    ```

2. Install the required dependencies:
    1. Using `pip`:

        ```bash:pip
        pip install -r requirements.txt
        ```

    2. Using `uv`:

        ```bash:uv
        uv sync
        ```

## Usage

To use the script, add the following command in the `external` keyword section of your Gaussian input file:

```bash
python dp2gau.py <model_path> <type_map_path>"
```

### Arguments

- `model_path`: Path to the trained DeepMD model (`.pb` file).
- `type_map_path`: Path to a file mapping element symbols to atomic types used by the DeepMD model (typically `type_map.raw`).

The `type_map.raw` should contain a list of element symbols, one per line, corresponding to the order of atomic types expected by the DeepMD model.
The following `type_map.raw` file maps the element symbols `H`, `O`, and `C` to atomic types `0`, `1`, and `2`, respectively:

```txt:type_map.raw
H
O
C

```



### Example

Here is an example of a Gaussian input file that integrates `dp2gau.py`:

```gaussian:example.gjf
# Opt=NoMicro External="python dp2gau.py model.pb type_map.txt"

example

0 1
H 0.0 0.0 0.0
H 0.0 0.0 1.0

```

## How It Works

1. The script reads the atomic numbers and atomic coordinates from the Gaussian input file.
2. It converts atomic numbers into atomic types using the mapping provided in the `type_map.txt`.
3. The DeepPot model computes the total energy and forces for the given atomic coordinates.
4. The results are written in a Gaussian-readable format.
5. Gaussian reads the output and continues the calculation.

### Limitations

- Hessian (second derivative) calculations are not currently supported.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

### Third-Party Libraries

This project uses the following third-party libraries:

- `deepmd-kit`: Licensed under the GNU LGPLv3 License. See the [DeepMD-kit's LICENSE](https://github.com/deepmodeling/deepmd-kit/blob/master/LICENSE) for details.
- `numpy`: Licensed under the BSD 3-Clause License. See the [NumPy's LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt) for details.
