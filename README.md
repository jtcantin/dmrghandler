# dmrghandler

A wrapper for the DMRG code Block2 ([Repository](https://github.com/block-hczhai/block2-preview), [paper](https://doi.org/10.1063/5.0180424)) to facilitate preparing DMRG calculations, processing the results, and running on the Niagara Compute Canada cluster.

## Installation

```bash
python -m pip install --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/ git+https://github.com/jtcantin/dmrghandler
```

Update with reinstallation of dependencies (recommended whenever `pyproject.toml` has changed):
```bash
python -m pip install --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/ --force-reinstall git+https://github.com/jtcantin/dmrghandler
```

Update without reinstallation of dependencies:
```bash
python -m pip install --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/ --force-reinstall --no-deps git+https://github.com/jtcantin/dmrghandler
```

`--extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/` is included so that a release candidate version of Block2 can be used.

## Usage

A usage example for the Niagara cluster is `examples/example_niagara_prepare_calcs_gsee_benchmark_coarse_set.py`. After this is run, configuration files and submit commands are generated. 

A usage example for running locally can be found as part of qb-gsee-benchmark: [run_dmrg.ipynb](https://github.com/isi-usc-edu/qb-gsee-benchmark/blob/main/examples/run_dmrg.ipynb)

## License

`dmrghandler` was created by Joshua T. Cantin. It is licensed under the terms of the MIT license.

## Credits

`dmrghandler` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
