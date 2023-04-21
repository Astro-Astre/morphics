# Morphics: gaLaxy survEy MorpholOgical classification by bayesian cNn
[![arxiv](1)](1)
[![Build Status](https://travis-ci.org/LSSTDESC/Morphics.svg?branch=master)](https://travis-ci.org/LSSTDESC/Morphics)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/Morphics/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/Morphics?branch=master)
[![Documentation Status](https://readthedocs.org/projects/morphics/badge/?version=latest)](http://morphics.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/morphics.svg)](https://badge.fury.io/py/morphics)

LEMON is a python package for galaxy morphology classification using bayesian convolutional neural networks (BCNNs). 
It is based on the Pytorch deep learning library. 
It is designed to be used with the [BASS+MzLS](https://legacysurvey.org) survey data, but can be used with any galaxy images after 
proper preprocessing and adversarial transfer learning, especially preparing for the next generation of large surveys such as the CSST, LSST, and Euclid surveys.

## Installation
You can install the latest version of LEMON from PyPI using pip:
```bash
pip install morphics
```
or from source:
```bash
git clone github.com/LSSTDESC/Morphics.git
cd Morphics
python setup.py install
```
## Documentation

### Tutorials
* [Tutorial 1: Training a BCNN model](
* [Tutorial 2: Using a pre-trained BCNN model](
* [Tutorial 3: Using a pre-trained BCNN model with adversarial transfer learning](
* [Tutorial 4: Predicting galaxy morphology using a LEMON model](

### Examples

### Contributing
## Citation
If you use LEMON in your research, please cite the following paper:
```bibtex
@article{Morphics,
  author = {Renhao Ye, Shiyin Shen, Rafael},
    title = {LEMON: gaLaxy survEy MorpholOgical classification by bayesian cNn},
    journal = {arXiv preprint arXiv:},
         year = {2023}
}
```
## License
This package is released under the MIT License (refer to the LICENSE file for details).

## Contact
If you have any questions or comments, please contact Renhao Ye (yerenhao22@mails.ucas.ac.cn)

## Acknowledgements


[...]