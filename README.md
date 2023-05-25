# Py-iCARE

**Py-iCARE** is a Python distribution of iCARE, a tool for **i**ndividualized **C**oherent **A**bsolute **R**isk **E**stimation. iCARE allows users to build, validate, and apply absolute risk models. [Absolute risk](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/absolute-risk) quantifies the chance of an event occurring. For example, the likelihood that a healthy individual, of a given age and a risk factor profile, will develop the disease of interest over a specified time interval.

The original iCARE was written in R and its archived version (1.26.0) is available via Bioconductor at: https://www.bioconductor.org/packages/release/bioc/html/iCARE.html.

## Motivation
The main motivation of porting iCARE from R to Python was to enable its use as a [WebAssembly](https://webassembly.org/) module (via [Pyodide](https://pyodide.org/en/latest/index.html)) for the proliferation of portable and privacy-preserving web applications that can build, validate, and apply absolute risk models. Python also enables researchers to leverage its rapidly evolving data science ecosystem— including [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [statsmodels](https://www.statsmodels.org/stable/index.html), and [scikit-learn](https://scikit-learn.org/stable/)— to explore novel absolute risk models that can incorporate evidence from wider sources of data.

## Installation

This repository contains a Python package. It can be installed via PyPI as shown below. It is also compiled into WebAssembly via Pyodide. The wrapper JavaScript library, as ES6 modules, is also distributed in this repository at GitHub Pages. It can be accessed by any JavaScript runtime environment, including Node.js, web browsers, and Quarto notebooks in RStudio (to interface with R, Julia, and/or Python). Specifically, the steps to access it via JavaScript and R are also shown below.

### Python

If you want to access iCARE from a purely Python runtime environment, you can install it via PyPI:

```bash
pip install pyicare
```

iCARE is supported on Python 3.7 and above.

### JavaScript
ES6 import JS SDK via GitHub Pages.

### R
ES6 import JS SDK via Quarto.

## Usage

Once installed, Py-iCARE can be imported into your Python scripts as follows:

```python
import icare
```

Py-iCARE is a library with three main functions: 1) `compute_absolute_risk()`, a method to build and apply absolute risk models; 2) `compute_absolute_risk_split_interval()`, a method to build and apply absolute risk models that relaxes the proportional hazard assumption to some extent by allowing you to specify different model parameters before and after a cut-point in time; and 3) `validate_absolute_risk_model()`, a method to validate absolute risk models on an independent cohort study data or a case-control study nested within a cohort.

Example usages of these functions are shown in Jupyter notebooks at the [examples/Python](https://github.com/jeyabbalas/py-icare/tree/master/examples/Python) directory.

### R
Quarto

### JavaScript
ES6 import into script

## License
Py-iCARE is open-source licensed under the MIT License.

## References
1. [Pal Choudhury, Parichoy, Paige Maas, Amber Wilcox, William Wheeler, Mark Brook, David Check, Montserrat Garcia-Closas, and Nilanjan Chatterjee. "iCARE: an R package to build, validate and apply absolute risk models." PloS one 15, no. 2 (2020): e0228198.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7001949/)
