# Py-iCARE
<p align="center">
<img src="./images/pyicare-logo.png" style="width: 40%;">
</p>

**Py-iCARE** is a Python distribution of iCARE, a tool for **i**ndividualized **C**oherent **A**bsolute **R**isk **E**stimation. iCARE allows users to build, validate, and apply absolute risk models. [Absolute risk](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/absolute-risk) quantifies the chance of an event occurring. For example, the likelihood that a healthy individual, of a given age and a risk factor profile, will develop the disease of interest over a specified time interval.

The original iCARE was written in R and its archived version (1.26.0) is available via Bioconductor at: https://www.bioconductor.org/packages/release/bioc/html/iCARE.html.

## Motivation
The main motivation of porting iCARE from R to Python was to enable its use as a [WebAssembly](https://webassembly.org/) module (via [Pyodide](https://pyodide.org/en/latest/index.html)) for the proliferation of portable and privacy-preserving web applications that can build, validate, and apply absolute risk models. Python also enables researchers to leverage its rapidly evolving data science ecosystem— including [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [statsmodels](https://www.statsmodels.org/stable/index.html), and [scikit-learn](https://scikit-learn.org/stable/)— to explore novel absolute risk models that can incorporate evidence from wider sources of data.

## Installation

This repository contains a Python package. It can be installed via PyPI as shown below. It is also compiled into WebAssembly via Pyodide. The wrapper JavaScript library, as ES6 modules, is also distributed in this repository at GitHub Pages. It can be accessed by any JavaScript runtime environment, including Node.js, web browsers, and Quarto notebooks in RStudio (to interface with R, Julia, and/or Python). Specifically, the steps to access it via JavaScript and R are also shown below.

If you want to access iCARE from a Python runtime environment, you can install it via [PyPI](https://pypi.org/project/pyicare/):

```bash
pip install pyicare
```

iCARE is supported on Python 3.9 and above.

## Usage

Once installed, Py-iCARE can be imported into your Python scripts as follows:

```python
import icare
```

Py-iCARE is a library with three main methods:

1. `compute_absolute_risk()`: a method to build and apply absolute risk models. Based on the type of risk factors present in the model and what information is available, there can be three broad variations in using this method:
   1. **Special SNP-only absolute risk model**: this variation shows you how to specify a SNP-based absolute risk model without the need to provide a reference dataset to represent the risk factor distribution of the underlying population. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeyabbalas/py-icare/blob/master/demo/Special%20SNP-only%20absolute%20risk%20model.ipynb) 
   2. **Covariate-only absolute risk model**: this option shows you how to specify a risk model with any type of covariates (including classical questionnaire-based risk factors and/or SNPs) so long as a reference dataset is available to represent the distribution of all the covariates in the underlying population. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeyabbalas/py-icare/blob/master/demo/Covariate-only%20absolute%20risk%20model.ipynb)
   3. **Combined SNP and covariate absolute risk model**: this option shows you how to specify a risk model that contains both SNPs and other type of covariates, such that, you have the reference dataset to represent the distribution of the covariates in the underlying population but you do not have the reference dataset to represent the SNP distribution. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeyabbalas/py-icare/blob/master/demo/Combined%20SNP%20and%20covariate%20absolute%20risk%20model.ipynb)
2. `compute_absolute_risk_split_interval()`: a method to build and apply absolute risk models that relaxes the proportional hazard assumption, to some extent, by allowing you to specify different model parameters that vary before and after a cut-point in time. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeyabbalas/py-icare/blob/master/demo/Absolute%20risk%20over%20split%20intervals.ipynb)
3. `validate_absolute_risk_model()`: a method to validate absolute risk models on an independent cohort study data or a case-control study nested within a cohort. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeyabbalas/py-icare/blob/master/demo/Model%20validation.ipynb)

The Jupyter notebooks for all the use-cases described above is available at the [demo](https://github.com/jeyabbalas/py-icare/tree/master/demo) directory of this repository.

## iCARE for the web
Wasm-iCARE is a distribution of Py-iCARE for web applications(applications running on web browser, Node.js, RStudio's Quarto, etc.). Applications where portability and privacy is critical can make use of Wasm-iCARE. Wasm-iCARE is available at: https://github.com/jeyabbalas/wasm-icare.

## License
Py-iCARE is open-source licensed under the MIT License.

## References
1. [Balasubramanian, Jeya Balaji, et al. "Wasm-iCARE: a portable and privacy-preserving web module to build, validate, and apply absolute risk models." arXiv preprint arXiv:2310.09252 (2023).](https://arxiv.org/abs/2310.09252)
2. [Pal Choudhury, Parichoy, Paige Maas, Amber Wilcox, William Wheeler, Mark Brook, David Check, Montserrat Garcia-Closas, and Nilanjan Chatterjee. "iCARE: an R package to build, validate and apply absolute risk models." PloS one 15, no. 2 (2020): e0228198.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7001949/)
