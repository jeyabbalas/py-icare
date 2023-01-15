# Py-iCARE

**Py-iCARE** is a Python distribution of iCARE, a tool for <u>i</u>ndividualized <u>C</u>oherent <u>A</u>bsolute <u>R</u>isk <u>E</u>stimation. iCARE allows users to build, validate, and apply absolute risk models. [Absolute risk](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/absolute-risk) quantifies the chance of an event occurring. For example, the likelihood that a healthy individual, of a given age and a risk factor profile, will develop the disease of interest over a specified time interval.

The original iCARE was written in R and its archived version (1.26.0) is available via Bioconductor at: https://www.bioconductor.org/packages/release/bioc/html/iCARE.html.

## Motivation
The main motivation of porting iCARE from R to Python was to enable its use as a [WebAssembly](https://webassembly.org/) module (via [Pyodide](https://pyodide.org/en/latest/index.html)) for the proliferation of portable and privacy-preserving web applications that can build, validate, and apply absolute risk models. Python also enables researchers to leverage its rapidly evolving data science ecosystem, including [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [statsmodels](https://www.statsmodels.org/stable/index.html), and [scikit-learn](https://scikit-learn.org/stable/). This encourages the development of machine learning-based absolute risk models.
