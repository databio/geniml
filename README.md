# Genomic interval machine learning (geniml)

Geniml is a python package for building machine learning models of genomic interval data (BED files). It also includes ancillary functions to support other types of analyses of genomic interval data.

Documentation is hosted at <https://docs.bedbase.org/geniml/>.


## Installation
### To install `geniml` use this commands.

Without specifying dependencies, the default dependencies will be installed, 
which DO NOT include machine learning (ML) or heavy processing libraries.


From pypi:
```
pip install geniml
```
or install the latest version from the GitHub repository:
```
pip install git+https://github.com/databio/geniml.git
```

### To install Machine learning dependencies use this command:

From pypi:
```
pip install geniml[ml]
```


## Development

Run tests (from `/tests`) with `pytest`. Please read the [contributor guide](https://docs.bedbase.org/geniml/contributing/) to contribute.
