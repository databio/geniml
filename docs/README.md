<p align="center">
<h1><img align="center" src="img/geniml_logo_horizontal.svg" class="img-header" height="100"></h1>
</p>


<p align="center">
<a href="https://pypi.org/project/geniml"><img src="https://img.shields.io/pypi/v/geniml" alt=""></a>
<a href="https://github.com/databio/geniml"><img src="https://img.shields.io/badge/source-github-354a75?logo=github"></a>
</p>



## Introduction

Geniml is a *genomic interval machine learning toolkit*, a Python package for building machine learning models of genomic interval data (BED files). It also includes ancillary functions to support other types of analyses of genomic interval data.

As of February 2024, this package and its documentation are undergoing rapid development, leading to some tutorials getting outdated. Please raise [github issues](https://github.com/databio/geniml) if you find outdated or unclear directions, so we know where to focus effort that will benefit users.

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



## Modules and resources

### Organization

`geniml` is organized into modules. The modules section gives an [overview of each module](modules.md).

### Browsing by publication

If you're coming here from a manuscript, you might find it easier to identify the tutorials relevant for a particular manuscript by visiting the landing page for the publication of interest. You can find documentation organized by manuscript in the [manuscripts section](/citations/).


