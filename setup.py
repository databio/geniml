import os

from setuptools import setup

PACKAGE_NAME = "geniml"

# Ordinary dependencies
DEPENDENCIES = []
with open("requirements/requirements-basic.txt", "r") as reqs_file:
    for line in reqs_file:
        if not line.strip():
            continue
        DEPENDENCIES.append(line)

# Additional keyword arguments for setup()
extra = {"install_requires": DEPENDENCIES}

with open(PACKAGE_NAME + "/_version.py", "r") as versionfile:
    version = versionfile.readline().split()[-1].strip("\"'\n")


# Optional dependencies
# Extras requires a dictionary and not a list?
def _read_reqs(path):
    with open(path, "r") as fh:
        return [line.strip() for line in fh if line.strip() and not line.strip().startswith("#")]


ml_dep = _read_reqs("requirements/requirements-ml.txt")
sc_dep = _read_reqs("requirements/requirements-sc.txt")
search_dep = _read_reqs("requirements/requirements-search.txt")
test_dep = _read_reqs("requirements/requirements-test.txt")

extra["install_requires"] = DEPENDENCIES
extra["extras_require"] = {
    "ml": ml_dep,
    "sc": sc_dep,
    "search": search_dep,
    "all": ml_dep + sc_dep + search_dep,
    "test": test_dep,
}


with open("README.md") as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    packages=[
        PACKAGE_NAME,
        "geniml.atacformer",
        "geniml.craft",
        "geniml.geneformer",
        "geniml.assess",
        "geniml.bedspace",
        "geniml.bedshift",
        "geniml.eval",
        "geniml.likelihood",
        "geniml.models",
        "geniml.region2vec",
        "geniml.scembed",
        "geniml.tokenization",
        "geniml.universe",
        "geniml.io",
        "geniml.text2bednn",
        "geniml.bbclient",
        "geniml.search",
        "geniml.search.backends",
        "geniml.search.interfaces",
        "geniml.search.query2vec",
        "geniml.nn",
    ],
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Genomic interval toolkit",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="BSD2",
    entry_points={
        "console_scripts": ["geniml = geniml.cli:main"],
    },
    keywords="bioinformatics, sequencing, ngs",
    package_data={"geniml": [os.path.join("geniml", "*")]},
    include_package_data=True,
    url="https://docs.bedbase.org/geniml/",
    author="Nathan Sheffield",
    **extra,
)
