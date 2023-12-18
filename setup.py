import os

from setuptools import setup

PACKAGE_NAME = "geniml"

# Ordinary dependencies
DEPENDENCIES = []
with open("requirements/requirements-all.txt", "r") as reqs_file:
    for line in reqs_file:
        if not line.strip():
            continue
        DEPENDENCIES.append(line)

# Additional keyword arguments for setup()
extra = {"install_requires": DEPENDENCIES}

with open(PACKAGE_NAME + "/_version.py", "r") as versionfile:
    version = versionfile.readline().split()[-1].strip("\"'\n")

with open("README.md") as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    packages=[
        PACKAGE_NAME,
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
        "geniml.classification",
        "geniml.nn",
    ],
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Genomic interval toolkit",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="BSD2",
    entry_points={
        "console_scripts": [
            "geniml = geniml.cli:main",
            "bedshift = geniml.bedshift.bedshift:main",
        ],
    },
    keywords="bioinformatics, sequencing, ngs",
    package_data={"geniml": [os.path.join("geniml", "*")]},
    include_package_data=True,
    url="http://geniml.databio.org",
    author="Nathan Sheffield",
    **extra,
)
