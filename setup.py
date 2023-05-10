import os
import sys

from setuptools import setup

PACKAGE_NAME = "gitk"

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
        "gitk.assess",
        "gitk.eval",
        "gitk.hmm",
        "gitk.likelihood",
        "gitk.scembed",
        "gitk.utils",
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
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="BSD2",
    entry_points={
        "console_scripts": [
            "gitk = gitk.cli:main",
        ],
    },
    keywords="bioinformatics, sequencing, ngs",
    package_data={"refgenie": [os.path.join("refgenie", "*")]},
    include_package_data=True,
    url="http://giss.databio.org",
    author="Nathan Sheffield",
    **extra
)
