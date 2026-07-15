[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name            = "kr-defect"
version         = "1.0.0"
description     = "Multivariable K–R Defect Framework for Hessian Recovery"
readme          = "README.md"
license         = {text = "MIT"}
requires-python = ">=3.9"
authors         = [{name = "RamaKrishna Pasupuleti",
                    email = "workisfun415@gmail.com"}]
keywords        = [
    "hessian", "derivative-free", "finite-differences",
    "numerical-analysis", "boundary-constrained", "scattered-data"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Mathematics",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = ["numpy>=1.21"]

[project.optional-dependencies]
dev  = ["pytest>=7", "matplotlib>=3.5"]
docs = ["sphinx", "sphinx-rtd-theme"]

[project.urls]
Homepage   = "https://github.com/workisfun415/kr-defect-framework"
Repository = "https://github.com/workisfun415/kr-defect-framework"
Paper      = "https://doi.org/10.5281/zenodo.21339639"

[tool.setuptools.packages.find]
where = ["."]
include = ["kr_defect*"]
