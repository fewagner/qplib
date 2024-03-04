import os
from setuptools import setup

setup(
    name = "qplib",
    version = "0.0.1",
    author = "Felix Wagner",
    author_email = "felix.wagner@phys.ethz.ch",
    description = ("Methods to understand quasiparticle tunneling."),
    license = "MIT",
    packages=['qplib','tests'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)