from setuptools import setup, find_packages
setup(
    name="BayesPaths",
    version="0.1",
    packages=["BayesPaths"],
    scripts=["bin/bayespaths"],
    install_requires=["matplotlib","numpy>=1.15.4","scipy>=1.0.0","pandas>=0.24.2","networkx>=2.4","sklearn","pygam>=0.8.0","gfapy"],
)
