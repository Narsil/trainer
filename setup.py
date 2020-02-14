# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="trainer",
    version="0.1.0",
    description="Simple trainer loop for Pytorch to avoid reusing same code everywhere.",
    long_description=readme,
    author="Nicolas Patry",
    author_email="patry.nicolas@protonmail.com",
    url="https://github.com/Narsil/trainer",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
)
