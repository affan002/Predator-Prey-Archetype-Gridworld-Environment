# setup.py
from setuptools import setup, find_packages

setup(
    name="single_agent_package",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        # e.g. "gymnasium>=0.28"
    ],
)