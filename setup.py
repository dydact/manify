from setuptools import setup, find_packages

setup(
    name="manify",
    version="0.1",
    packages=find_packages(where="."),
    package_dir={"": "."},
    requires=[
        "torch",
        "geoopt",
        "networkx",
        "numpy ",
        "pandas",
        "matplotlib",
        "scipy",
        "torchtyping",
    ],
)
