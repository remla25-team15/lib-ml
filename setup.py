from setuptools import setup, find_packages

setup(
    name="libml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn==1.3.0"
    ],
)
