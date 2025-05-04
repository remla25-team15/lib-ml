from setuptools import setup, find_packages

setup(
    name="libml",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "scikit-learn==1.3.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "nltk==3.9",
        "joblib>=1.1.0",
    ],
)
