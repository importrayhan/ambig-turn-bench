"""Setup script for ConQA-Bench."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ambig-turn-bench",
    version="0.1.0",
    author="Md-Rayhan",
    description="Ambiguous Query Detection Evaluation Suite",
    long_description="Comprehensive assorted benchmarking for quickly ranking ambiguous query detection models",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.0.0",
        "python-terrier>=0.10.0",
        "pyterrier-rag>=0.1.0",
        "tqdm>=4.60.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
