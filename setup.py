#!/usr/bin/env python3
"""
Setup script for PlanSheet Scanner
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="plansheet-scanner",
    version="1.0.0",
    author="HNTB DIS SEA_DTS Python Working Group",
    author_email="",  # Add your email if desired
    description="A modular tool for extracting, processing, and georeferencing legend symbols from engineering plan sheets",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/plansheet-scanner-new",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "bandit>=1.6",
            "safety>=1.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "plansheet-scanner=src.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
