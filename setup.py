"""
Setup script for AutoML System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="auto-ml-system",
    version="1.0.0",
    author="Auto ML System",
    author_email="",
    description="소량의 데이터를 위한 자동 머신러닝 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "full": [
            "imblearn>=0.11.0",
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "catboost>=1.2.0",
            "optuna>=3.3.0",
            "ctgan>=0.7.0",
            "copulas>=0.9.0",
            "shap>=0.42.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "automl=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "auto_ml_system": ["*.yaml", "*.json"],
    },
)