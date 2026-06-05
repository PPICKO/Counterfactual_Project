"""Setup script for counterfactual fairness comparison package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="counterfactual-fairness-comparison",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comparison of Wachter and GLANCE counterfactual methods with FACTS fairness evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/counterfactual-fairness-comparison",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/counterfactual-fairness-comparison/issues",
        "Documentation": "https://github.com/yourusername/counterfactual-fairness-comparison#readme",
        "Source Code": "https://github.com/yourusername/counterfactual-fairness-comparison",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "folktables>=0.0.11",
        "statsmodels>=0.13.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-acs-experiment=run_acs_experiment:main",
            "run-acs-comparison=run_acs_comparison:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "counterfactual explanations",
        "explainable AI",
        "fairness",
        "machine learning",
        "algorithmic recourse",
        "FACTS",
        "Wachter",
        "GLANCE",
    ],
)
