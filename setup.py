import setuptools
import empulse

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="empulse",
    version=empulse.__version__,
    author="Shimanto Rahman",
    author_email="shimanto.rahman@ugent.be",
    description="EMP metrics and models for scikit-learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShimantoRahman/empulse",
    packages=setuptools.find_packages(),
    install_requires=[
        'numba>=0.57.0',
        'numpy>=1.24.2',
        'scikit_learn>=1.2.1',
        'scipy>=1.10.1',
        'xgboost>=1.7.4',
        'joblib>=1.3.2',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    python_requires='>=3.9',
)
