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
    packages=setuptools.find_packages(exclude=['*.proflogit', '*.optimizers']),
    install_requires=[
        'numba>=0.57.0',
        'numpy>=1.24.2',
        'patsy>=0.5.3',
        'scikit_learn>=1.2.1',
        'scipy>=1.10.1',
        'xgboost>=1.7.4'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
        ],
    python_requires='>=3.9',
)
