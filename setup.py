from setuptools import setup, find_packages

setup(
    name="ButcherPy",            
    version="0.1.0",                     
    author="Leonie Boland",                  
    author_email="hw199@stud.uni-heidelberg.de", 
    description="This package provides functions to perform non-negative matrix factorization using Tensorflow", 
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown", 
    url="https://github.com/hdsu-bioquant/ButcherPy",
    packages=find_packages(),             
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  # Choose a license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",              
    install_requires=[
        "anndata>=0.11.1",
        "matplotlib>=3.8.2",
        "numpy>=2.2.1",
        "pandas>=2.2.3",
        "plotly>=5.23.0",
        "rds2py==0.4.2",
        "scikit_learn>=1.4.0",
        "scikit_learn_extra>=0.3.0",
        "scipy>=1.15.0",
        "seaborn>=0.13.2",
        "setuptools>=59.5.0",
        "torch>=1.13.1"
    ],
)
