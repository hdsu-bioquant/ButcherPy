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
    python_requires=">=3.6",              
    install_requires=[
        # List your package's dependencies here, e.g.:
        # "numpy>=1.21.0",
    ],
)
