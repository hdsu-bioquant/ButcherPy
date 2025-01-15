# ButcherPy 
  
`ButcherPy` is the Python (re)implementation of the R package [ButchR](https://github.com/hdsu-bioquant/ButchR). This package provides functions to perform non-negative matrix factorization using Tensorflow. 

You can use `ButcherPy` to obtain molecular signatures from different types of data - expression, epigenomics, among others. This software also provides easy interpretation of results with the help of extensive visualization options.

# Installation  

If you have access to Bioquant's NVIDIA GPU workstation you can follow these instructions:
1. Clone this repository.
2. Start the butcherpy docker container, by calling `docker run --gpus device=1 -p 8888:8888 --rm -ti -v /raid/username/projects/:$HOME pytorch:butcherpy` and make sure to specify a device with free memory available and use your username. You can read more about the workstation usage, like how to find out which device is free in this [Readme](https://github.com/hdsu-bioquant/dgx-workstation).
3. In the running docker container navigate to where you have cloned the repository `cd your_directory/ButcherPy`.

Now you are ready to use ButcherPy as all necessary packages are installed in the docker container. However, note that only ButcherPy requirements are installed, thus for example scanpy is not installed, wich you might need when working with AnnData. If you need further packages installed for your implementations you can add these packages to the butcherpy docker image, but please make sure to rename the image you are creating, so that the butcherpy docker image stays clean. The [Readme](https://github.com/hdsu-bioquant/dgx-workstation) provides clear instructions.

Otherwise, without Bioquant workstation access, you can do the following:
1. Clone this repository.
2. In the python environment of your choice (docker containers, virtual environment, etc.) install all necessary packages given in `setup.py`. Pay attention to version requirements.
3. In the environment navigate to where you have clone the repository `cd your_directory/ButcherPy` and you are ready to use ButcherPy.
 simply access a docker 
You can install the package from this GitHub repository using `pip`.

# Usage

Import necessary modules to perform NMF

```python
import src.butcherPy.multiplerun_NMF as multinmf
import src.butcherPy.nmf_run as nmf
```

 
# Citation

If you use `ButcherPy`, please cite the original publication of `ButchR`: 
Andres Quintero, Daniel Hübschmann, Nils Kurzawa, Sebastian Steinhauser, Philipp Rentzsch, Stephen Krämer, Carolin Andresen, Jeongbin Park, Roland Eils, Matthias Schlesner, Carl Herrmann, [ShinyButchR: interactive NMF-based decomposition workflow of genome-scale datasets](https://doi.org/10.1093/biomethods/bpaa022), Biology Methods and Protocols, bpaa022.
