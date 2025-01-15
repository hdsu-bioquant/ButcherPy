# ButcherPy 
  
***ButcherPy*** is the Python (re)implementation of the R package [***ButchR***](https://github.com/hdsu-bioquant/ButchR). This package provides functions to perform non-negative matrix factorization (NMF) using Tensorflow. 

You can use ***ButcherPy*** to obtain molecular signatures from various types of data - such as gene expression or epigenomics datasets. This software also provides intuitive interpretation of results through extensive visualization options.

# Installation  

### Using Bioquant's NVIDIA GPU W
If you have access to Bioquant's workstation, follow these steps:
1. Clone this repository:
   ```bash
   git clone https://github.com/hdsu-bioquant/ButcherPy.git

2. Start the `butcherpy` Docker container. To do that replace `device=1`with the device ID that has available memory, and substitute `username` with your Bioquant username in the following command. For information on checking GPU availability, refer to [DGX Workstation Readme](https://github.com/hdsu-bioquant/dgx-workstation).
   ```bash
   docker run --gpus device=1 -p 8888:8888 --rm -ti -v /raid/username/projects/:$HOME pytorch:butcherpy
   
3. In the running Docker container navigate to the directory where you cloned this repository:
   ```bash
   cd your_directory/ButcherPy

At this point, you are ready to use ***ButcherPy*** as the Docker container comes pre-installed with all required packages.
***Note***: The Docker container only includes ***ButcherPy***'s dependencies. Additional packages like `scanpy` (commonly needed for AnnData) are not pre-installed. If you need extra packages, extend the `butcherpy` Docker image with the required packages. Be sure to save the customized image with a unique namr to keep the official `butcherpy` image unchanged. More details on this process are available in the [DGX Workstation Readme](https://github.com/hdsu-bioquant/dgx-workstation).

### Without Bioquant Workstation Access
If you do not have access to the Bioquant GPU workstation:
Otherwise, without Bioquant workstation access, you can do the following:
1. Clone this repository:
   ```bash
   git clone https://github.com/hdsu-bioquant/ButcherPy.git
   
2. Use a python environment of your choice (e.g., Docker, virtualenv, etc.) and install all necessary packages specified in `setup.py`. Pay attention to version requirements.
3. Navigate to the repository directory:
   ```bash
   cd your_directory/ButcherPy
   

You are now ready to use ***ButcherPy***.

# Usage

To perform NMF and analyze results, import the necessary modules:

```python
import src.butcherPy.multiplerun_NMF_class as nmf_class
import src.butcherPy.nmf_run as nmf
```

After preparing your data, run NMF and interpret the results using the NMF class:

```python
nmf_obj = nmf.multiple_rank_NMF(data,
                                ranks=[3, 4, 5],
                                n_initializations=10,
                                iterations=100,
                                seed=123)
nmf_obj.compute_OptKStats_NMF()
nmf_obj.compute_OptK()
nmf_obj.WcomputeFeatureStats()

save_path="SignatureHeatmap"
nmf_obj.signature_heatmap(save_path)
```

This is just a basic example. For your real-world dataset, consider increasing the number of ranks, initializations, and iterations. Refer to the `nmf_vignette` for a comprehensive guide on data preparation, usage of analytical functions, and visualizations.
 
# Citation

If you use ***ButcherPy***, please cite the original publication of ***ButchR***: 
Andres Quintero, Daniel Hübschmann, Nils Kurzawa, Sebastian Steinhauser, Philipp Rentzsch, Stephen Krämer, Carolin Andresen, Jeongbin Park, Roland Eils, Matthias Schlesner, Carl Herrmann, [ShinyButchR: interactive NMF-based decomposition workflow of genome-scale datasets](https://doi.org/10.1093/biomethods/bpaa022), Biology Methods and Protocols, bpaa022.
