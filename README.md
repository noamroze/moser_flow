# moserflow
This is the official repository for the paper 
> "Moser Flow: Divergence-based Generative Modeling on Manifolds"
[[arxiv](https://arxiv.org/abs/2108.08052)]


## Dependencies:  

create a conda enviorenment, we used python 3.7.
Then run 
'''
bash install_requirements.sh
'''


## Datasets
For the earth datasets unzip data/earth_data.zip

## Experiments
All results are stored by default in a directory ./_experiments
#### Toy 2D experiments
 
To reproduce the toy data experiments, run the following example cmd line
```
bash scripts/run_toy_data.sh
```
#### Camerman experiments
To run Moser Flow on the high resolution example, run 

```
bash scripts/run_high_density_moser.sh
```
To run FFJORD on the high resolution example, run 
```
bash scripts/run_high_density_ffjord.sh
```
These scripts will run the experiments and save checkpoints during the time processes, and no plots will be generated.
To compare the results of the two experiments use evaluate_time.py. 
For arguments definitions run
'''
python evaluate_time.py -h
'''
#### Earth experiments

To run the earth data experiments 
```
bash scripts/run_earth_data.sh
```
The code that visualizes the resulting densities is property of Facebook, hence unavailable. We present only the likelihood results.

#### Surface experiment
To learn the signed distance function of the stanford bunny, run
```
python eikonal/implicit_network_3d.py --pc_path eikonal/deep_geometric_prior_data/standford/bun_zipper.ply --name bunny
```
To run the experiment on the Stanford bunny, update the correct path to the saved checkpoint of the SDF in configs/surface.yml and run
```
bash scripts/run_surface.sh
```


