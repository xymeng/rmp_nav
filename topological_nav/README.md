# Scaling Local Control to Large Scale Topological Navigation

This repository contains related code for 
- Building sparse topological maps from dense trajectories
- Performing visual topological navigation in Gibson environments

## Preparation
- Follow [README](../README.md) to set up the base code.
- Download the [datasets and models](https://drive.google.com/file/d/1iCjMLjBsL9_fSRC8qVAsj0BoKzPm8gHy/view?usp=sharing).
  Put it inside the source root and run `tar xf topo_nav.tar.gz`.
  
## Run the code
Currently we provide you with a license-free environment (space8) for you to test.

First launch the Gibson simulation server:

```
cd tools
python launch_sim_server_load_balancer_multi.py gibson_space8_gpu0_128.yaml 5000
```

### Local controller
You can then test the controller by running the following command

```
python topological_nav/tools/eval_controller_multiframe_dst.py
```

### Trajectory following
This first sparsifies a dense trajectory using the reachability estimator (RE), and then follows the trajectory using
both RE and the controller.

```
python topological_nav/tools/eval_traj_following.py
```

### Topological navigation
Code being cleaned up. Coming soon!
