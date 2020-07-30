# Scaling Local Control to Large Scale Topological Navigation

This repository contains related code for 
- Building sparse topological maps from dense trajectories
- Performing visual topological navigation in Gibson environments

See the [supplementary video](https://homes.cs.washington.edu/~xiangyun/topological_nav/videos/submission.mp4).

## Preparation
- Follow [README](../README.md) to set up the base code.
- Download the [datasets and models](https://drive.google.com/file/d/1qpVgmJUbMa8z3pOIpTDs4qHlc3mqBZXD/view?usp=sharing).
  Put it inside the source root and run `tar xf topo_nav.tar.gz`.
  
## Run the code
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
#### Build a topological map (simulation)

We provide a trajectory dataset `data/minirccar_agent_local_240fov_space8_pairwise_destination`, which contains 90
trajectories in `space8`.
To build the map, run the following commands
```
cd topological_nav/tools
bash build_graph.sh space8_pairwise_destination ../nav_graphs/space8/graph_config.yaml
```

This may take a while. After it completes, the map will be stored in `topological_nav/nav_graphs/space8/graph.pickle`.
In that directory you can also check out `graph.svg` which visualizes the map.

#### Build a topological map (real)
We provide an example of building a map from real images. Please follow the steps below:

* Download the [dataset](https://drive.google.com/file/d/1PBkhpfJMUTjWbj90slfk5frpQ4q1iGC2/view?usp=sharing) and put it inside `data/`. You should have `data/cse2roboticslab/traces/...`
* Download the [map](https://drive.google.com/file/d/16EYgVAczPKORRPi0fy2CQhgGwQNQLmSU/view?usp=sharing) for visualization. Put it inside `rmp_nav/gibson/assets/dataset/`. You should have `dataset/cse2roboticslab/...`
* Run the following command:
  ```
  bash build_graph.sh '' ../nav_graphs/cse2roboticslab/graph_config.yaml
  ```


#### Planning

Once you have built the map, you can try out the planning part. You can reproduce the "resolve ambiguity" part of the
[supplementary video](https://homes.cs.washington.edu/~xiangyun/topological_nav/videos/submission.mp4) by running the
following commands:

```
cd topological_nav/tools
python eval_planning.py --model=model_12env_v2_future_pair_proximity_z0228 --graph_save_file=../nav_graphs/space8/graph.pickle --task=plan_to_dest_single --start_pos="10,0" --start_heading="30" --goal="(128,0,0)" --online_planning --visualize --seed=54321 --model_param="search_thres=0.7"
```

## Train the models
Download the [simulation environments](https://drive.google.com/file/d/1O-JzLFVjGfJ-SnvW9jMh6DqcBpI6anj1/view?usp=sharing).
Put the simulation environments inside `rmp_nav/gibson/assets/dataset/`.
You should have directories like `rmp_nav/gibson/assets/dataset/house24`.

Download the [trajectory dataset](https://drive.google.com/file/d/11QAtu7Z1HrF5RUCxCc1xO756pPyEiIqR/view?usp=sharing).
Put the dataset inside `data/`. You should have a directory structure like `data/minirccar_agent_local_240fov_12env_v2_farwp/train_1.hd5`.

To train the controller, see [controller/README.md](controller/README.md).
