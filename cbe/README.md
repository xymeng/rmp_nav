# Learning Composable Behavior Embeddings for Long-horizon Visual Navigation

Check out the project [website](https://homes.cs.washington.edu/~xiangyun/ral21/) for more information.

## Preparation

### Trajectory datasets
Download the datasets and put them under `${RMP_NAV_ROOT}/data/gibson2/`. Keep the directory name.

* [training trajectories] (to be released soon)
* [test trajectories](https://drive.google.com/drive/folders/11l72WRrfG5EIiETPbdbF_es3AOfhGy4B?usp=sharing)
  * You should have something like `${RMP_NAV_ROOT}/data/gibson2/pairwise_destination_testenv/*.hd5`


### Gibson2 datasets
**Make sure you have signed the agreement to use the official gibson datasets.**
 
* Download the environments from [here](https://drive.google.com/file/d/117q9zpi1z11_NXDQ8EYoxmPw7jtrGeeo/view?usp=sharing). 

* Move them to `${RMP_NAV_ROOT}/rmp_nav/gibson2`
  * You should have something like `${RMP_NAV_ROOT}/rmp_nav/gibson2/Aldrich/*`

There are some differences between ours and the official gibson2 dataset:
* We manually selected some of the largest environments in gibson2.
* We generated our own floorplans which are cleaner than some of those in the official datasets.
* We reduced the texture size to make them better fit into GPUs.

## Run the pretrained models

Before you run any code, make sure you have run `source <project_root>/set_envs.sh`

### Download the pretrained models
* Download the pretrained models from [here](https://drive.google.com/drive/folders/1SWA9N71EOW9z62lHWnxZlKtX2nLOSJfL?usp=sharing).
* Put them inside `${RMP_NAV_ROOT}/models/cbe/`

### Test

Launch the simulators for the test environments. Note that this will occupy the current terminal window.
```
source <project_root>/set_envs.sh
cd ${RMP_NAV_ROOT}/tools
python launch_sim_server_load_balancer_multi.py gibson2_testenvs_gpu01_128.yaml 5000
```

To evaluate a model, run:
```
source <project_root>/set_envs.sh
cd ${RMP_NAV_ROOT}/cbe

python eval_tracker.py \
--model=<model> \
--n_frame=64 \
--env=testenvs \
--start_idx=0 --n_traj=500 \
--nonoisy_actuation \
--jitter \
--noobstacle --obstacle_offset=0.0 \
--agent=minirccar_240fov_rmp_v2 \
--seed=12345 \ 
--visualize
```

This will evaluate the model on 500 test trajectories pre-generated in the test environments. 
You can tweak the command line parameters to evaluate the model in different scenarios:

* `--model=<str>` specifies the model. For available models check out `model_factory.py`. We have two pretrained 
  models to test: `cbe` and a baseline `rpf`.
  
* `--n_frame=<int> --frame_interval=<int>` control the length of the trajectory. `--frame_interval` specifies the gap
  between two frames. The length of a trajectory in terms of timesteps is `n_frame * frame_interval`. Our pretrained
  models were all trained with `n_frame=64` and `frame_interval=2`. You can play with different `n_frame` but don't
  change `frame_interval` for the pretrained models.
  
* `--start_idx=<int> --n_traj=<int>` control which part of the test dataset to evaluate.
* `--noisy_actuation` simulates noisy actuation.
* `--jitter` simulates initial misalignment.
* `--obstacle -obstacle_offset=<float>` simulates obstacles.
* `--env=<str>` specifies the evaluation environments.
* `--agent=<str>` specifies the robot.
* `--visualize` turns on visualization.

An example command line to test the pretrained model:
```
python eval_tracker.py --model=cbe --n_frame=64 --start_idx=0 --nonoisy_actuation --jitter --noobstacle --obstacle_offset=0.0 --agent=minirccar_240fov_rmp_v2 --seed=12345 --env=testenvs --n_traj=500 --visualize --nosave_trace --nosave_screenshot
```

An example command line to test the RPF baseline:
```
python eval_tracker.py --model=rpf --n_frame=64 --start_idx=16 --nonoisy_actuation --jitter --noobstacle --obstacle_offset=0.0 --agent=minirccar_240fov_rmp_v2 --seed=12345 --env=testenvs --n_traj=500 --visualize --nosave_trace --nosave_screenshot
```

## Training
(Stay tuned)
