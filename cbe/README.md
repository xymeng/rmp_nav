# Learning Composable Behavior Embeddings for Long-horizon Visual Navigation

Check out the project [website](https://homes.cs.washington.edu/~xiangyun/ral21/) for more information.

## Preparation

* Follow the [README](../README.md) in the project root to install all the dependencies.
* Before you run any code, make sure you have run `source <project_root>/set_envs.sh`


### Trajectory datasets
Download the datasets and put them under `${RMP_NAV_ROOT}/data/gibson2/`. Keep the directory name.

* [training trajectories](https://drive.google.com/file/d/1Lqgp-ulRUo-FleteD3yLaNK2Prx0uggv/view?usp=sharing)
  * You should have something like `${RMP_NAV_ROOT}/data/gibson2/minirccar_agent_local_240fov_18env_slow/train_1.hd5`
* [test trajectories](https://drive.google.com/drive/folders/11l72WRrfG5EIiETPbdbF_es3AOfhGy4B?usp=sharing)
  * You should have something like `${RMP_NAV_ROOT}/data/gibson2/pairwise_destination_testenv/Calavo.hd5`


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

### Download the pretrained models
* Download the pretrained models from [here](https://drive.google.com/drive/folders/1SWA9N71EOW9z62lHWnxZlKtX2nLOSJfL?usp=sharing).
* Put them inside `${RMP_NAV_ROOT}/models/cbe/`

### Run the evaluator

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

### Launch the training environments:
```shell
cd ${RMP_NAV_ROOT}/tools
python launch_sim_server_load_balancer_multi.py gibson2_18envs_gpu01_128.yaml 5000
```
If you have more GPUs, you can modify `gibson2_18envs_gpu01_128.yaml` to add more servers with additional GPU ids.
You don't have to launch the training environments on the same machine used for training the network. You can launch
them on a different machine but you need to specify a different persistent server config file. The default config
file `configs/gibson_persistent_servers/local.yaml` points to the `localhost`, but you can change it to any host that
runs the simulation environments.

### Start the dataset server
Start the dataset server for dagger training:
```shell
python -u dataset_server.py --addr='tcp://*:5002' --n_worker=18 \
--param_file dataset_server_config/cbe.yaml \
--devices='cuda:0' \ 
--n_tracker=6 \
--class_name=DatasetGibson2TrajsDagger
```
* Adjust `--n_worker`, `--devices`, `--n_tracker` to adapt to the number of available CPUs and GPUs.
  * If you have 16 CPUs, you can set `--n_worker` to 16.
  * `--devices` is a comma separated list of devices in pytorch format.    
  * You can set `--n_tracker` to `2 * number of GPUs`.

* If you have multiple machines, you can run one dataset server on each machine. This can significantly speed up dagger
training. To do so, add the host names of the machines to `dagger_constructor()` in `train_cbe.py`

* You also need to launch the training environments on each machine that runs the dataset server. This is not strictly
  necessary, but would require configuring the persistent servers in the dataset server config file.
  
* Since during dagger training the main training machine will send network weights to the dataset server machines,
  you need to set up ssh key authentication so that `scp` from the main training machine to the dataset server
  machines does not ask for the password.

### Train the network
Change current working directory:
```cd ${RMP_NAV_ROOT}/cbe/experiments```

Launch the visdom server on a separate terminal window:
```visdom -port=5001```

Run the training script
```shell
OMP_NUM_THREADS=2 bash run_gibson2_18env.sh gibson2_18env/default default \
--visdom_server='http://localhost' \
--persistent_server_cfg ../../configs/gibson_persistent_servers/local.yaml \
--batch_size=24 --n_dataset_worker=64 --log_interval=100 --save_interval=1
```
Modify `--persistent_server_cfg` if you have other machines running the gibson simulators.

In general we do not recommend training the model with just two GPUs, because it would take a very long time.
Here is a roughly guideline of how much GPU resource is required:
* One GPU with > 10GB memory for training the network.
* 4 to 8 GPUs for running gibson simulators and Dagger inference.
* 20 CPUs for running robot simulation.

If a single machine does not have sufficient resources, you can train the model on multiple machines. For example,
* One machine for training the network.
* Two machines with 4 GPUs each run the dataset servers and simulation environments.
