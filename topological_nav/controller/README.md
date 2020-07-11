## Train the Controller

First launch the simulators
```
cd tools
python launch_sim_server_load_balancer_multi.py gibson_12envs_gpu01_128.yaml 5000
```
Note that the simulation environments can take a lot of GPU memory. The above configuration uses two GPUs. 
If you notice out of memory error and have more than two GPUs, you can edit the yaml file to change
the GPU location of each environment.


Then run the following command to train the controller.


```
cd ${RMP_NAV_ROOT}/topological_nav/controller/experiments

OMP_NUM_THREADS=2 bash run_12env.sh \
12env/multiframe_dst/future/pair/pred_proximity/pred_heading_diff/conv \
gtwp-normwp-farwp-jitter-weightedloss-checkwp-nf6-interval3-dmax3 \
--vis_interval=20 --log_interval=20 --save_interval=1 \
--dmin=-0.5 --dmax=3.0 --overlap_ratio=0.1 \
--n_dataset_worker=32 \
--lr_decay_epoch=1 --samples_per_epoch=200000 \
--use_gt_wp --normalize_wp --jitter --weight_loss --check_wp \
--n_frame=6 --frame_interval=3 \
--model_variant='future_pair_conv' \
--persistent_server_cfg ../../../configs/gibson_persistent_servers/local.yaml \
--proximity_label --heading_diff_label
```

The model will be stored in `experiments/12env/multiframe_dst/future/pair/pred_proximity/pred_heading_diff/conv/`