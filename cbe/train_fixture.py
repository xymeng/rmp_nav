import os
import time
import numpy as np
import tabulate
import torch
from torch.utils.data import DataLoader, RandomSampler
from rmp_nav.common.utils import save_model, load_model, module_grad_stats, module_weights_stats


def _load_weights(model_file, nets, net_opts):
    state = load_model(os.path.dirname(model_file),
                       os.path.basename(model_file), load_to_cpu=True)
    epoch = int(state['epoch'])

    for name, net in nets.items():
        net.load_state_dict(state['nets'][name])

    for name, opt in net_opts.items():
        opt.load_state_dict(state['optims'][name])
        # Move the parameters stored in the optimizer into gpu
        for opt_state in opt.state.values():
            for k, v in opt_state.items():
                if torch.is_tensor(v):
                    opt_state[k] = v.to(device='cuda')
    return epoch


def _save_model(nets, net_opts, epoch, global_args, model_file):
    state = {
        'epoch': epoch,
        'global_args': global_args,
        'optims': {
            name: opt.state_dict() for name, opt in net_opts.items()
        },
        'nets': {
            name: net.state_dict() for name, net in nets.items()
        }
    }
    save_model(state, epoch, '', model_file)


def _worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.worker_id = worker_id


def make_embedding(nets, obs, traj_len, global_args):
    batch_size, max_traj_len, ob_c, ob_h, ob_w = obs.size()

    # ob feature at time t is computed by taking ob_t and ob_{t-1}
    # batch_obs2 consists of ob_0, ob_0, ... ob_{T-1}
    # Note that batch_obs consists of ob_0, ob_1, ..., ob_T
    batch_obs2 = torch.cat([obs[:, 0].unsqueeze(1), obs[:, :-1]], dim=1)
    ob_features = nets['embedding_img_pair_encoder'](
        obs.view(batch_size * max_traj_len, ob_c, ob_h, ob_w),
        batch_obs2.view(batch_size * max_traj_len, ob_c, ob_h, ob_w))
    features = ob_features.view(batch_size, max_traj_len, -1)

    # features is of shape batch_size x max_traj_len x d
    hs, _ = nets['embedding_recurrent'](features.transpose(1, 0).contiguous())

    if 'embedding_bottleneck' in nets:
        embeddings = nets['embedding_bottleneck'](hs)
    else:
        embeddings = hs

    # obs_embeddings contains stepwise trajectory embedding
    obs_embeddings = embeddings.transpose(1, 0).contiguous()  # batch_size x max_traj_len x d

    # traj_embeddings contains only the final embedding
    traj_len = traj_len[:, None, None].expand(batch_size, 1, obs_embeddings.size(-1))
    traj_embeddings = torch.gather(obs_embeddings, 1, traj_len - 1).view(batch_size, -1)

    return {
        'obs_embedding': obs_embeddings,
        'traj_embedding': traj_embeddings,
    }


def train_simple(nets, net_opts, dataset, dagger_dataset_constructor, vis, global_args):
    (
        model_file,
        max_epochs,
        batch_size,
        n_worker,
        log_interval,
        vis_interval,
        save_interval,
        train_device,
        resume,
        model_variant,
    ) = [global_args[_] for _ in ['model_file',
                                  'max_epochs',
                                  'batch_size',
                                  'n_dataset_worker',
                                  'log_interval',
                                  'vis_interval',
                                  'save_interval',
                                  'train_device',
                                  'resume',
                                  'model_variant']]
    epoch = 0
    if resume:
        epoch = _load_weights(model_file, nets, net_opts)
        torch.manual_seed(231239 + epoch)
        print('loaded saved state. epoch: %d' % epoch)

    # FIXME: hack to mitigate the bug in torch 1.1.0's schedulers
    # TODO: should we remove this now?
    if epoch == 0:
        last_epoch = -1
    else:
        last_epoch = epoch

    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=global_args['lr_decay_epoch'],
            gamma=global_args['lr_decay_rate'],
            last_epoch=last_epoch)
        for name, opt in net_opts.items()
    }

    samples_per_epoch = global_args['samples_per_epoch']

    def train_helper(dataset, start_global_step, n_samples, dagger_training):
        sampler = RandomSampler(dataset, replacement=True, num_samples=n_samples)
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=n_worker, pin_memory=True, drop_last=True,
            worker_init_fn=_worker_init)
        last_log_time = time.time()

        n_batch = n_samples // batch_size
        batch_time = 0.0

        for idx, batch_data in enumerate(loader):
            for _, opt in net_opts.items():
                opt.zero_grad()

            batch_demo_traj_len = batch_data['demo_traj_len'].to(device=train_device, non_blocking=True)
            batch_demo_obs = batch_data['demo_obs'].to(device=train_device, non_blocking=True)
            batch_demo_start_ob = batch_data['demo_start_ob'].to(device=train_device, non_blocking=True)
            batch_demo_goal_ob = batch_data['demo_goal_ob'].to(device=train_device, non_blocking=True)
            batch_demo_mask = batch_data['demo_mask'].to(device=train_device, non_blocking=True)
            batch_rollout_obs = batch_data['rollout_obs'].to(device=train_device, non_blocking=True)

            _, max_traj_len, ob_c, ob_h, ob_w = batch_demo_obs.size()

            if idx % vis_interval == 0:
                vis.images(batch_data['demo_obs'][:8, :8].contiguous().view(-1, ob_c, ob_h, ob_w),
                           nrow=8,
                           win='batch_demo_obs', opts={'title': 'demo_obs'})
                vis.images(batch_data['rollout_obs'][:8, :8].contiguous().view(-1, ob_c, ob_h, ob_w),
                           nrow=8,
                           win='batch_rollout_obs', opts={'title': 'rollout_obs'})

            loss = 0

            batch_demo_start_ob = batch_demo_start_ob.unsqueeze(1)
            batch_demo_goal_ob = batch_demo_goal_ob.unsqueeze(1)

            # Train the embedding and the controller jointly.
            if dagger_training:
                demo_embedding = make_embedding(nets, batch_demo_obs, batch_demo_traj_len, global_args)

                rollout_traj_len = batch_data['rollout_traj_len'].to(device=train_device, non_blocking=True)
                rollout_mask = batch_data['rollout_mask'].to(device=train_device, non_blocking=True)
                rollout_progress = batch_data['rollout_progress'].to(device=train_device, non_blocking=True)
                rollout_waypoints = batch_data['rollout_waypoints'].to(device=train_device, non_blocking=True)

                rollout_embedding = make_embedding(nets, batch_rollout_obs, rollout_traj_len, global_args)

                rollout_stepwise_embeddings = rollout_embedding['obs_embedding']
                traj_embedding = demo_embedding['traj_embedding']

            else:
                # In behavior cloning mode, demo and rollout are the same except for the observations
                # (camera z could be different)
                demo_embedding = make_embedding(nets, batch_demo_obs, batch_demo_traj_len, global_args)
                rollout_embedding = make_embedding(nets, batch_rollout_obs, batch_demo_traj_len, global_args)

                batch_demo_progress = batch_data['demo_progress'].to(device=train_device, non_blocking=True)
                batch_demo_waypoints = batch_data['demo_waypoints'].to(device=train_device, non_blocking=True)

                rollout_progress = batch_demo_progress
                rollout_waypoints = batch_demo_waypoints
                rollout_mask = batch_demo_mask

                rollout_stepwise_embeddings = rollout_embedding['obs_embedding']
                traj_embedding = demo_embedding['traj_embedding']

            def make_attractor_feature(target):
                context_len = target.size(1)
                out = nets['img_encoder'](target.view(context_len * batch_size, ob_c, ob_h, ob_w))
                return out.view((batch_size, context_len) + out.size()[1:])

            def make_context_feature(ob_feature, target_feature):
                # Even though this says "context" feature, there is actually no context (context_len = 1).
                # I was playing around contextual features before and noticed that it was not necessary
                # anymore. The model works well without contextual features.
                if ob_feature.dim() == 4:
                    # ob_feature is of shape batch_size x d x s x s
                    # target_feature is of shape batch_size x context_len x d x s x s
                    ob_feature = ob_feature.unsqueeze(1).expand_as(target_feature)
                    si = target_feature.size()[2:]
                    pair_features = nets['feature_map_pair_encoder'](
                        ob_feature.reshape((-1,) + si), target_feature.reshape((-1,) + si))
                    context_len = target_feature.size(1)
                    pair_features = pair_features.view(batch_size, context_len, -1)
                    return nets['conv_encoder'](pair_features.transpose(1, 2))

                else:
                    # ob_feature is of size batch_size x ob_len x s x s x d
                    # target_feature is of size batch_size x context_len x s x s x d
                    ob_len = ob_feature.size(1)
                    context_len = target_feature.size(1)
                    si = target_feature.size()[2:]
                    ob_feature = ob_feature.unsqueeze(2).expand(-1, -1, context_len, -1, -1, -1)
                    target_feature = target_feature.unsqueeze(1).expand(-1, ob_len, -1, -1, -1, -1)
                    pair_features = nets['feature_map_pair_encoder'](
                        ob_feature.reshape((-1,) + si), target_feature.reshape((-1,) + si))
                    pair_features = pair_features.view(batch_size * ob_len, context_len, -1)
                    conv_features = nets['conv_encoder'](pair_features.transpose(1, 2))
                    return conv_features.view(batch_size, ob_len, -1)

            if global_args['attractor']:
                rollout_ob_features = nets['img_encoder'](
                    batch_rollout_obs.view(batch_size * max_traj_len, ob_c, ob_h, ob_w))

                rollout_ob_features = rollout_ob_features.view(
                    (batch_size, max_traj_len) + rollout_ob_features.size()[1:])

                demo_start_features = make_attractor_feature(batch_demo_start_ob)
                demo_goal_features = make_attractor_feature(batch_demo_goal_ob)

                init_start_features = make_context_feature(rollout_ob_features[:, 0], demo_start_features)
                ob_goal_features = make_context_feature(rollout_ob_features, demo_goal_features)

                if global_args['no_embedding']:
                    # This is for ablation study.
                    features = torch.cat([
                        init_start_features.unsqueeze(1).expand(-1, max_traj_len, -1),
                        ob_goal_features,
                    ], dim=-1)
                else:
                    features = torch.cat([
                        init_start_features.unsqueeze(1).expand(-1, max_traj_len, -1),
                        ob_goal_features,
                        rollout_stepwise_embeddings,
                        traj_embedding.unsqueeze(1).expand(-1, max_traj_len, -1),
                    ], dim=-1)

            else:
                # This is for ablation study.
                features = torch.cat([
                    rollout_stepwise_embeddings,
                    traj_embedding.unsqueeze(1).expand(-1, max_traj_len, -1),
                ], dim=-1)

            hs, _ = nets['recurrent'](features.transpose(1, 0).contiguous())
            pred_progress = nets['progress_regressor'](hs).transpose(1, 0).squeeze(-1)

            progress_loss_type = global_args['progress_loss']
            if progress_loss_type == 'l1':
                progress_loss = torch.abs(pred_progress - rollout_progress)
            else:
                raise ValueError('Unsupported progress loss type:', progress_loss_type)

            assert progress_loss.size() == rollout_mask.size()
            progress_loss = progress_loss * rollout_mask
            # Note that this is the mean of every data element, not per-batch mean
            progress_loss = torch.mean(progress_loss)

            loss = loss + progress_loss

            pred_waypoints = nets['waypoint_regressor'](hs).transpose(1, 0)
            # batch_size x max_traj_len x 2
            waypoint_loss = torch.norm(pred_waypoints - rollout_waypoints, p=2, dim=-1)
            waypoint_loss = torch.mean(waypoint_loss * rollout_mask)
            loss = loss + torch.mean(waypoint_loss)

            if dagger_training:
                if global_args['attractor']:
                    batch_heading_diff = batch_data['heading_diff'].to(device=train_device, non_blocking=True)
                    pred_heading_diff = nets['heading_diff_regressor'](init_start_features)
                    heading_diff_loss = torch.mean(torch.abs(pred_heading_diff - batch_heading_diff))
                    loss = loss + heading_diff_loss

            loss.backward()

            for _, opt in net_opts.items():
                opt.step()

            if idx % log_interval == 0:
                batch_time = 0.5 * batch_time + 0.5 * (time.time() - last_log_time) / log_interval
                print('epoch %d batch time %.2f sec remaining time %.2f sec loss: %6.2f' % (
                    epoch, batch_time, (n_batch - idx) * batch_time, loss.item()))
                print('learning rate:\n%s' % tabulate.tabulate([
                    (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))

                for name, net in nets.items():
                    print('%s weights:\n%s' % (name, module_weights_stats(net)))

                for name, net in nets.items():
                    print('%s grad:\n%s' % (name, module_grad_stats(net)))

                print('progress:\n%s' % tabulate.tabulate([
                    ['pred'] + ['%.2f' % _ for _ in pred_progress[0].tolist()],
                    ['gt'] + ['%.2f' % _ for _ in rollout_progress[0].tolist()]
                ]))

                global_step = start_global_step + idx * batch_size

                vis.line(X=np.array([global_step]),
                         Y=np.array([waypoint_loss.item()]),
                         win='waypoint_loss', update='append', opts={'title': 'waypoint_loss'})

                vis.line(X=np.array([global_step]),
                         Y=np.array([progress_loss.item()]),
                         win='progress_loss', update='append', opts={'title': 'progress_loss'})

                vis.line(X=np.array([global_step]),
                         Y=np.array([loss.item()]),
                         win='loss', update='append', opts={'title': 'loss'})

                if dagger_training:
                    vis.line(X=np.array([global_step]),
                             Y=np.array([np.mean(batch_data['reachability'].cpu().numpy())]),
                             win='fraction reached', update='append', opts={'title': 'fraction reached'})

                    if global_args['attractor']:
                            vis.line(X=np.array([global_step]),
                                     Y=np.array([heading_diff_loss.item()]),
                                     win='heading_diff', update='append', opts={'title': 'heading_diff_loss'})

                last_log_time = time.time()
                vis.save([vis.env])

    while True:
        print('===== epoch %d =====' % epoch)

        dagger_start_epoch = global_args['dagger_epoch']
        if dagger_start_epoch <= epoch and dagger_dataset_constructor is not None:
            n_dagger_samples = global_args['dagger_init'] + (epoch - dagger_start_epoch) * global_args['dagger_inc']
            n_dagger_samples = min(n_dagger_samples, samples_per_epoch)
            print('dagger training %d samples' % n_dagger_samples)
            dagger_dataset = dagger_dataset_constructor(epoch)
            train_helper(dagger_dataset, epoch * samples_per_epoch, n_dagger_samples, True)
            del dagger_dataset
        else:
            n_dagger_samples = 0

        n_samples = samples_per_epoch - n_dagger_samples
        if n_samples > 0:
            print('normal training %d samples' % n_samples)
            train_helper(dataset, epoch * samples_per_epoch + n_dagger_samples, n_samples, False)
        else:
            print('skip normal training because dagger has used up all samples.')

        for _, sched in net_scheds.items():
            sched.step()

        epoch += 1
        if epoch > max_epochs:
            break

        if epoch % save_interval == 0:
            print('saving model...')
            _save_model(nets, net_opts, epoch, global_args, model_file)
