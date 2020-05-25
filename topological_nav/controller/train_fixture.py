import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits, kl_div, cross_entropy, relu
from torch.distributions import Categorical
from torch.utils.data import DataLoader, WeightedRandomSampler
from rmp_nav.common.utils import save_model, load_model, module_grad_stats
import tabulate
import os


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


def train_multiframedst(nets, net_opts, dataset, vis, global_args):
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
        weight_loss,
        weight_loss_min_clip,
        model_variant,
        proximity_label,
        heading_diff_label
    ) = [global_args[_] for _ in ['model_file',
                                  'max_epochs',
                                  'batch_size',
                                  'n_dataset_worker',
                                  'log_interval',
                                  'vis_interval',
                                  'save_interval',
                                  'train_device',
                                  'resume',
                                  'weight_loss',
                                  'weight_loss_min_clip',
                                  'model_variant',
                                  'proximity_label',
                                  'heading_diff_label']]

    epoch = 0
    if resume:
        epoch = _load_weights(model_file, nets, net_opts)
        torch.manual_seed(231239 + epoch)
        print('loaded saved state. epoch: %d' % epoch)

    # FIXME: hack to mitigate the bug in torch 1.1.0's schedulers
    if epoch <= 1:
        last_epoch = epoch - 1
    else:
        last_epoch = epoch - 2

    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=global_args['lr_decay_epoch'],
            gamma=global_args['lr_decay_rate'],
            last_epoch=last_epoch)
        for name, opt in net_opts.items()
    }

    n_samples = global_args['samples_per_epoch']

    while True:
        print('===== epoch %d =====' % epoch)

        sampler = WeightedRandomSampler([1.0] * len(dataset), n_samples)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=n_worker,
                            pin_memory=True,
                            drop_last=True)

        last_log_time = time.time()

        for idx, (batch_src_imgs, batch_dst_imgs, batch_waypoints, batch_extras) in enumerate(loader):
            for _, opt in net_opts.items():
                opt.zero_grad()

            if idx % vis_interval == 0:
                imgs = []
                for i in range(3):
                    src_img = batch_src_imgs[i].data.numpy()
                    dst_imgs = batch_dst_imgs[i].data.numpy()
                    imgs.append(src_img[None])
                    imgs.append(dst_imgs)
                imgs = np.concatenate(imgs, axis=0)
                vis.images(imgs, nrow=(dst_imgs.shape[0] + 1),
                           win='batch_imgs', opts={'title': 'src-dst'})

            batch_src_imgs = batch_src_imgs.to(device=train_device, non_blocking=True)
            batch_dst_imgs = batch_dst_imgs.to(device=train_device, non_blocking=True)
            batch_waypoints = batch_waypoints.to(device=train_device, non_blocking=True)

            for k, v in batch_extras.items():
                batch_extras[k] = v.to(device=train_device, non_blocking=True)

            batch_size, win_size, c, h, w = batch_dst_imgs.size()

            if model_variant == 'attention':
                src_features = nets['img_encoder'](batch_src_imgs)
                dst_features = nets['img_encoder'](batch_dst_imgs.view(-1, c, h, w)).view(
                    batch_size, win_size, -1)  # batch_size x win_size x dim

                # FIXME: disabled attention temporarily
                # dst_terminal_features = dst_features[:, -1, :]
                # attention = nets['attention_encoder'](torch.cat([src_features,
                #                                                  dst_terminal_features], dim=1))
                dst_temporal_features = nets['seq_encoder'](dst_features)
                final_features = torch.cat([src_features, dst_temporal_features], dim=1)
                pred_waypoints = nets['wp_regressor'](final_features)

            elif model_variant == 'concat_early':
                src_features = nets['img_encoder'](batch_src_imgs)
                dst_features = nets['img_encoder'](batch_dst_imgs.view(-1, c, h, w)).view(
                    batch_size, win_size, -1)  # batch_size x win_size x dim

                src_dst_features = torch.cat([src_features.unsqueeze(1).expand_as(dst_features),
                                              dst_features], dim=-1)
                temporal_features = nets['seq_encoder'](src_dst_features)
                pred_waypoints = nets['wp_regressor'](temporal_features)

            elif model_variant == 'future':
                src_features = nets['img_encoder'](batch_src_imgs)
                dst_features = nets['img_encoder'](batch_dst_imgs.view(-1, c, h, w)).view(
                    batch_size, win_size, -1)  # batch_size x win_size x dim

                win_size = dst_features.size(1) // 2

                past_features = dst_features[:, :win_size + 1]
                future_features = dst_features[:, win_size:]

                past_temporal_features = nets['seq_encoder'](past_features)
                future_temporal_features = nets['seq_encoder'](future_features)

                final_features = torch.cat([src_features,
                                            past_temporal_features,
                                            future_temporal_features], dim=1)
                pred_waypoints = nets['wp_regressor'](final_features)

            elif model_variant == 'future_stack':
                img_stack = torch.cat([batch_src_imgs.unsqueeze(1), batch_dst_imgs], dim=1)
                features = nets['stack_encoder'](img_stack)
                pred_waypoints = nets['wp_regressor'](features)

            elif model_variant == 'future_stack_v2':
                # Only stack dst images.
                src_features = nets['img_encoder'](batch_src_imgs)
                dst_features = nets['stack_encoder'](batch_dst_imgs)
                features = torch.cat([src_features, dst_features], dim=-1)
                pred_waypoints = nets['wp_regressor'](features)

            elif model_variant == 'future_pair':
                batch_src_imgs2 = batch_src_imgs.unsqueeze(1).expand_as(batch_dst_imgs).contiguous()
                pair_features = nets['img_pair_encoder'](
                    batch_src_imgs2.view(batch_size * win_size, c, h, w),
                    batch_dst_imgs.view(batch_size * win_size, c, h, w)).view(batch_size, -1)
                pred_waypoints = nets['wp_regressor'](pair_features)
                if proximity_label:
                    pred_proximity = nets['proximity_regressor'](pair_features)
                if heading_diff_label:
                    pred_heading_diff = nets['heading_diff_regressor'](pair_features)

            elif model_variant == 'future_pair_conv':
                batch_src_imgs2 = batch_src_imgs.unsqueeze(1).expand_as(batch_dst_imgs).contiguous()
                pair_features = nets['img_pair_encoder'](
                    batch_src_imgs2.view(batch_size * win_size, c, h, w),
                    batch_dst_imgs.view(batch_size * win_size, c, h, w)).view(batch_size, win_size, -1)
                conv_feature = nets['conv_encoder'](pair_features.transpose(1, 2))
                pred_waypoints = nets['wp_regressor'](conv_feature)
                if proximity_label:
                    pred_proximity = nets['proximity_regressor'](conv_feature)
                if heading_diff_label:
                    pred_heading_diff = nets['heading_diff_regressor'](conv_feature)

            elif model_variant == 'future_pair_featurized':
                src_features = nets['img_encoder'](batch_src_imgs)
                dst_features = nets['img_encoder'](batch_dst_imgs.view(
                    batch_size * win_size, c, h, w)).view(batch_size, win_size, -1)
                src_features = src_features.unsqueeze(1).expand_as(dst_features).contiguous()
                pair_features = nets['feature_pair_encoder'](
                    src_features.view(batch_size * win_size, -1),
                    dst_features.view(batch_size * win_size, -1)).view(batch_size, -1)
                pred_waypoints = nets['wp_regressor'](pair_features)

            elif model_variant == 'future_pair_featurized_v2':
                src_features = nets['src_img_encoder'](batch_src_imgs)
                dst_features = nets['dst_img_encoder'](batch_dst_imgs.view(
                    batch_size * win_size, c, h, w)).view(batch_size, win_size, -1)
                src_features = src_features.unsqueeze(1).expand_as(dst_features).contiguous()
                pair_features = nets['feature_pair_encoder'](
                    src_features.view(batch_size * win_size, -1),
                    dst_features.view(batch_size * win_size, -1)).view(batch_size, -1)
                pred_waypoints = nets['wp_regressor'](pair_features)

            elif model_variant == 'raw_control':
                batch_src_imgs2 = batch_src_imgs.unsqueeze(1).expand_as(batch_dst_imgs).contiguous()
                pair_features = nets['img_pair_encoder'](
                    batch_src_imgs2.view(batch_size * win_size, c, h, w),
                    batch_dst_imgs.view(batch_size * win_size, c, h, w)).view(batch_size, win_size, -1)
                conv_feature = nets['conv_encoder'](pair_features.transpose(1, 2))

                velocity = batch_extras['velocity'].to(device=train_device, non_blocking=True)
                angular_vel = batch_extras['angular_vel'].to(device=train_device, non_blocking=True)

                all_features = torch.cat([conv_feature, velocity, angular_vel], dim=-1)

                # Note that pred_waypoints here are actually raw controls.
                pred_waypoints = nets['wp_regressor'](all_features)

                if proximity_label:
                    pred_proximity = nets['proximity_regressor'](conv_feature)

                if heading_diff_label:
                    pred_heading_diff = nets['heading_diff_regressor'](conv_feature)

            else:
                raise RuntimeError('Unknown model variant %s' % model_variant)

            l2_loss = torch.sum(torch.pow(pred_waypoints - batch_waypoints, 2), dim=1)
            if weight_loss:
                l2_loss *= 1.0 / torch.max(batch_waypoints.norm(p=2, dim=1),
                                           batch_waypoints.new_tensor(weight_loss_min_clip))
            loss = torch.mean(l2_loss)
            if proximity_label:
                assert pred_proximity.size() == batch_extras['proximity'].size()
                proximity_loss = binary_cross_entropy_with_logits(pred_proximity,
                                                                  batch_extras['proximity'])
                loss += proximity_loss

            if heading_diff_label:
                assert pred_heading_diff.size() == batch_extras['heading_diff'].size()
                heading_diff_loss = torch.mean(torch.sum(torch.pow(
                    pred_heading_diff - batch_extras['heading_diff'], 2), dim=1))
                loss += heading_diff_loss

            loss.backward()

            for _, opt in net_opts.items():
                opt.step()

            if idx % log_interval == 0:
                print('epoch %d batch time %.2f sec loss: %6.2f' % (
                    epoch, (time.time() - last_log_time) / log_interval, loss.item()))
                print('learning rate:\n%s' % tabulate.tabulate([
                    (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))
                for name, net in nets.items():
                    print('%s grad:\n%s' % (name, module_grad_stats(net)))

                vis.line(X=np.array([epoch * n_samples + idx * batch_size]),
                         Y=np.array([loss.item()]),
                         win='loss', update='append', opts={'title': 'loss'})

                if proximity_label:
                    def format(l):
                        return '(' + ','.join(['%.2f' % _ for _ in l]) + ')'
                    print('proximity:\n%s' % tabulate.tabulate([
                        ['pred'] + [format(_) for _ in torch.sigmoid(pred_proximity[:10]).tolist()],
                        ['gt'] + [format(_) for _ in batch_extras['proximity'][:10].tolist()]
                    ]))
                    vis.line(X=np.array([epoch * n_samples + idx * batch_size]),
                             Y=np.array([proximity_loss.item()]),
                             win='proximity loss', update='append',
                             opts={'title': 'proximity loss'})

                if heading_diff_label:
                    def format(l):
                        return '(' + ','.join(['%.2f' % _ for _ in l]) + ')'
                    print('heading_diff:\n%s' % tabulate.tabulate([
                        ['pred'] + [format(_) for _ in pred_heading_diff[:10].tolist()],
                        ['gt'] + [format(_) for _ in batch_extras['heading_diff'][:10].tolist()]
                    ]))
                    vis.line(X=np.array([epoch * n_samples + idx * batch_size]),
                             Y=np.array([heading_diff_loss.item()]),
                             win='heading_diff loss', update='append',
                             opts={'title': 'heading diff loss'})

                last_log_time = time.time()
                vis.save([vis.env])

        for _, sched in net_scheds.items():
            sched.step()

        epoch += 1
        if epoch > max_epochs:
            break

        if epoch % save_interval == 0:
            print('saving model...')
            _save_model(nets, net_opts, epoch, global_args, model_file)
