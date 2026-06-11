from .mainlog import *
from .ppo import *
from .storage import *
from .sharestorage import *
from .subp import *
import math

from environment.gen_peo_pos_npy import ensure_peo_pos_npy
from .conf import comm_spectral_k_bounds
from .pred import (
    PredModule,
    compute_peo_region_id,
    compute_region_stats,
    pretrain_pred_module,
    compute_pred_feat_timeseries,
    _pred_pretrain_lr_for_epoch,
)

PRED_FEAT_TIMESERIES_NAME = 'pred_feat_timeseries.npy'


def _save_comm(mainlog, comm, best=False):
    prefix = '' if best else 'tmp_'
    path = mainlog._log_path + '/' + prefix + 'comm.pth'
    torch.save({'state_dict': comm.state_dict(),
                'groups': comm.get_groups()}, path)


def _save_pred(mainlog, pred_module, best=False):
    prefix = '' if best else 'tmp_'
    path = mainlog._log_path + '/' + prefix + 'pred.pth'
    torch.save(pred_module.state_dict(), path)


def main(ENV_CONF, Env):
    mp.set_start_method("spawn", force=True)
    start_datetime = datetime.now()
    CONF['uav_num'] = ENV_CONF['uav_num']
    _mink, _maxk = comm_spectral_k_bounds(ENV_CONF['uav_num'])
    CONF['comm_min_clusters'], CONF['comm_max_clusters'] = _mink, _maxk
    is_comm = CONF.get('method_name', '') == 'MAPPO_COMM'
    _method = CONF.get('method_name', 'MAPPO')
    _need_pred_assets = CONF.get('actor_use_pred', False) or (
        CONF.get('critic_use_pred', False) and _method in ('MAPPO', 'MAPPO_CNN'))
    pred_update_each_iter = CONF.get('pred_update_each_iter', True)
    pred_rl_cosine_lr_then_freeze = bool(CONF.get('pred_rl_cosine_lr_then_freeze', False))
    pred_rl_cosine_iters = int(CONF.get('pred_rl_cosine_iters', 5000))
    _pred_pretrain_final_lr = float(CONF.get('pred_lr_init', CONF.get('pred_lr', CONF['lr'])))

    mainlog = MainLog(ENV_CONF)
    mainlog.record_env_conf()
    mainlog.record_conf()

    ac_list = [Policy(_).to(CONF['device']) for _ in range(ENV_CONF['uav_num'])]
    for ac in ac_list:
        ac.eval()

    # ── 预测模块：peo_pos.npy 预训练 → 可选每 iter 更新并重写 pred_feat 表 ───────
    pred_module = None
    _peo_region_id = None
    _region_stats = None
    _pred_ts_path = None
    if _need_pred_assets:
        K_cells = CONF.get('region_cell_num', 6)
        K = K_cells * K_cells
        P = ENV_CONF['peo_num']
        _peo_pos_path = ENV_CONF.get('peo_pos_path')
        if not _peo_pos_path:
            raise ValueError(
                "需要预测特征时（actor_use_pred 或 MAPPO/MAPPO_CNN 下 critic_use_pred）"
                "必须配置 ENV_CONF['peo_pos_path']（101 帧 peo_pos.npy），"
                '请运行 python environment/gen_peo_pos_npy.py 并在 conf 中配置 peo_pos_path')
        _peo_pos_path = ensure_peo_pos_npy(ENV_CONF)
        peo_pos = np.load(_peo_pos_path, allow_pickle=True)
        peo_pos = np.asarray(peo_pos, dtype=np.float32)
        need_t = ENV_CONF['max_step'] + 1
        if peo_pos.shape[0] < need_t or peo_pos.shape[1] != P:
            raise ValueError(
                f'peo_pos.npy 期望 shape 至少 ({need_t}, {P}, ·), 实际 {peo_pos.shape}')
        peo_pos = peo_pos[:need_t]

        _peo_region_id = compute_peo_region_id(
            peo_pos[:, :, :2], ENV_CONF['field_length'], K_cells)
        _region_stats = compute_region_stats(_peo_region_id, K)

        pred_module = PredModule(
            P=P, K=K,
            d=CONF.get('pred_d', 64),
            d_pred=CONF.get('pred_out_d', 64),
            L=CONF.get('pred_L', 5),
            T_max=ENV_CONF['max_step'] + 1,
            n_heads=CONF.get('pred_n_heads', 2),
        ).to(CONF['device'])

        print('[train] pred: pretraining on peo_pos.npy ...')
        pretrain_pred_module(
            pred_module, _peo_region_id, _region_stats,
            epochs=CONF.get('pred_pretrain_epochs', 200),
            device=CONF['device'],
        )
        pred_ts = compute_pred_feat_timeseries(
            pred_module, _peo_region_id, _region_stats,
            device=CONF['device'],
        )
        if pred_ts.shape != (ENV_CONF['max_step'], CONF.get('pred_out_d', 64)):
            raise RuntimeError(f'pred_feat timeseries shape {pred_ts.shape} unexpected')
        _pred_ts_path = os.path.join(mainlog._log_path, PRED_FEAT_TIMESERIES_NAME)
        np.save(_pred_ts_path, pred_ts)
        print('[train] pred: initial', _pred_ts_path, pred_ts.shape)

        pred_module.eval()
        _save_pred(mainlog, pred_module)
        _ep_pre = int(CONF.get('pred_pretrain_epochs', 200))
        _pred_pretrain_final_lr = _pred_pretrain_lr_for_epoch(
            _ep_pre - 1, _ep_pre,
            float(CONF.get('pred_lr_init', CONF.get('pred_lr', CONF['lr']))),
            float(CONF.get('pred_lr_late', 1e-5)),
            int(CONF.get('pred_pretrain_warmup_epochs', 3000)),
            bool(CONF.get('pred_pretrain_cosine_schedule', False)),
        )
        if not pred_update_each_iter:
            print('[train] pred: pred_update_each_iter=False，RL 阶段固定使用上述 pred_feat 与 tmp_pred.pth，不再每 iter 更新')
        elif pred_rl_cosine_lr_then_freeze:
            print('[train] pred: pred_rl_cosine_lr_then_freeze=True，RL 初 pred_lr=%.2e，前 %d iter 余弦至 %.2e 后冻结'
                  % (_pred_pretrain_final_lr, pred_rl_cosine_iters,
                     float(CONF.get('pred_rl_lr_floor', 1e-5))))

    comm_module = None
    if is_comm:
        z_dim = int(torch.tensor(CONF['M_size']).prod().item())
        _mx = CONF.get('comm_max_group_size', 0)
        comm_module = CommModule(
            n_agents=ENV_CONF['uav_num'], z_dim=z_dim,
            hgcn_hidden_dim=CONF['hgcn_hidden_dim'],
            hgcn_out_dim=CONF['hgcn_out_dim'],
            hgcn_num_layers=CONF['hgcn_num_layers'],
            min_clusters=CONF['comm_min_clusters'],
            max_clusters=CONF['comm_max_clusters'],
            stability_threshold=CONF['comm_stability_threshold'],
            clustering_interval=CONF['comm_clustering_interval'],
            min_group_size=CONF.get('comm_min_group_size', 1),
            max_group_size=(None if _mx is None or _mx <= 0 else int(_mx)),
            hgcn_mha_num_heads=CONF.get('hgcn_MHA_num_heads', 2),
            node2edge_weight=CONF.get('hgcn_node2edge_weight', False),
        ).to(CONF['device'])
        comm_module.eval()
        _pred_for_ppo = pred_module if (_need_pred_assets and pred_update_each_iter) else None
        agent = PPO(
            ac_list=ac_list, comm_module=comm_module,
            pred_module=_pred_for_ppo)
        _save_comm(mainlog, comm_module)
    else:
        _pred_for_ppo = pred_module if (_need_pred_assets and pred_update_each_iter) else None
        agent = PPO(ac_list=ac_list, pred_module=_pred_for_ppo)

    for uid in range(ENV_CONF['uav_num']):
        mainlog.save_cur_model(ac_list[uid], uid)

    rollouts = RolloutStorage(ENV_CONF)
    shared_rollout_list = []
    for env_id in range(CONF['env_num']):
        shared_rollout_list.append(ShareRolloutStorage(ENV_CONF))

    cur_datetime = datetime.now()
    print('process time:', cur_datetime - start_datetime)
    shared_ifdone_list = [mp.Value('b', False) for _ in range(CONF['env_num'])]
    init_poi_value_s = Env.gen_whole_init_poi_value()
    processes = []
    for env_id in range(CONF['env_num']):
        p = mp.Process(target=subp,
                       args=(env_id,
                             mainlog._log_path,
                             shared_rollout_list[env_id],
                             shared_ifdone_list[env_id],
                             init_poi_value_s,
                             ENV_CONF, Env,
                             ))
        processes.append(p)
        p.start()

    eff_list = []
    max_avg_eff = 0
    max_avg_eff_iter = 0
    traj_last1k_best_max = {'best': float('-inf')}
    for iter_id in range(CONF['train_iter']):
        iter_start_datetime = datetime.now()
        while True:
            global_ifdone = 0
            for shared_ifdone in shared_ifdone_list:
                if shared_ifdone.value:
                    global_ifdone += 1
                else:
                    break
            if global_ifdone < CONF['env_num']:
                time.sleep(0.01)
                continue
            if True:
                for env_id in range(CONF['env_num']):
                    rollouts.insert(shared_rollout_list[env_id], env_id)

                mainlog.load_envs_info()
                mainlog.maybe_save_traj_best_episode(
                    iter_id, mainlog.envs_info['episode_total_reward'][iter_id], traj_last1k_best_max)
                mean_eff = np.mean(mainlog.envs_info['eff'][iter_id])
                eff_list.append(mean_eff)
                avg_eff = np.mean(eff_list[-50:])
                if avg_eff > max_avg_eff:
                    max_avg_eff = avg_eff
                    max_avg_eff_iter = iter_id
                    for uid in range(ENV_CONF['uav_num']):
                        mainlog.save_model(ac_list[uid], uid)
                    if is_comm:
                        _save_comm(mainlog, comm_module, best=True)
                    if pred_module is not None:
                        _save_pred(mainlog, pred_module, best=True)
                for ac in ac_list:
                    ac.train()
                if is_comm:
                    comm_module.train()

                value_loss_per_sample, action_loss_per_sample, \
                    dist_entropy_per_sample, loss_per_sample = agent.update(rollouts, iter_id)

                # ── 预测网络：可选每 iter 更新并重写 pred_feat 时隙表 ─────────────────
                _do_pred_iter = (
                    pred_module is not None and _peo_region_id is not None
                    and pred_update_each_iter)
                if _do_pred_iter and pred_rl_cosine_lr_then_freeze:
                    if iter_id >= pred_rl_cosine_iters:
                        _do_pred_iter = False
                    elif agent.pred_optimizer is not None:
                        _floor = float(CONF.get('pred_rl_lr_floor', 1e-5))
                        _n = pred_rl_cosine_iters
                        if _n <= 1:
                            _cur_pred_lr = _pred_pretrain_final_lr
                        else:
                            _t = iter_id / float(_n - 1)
                            _cur_pred_lr = _floor + (_pred_pretrain_final_lr - _floor) * 0.5 * (
                                1.0 + math.cos(math.pi * _t))
                        for _g in agent.pred_optimizer.param_groups:
                            _g['lr'] = _cur_pred_lr
                if _do_pred_iter:
                    _pl = agent.update_pred(_peo_region_id, _region_stats)
                    pred_ts = compute_pred_feat_timeseries(
                        pred_module, _peo_region_id, _region_stats,
                        device=CONF['device'],
                    )
                    np.save(_pred_ts_path, pred_ts)
                    _save_pred(mainlog, pred_module)
                    if iter_id % 100 == 0 and _pl:
                        _pd = {k: round(v, 4) for k, v in _pl.items()}
                        if pred_rl_cosine_lr_then_freeze and agent.pred_optimizer is not None:
                            print('pred_iter_loss:', _pd,
                                  'pred_lr=%.2e' % agent.pred_optimizer.param_groups[0]['lr'])
                        else:
                            print('pred_iter_loss:', _pd)

                for ac in ac_list:
                    ac.eval()
                if is_comm:
                    comm_module.eval()

                mainlog.record_metrics_result(iter_id)
                # train_iter=10000 → iter_id 仅 0..9999（共 10000 次更新）；%1000 得到 0,1000,..,9000，最后一轮需单独画
                if iter_id % 1000 == 0 or iter_id == CONF['train_iter'] - 1:
                    mainlog.record_reward_train_fig(iter_id)
                mainlog.record_loss(value_loss_per_sample, action_loss_per_sample,
                                    dist_entropy_per_sample, loss_per_sample)
                for uid in range(ENV_CONF['uav_num']):
                    mainlog.save_cur_model(ac_list[uid], uid)
                if is_comm:
                    _save_comm(mainlog, comm_module)
                mean_f = np.mean(mainlog.envs_info['f'][iter_id])
                mean_dcr = np.mean(mainlog.envs_info['dcr'][iter_id])
                mean_ec = np.mean(mainlog.envs_info['ec'][iter_id])
                mean_mec = np.mean(mainlog.envs_info['mec'][iter_id])
                mean_cec = np.mean(mainlog.envs_info['cec'][iter_id])
                mean_hit = np.mean(mainlog.envs_info['hit'][iter_id])
                mean_co = np.mean(mainlog.envs_info['co'][iter_id])
                mean_ecr = np.mean(mainlog.envs_info['ecr'][iter_id])
                mean_charge = np.mean(mainlog.envs_info['charge'][iter_id])
                mean_episode_total_reward = np.mean(mainlog.envs_info['episode_total_reward'][iter_id])
                report_str = ('iter: ' + str(iter_id)
                             + ' max_avg_eff: ' + str(np.round(max_avg_eff, 5))
                             + ' max_avg_eff_iter: ' + str(max_avg_eff_iter)
                             + ' eff: ' + str(np.round(mean_eff, 5))
                             + ' f: ' + str(np.round(mean_f, 5))
                             + ' dcr: ' + str(np.round(mean_dcr, 5))
                             + ' ec: ' + str(np.round(mean_ec, 5))
                             + '\n'
                             + ' mec: ' + str(np.round(mean_mec, 5))
                             + ' cec: ' + str(np.round(mean_cec, 5))
                             + ' hit: ' + str(np.round(mean_hit, 5))
                             + ' co: ' + str(np.round(mean_co, 5))
                             + ' ecr: ' + str(np.round(mean_ecr, 5))
                             + ' charge: ' + str(np.round(mean_charge, 5))
                             + ' episode_total_reward: ' + str(np.round(mean_episode_total_reward, 5)))
                mainlog.record_report(report_str)
                print(report_str)
                print('pid', os.getpid(), CONF['device'],
                      '\'' + ENV_CONF['env_name'] + '/' + CONF[
                          'method_name'] + '\': \'' + mainlog.get_start_time() + '\',')
                cur_datetime = datetime.now()
                print('process time:', cur_datetime - start_datetime, 'iter duration:',
                      cur_datetime - iter_start_datetime, '\n')
                for shared_ifdone in shared_ifdone_list:
                    shared_ifdone.value = False
                break

    for p in processes:
        p.join()
