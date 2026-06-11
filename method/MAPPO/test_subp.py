from .model import *
from .sublog import *
from .conf import comm_spectral_k_bounds, get_comm_alpha


@jit(nopython=True)
def fast_clip(a, min_v, max_v):
    l = a.shape[0]
    for i in range(l):
        if a[i] < min_v:
            a[i] = min_v
        elif a[i] > max_v:
            a[i] = max_v

    return a


@jit(nopython=True)
def fast_clip_val(a, min_v, max_v):
    if a < min_v:
        a = min_v
    elif a > max_v:
        a = max_v
    return a


@jit(fastmath=True)
def fast_int(a):
    a = np.round(a)
    a = int(a)
    return a


@jit(nopython=True)
def resize_img(scrH, scrW, dstH, dstW, obs, retimg):
    for i in range(dstH):
        for j in range(dstW):
            scrx = fast_int((i + 1) * (scrH * 1.0 / dstH))
            scry = fast_int((j + 1) * (scrW * 1.0 / dstW))
            new_x = fast_clip_val(scrx - 1, 0, obs.shape[2] - 1)
            new_y = fast_clip_val(scry - 1, 0, obs.shape[3] - 1)

            retimg[..., i, j] = obs[..., new_x, new_y]

    return retimg


@jit(nopython=True, )
def gen_obs(hr_obs, obs, ux, uy, blk_hr_grid_range, hr_shape, uav_num,
            cur_poi2hr_grid_range_channel, obs_shape, px, py, retimg,
            uav_types, org_ux, org_uy, uid_pos, field_length, M_size,
            p_msk, n_msk, mtx_size, uav_e):
    ux = fast_clip(ux, 0, hr_shape[0] - 1)
    uy = fast_clip(uy, 0, hr_shape[1] - 1)
    px = fast_clip(px, 0, hr_shape[0] - 1)
    py = fast_clip(py, 0, hr_shape[0] - 1)

    for uid in range(uav_num):
        hr_obs[uid, :-3] = cur_poi2hr_grid_range_channel
        for i in range(blk_hr_grid_range[0].shape[0]):
            hr_obs[uid, -1, blk_hr_grid_range[0][i], blk_hr_grid_range[1][i]] = -1

        hr_obs[uid, -2, ux[uid], uy[uid]] = uav_e[uid]

    for i in range(len(px)):
        hr_obs[:, -3, px[i], py[i]] = 1

    half_shape_x = int(obs.shape[2] / 2)
    half_shape_y = int(obs.shape[3] / 2)

    hr_x_min = ux - half_shape_x
    hr_y_min = uy - half_shape_y
    hr_x_max = ux - half_shape_x + obs.shape[2]
    hr_y_max = uy - half_shape_y + obs.shape[2]

    hr_x_min = fast_clip(hr_x_min, 0, hr_shape[0] - 1)
    hr_y_min = fast_clip(hr_y_min, 0, hr_shape[1] - 1)
    hr_x_max = fast_clip(hr_x_max, 1, hr_shape[0])
    hr_y_max = fast_clip(hr_y_max, 1, hr_shape[1])

    x_min = half_shape_x - ux
    y_min = half_shape_y - uy
    x_min = fast_clip(x_min, 0, obs.shape[2] - 1)
    y_min = fast_clip(y_min, 0, obs.shape[3] - 1)
    x_max = x_min + hr_x_max - hr_x_min
    y_max = y_min + hr_y_max - hr_y_min
    x_max = fast_clip(x_max, 1, obs.shape[2])
    y_max = fast_clip(y_max, 1, obs.shape[3])

    for uid in range(uav_num):
        obs[uid, 0,
        x_min[uid]:x_max[uid], y_min[uid]:y_max[uid]] = hr_obs[uid, uav_types[uid],
                                                        hr_x_min[uid]:hr_x_max[uid], hr_y_min[uid]:hr_y_max[uid]]
        obs[uid, 1:,
        x_min[uid]:x_max[uid], y_min[uid]:y_max[uid]] = hr_obs[uid, -3:,
                                                        hr_x_min[uid]:hr_x_max[uid], hr_y_min[uid]:hr_y_max[uid]]

    w_size = mtx_size // 2
    for i in range(uav_num):
        uid_pos[i, 0] = fast_clip_val(int(org_ux[i] / field_length[0] * (M_size[1] - w_size * 2)) + 1, 1, M_size[1] - 2)
        uid_pos[i, 1] = fast_clip_val(int(org_uy[i] / field_length[1] * (M_size[2] - w_size * 2)) + 1, 1, M_size[2] - 2)

    for i in range(uav_num):
        c = 0
        for x in range(uid_pos[i, 0] - w_size, uid_pos[i, 0] + w_size + 1):
            for y in range(uid_pos[i, 1] - w_size, uid_pos[i, 1] + w_size + 1):
                p_msk[i, 0, c, :, x, y] = 1
                c += 1
    for i in range(uav_num):
        n_msk[i, :, uid_pos[i, 0] - w_size:uid_pos[i, 0] + w_size + 1,
        uid_pos[i, 1] - w_size:uid_pos[i, 1] + w_size + 1] = 0

    if obs_shape[1] == obs.shape[2]:
        return obs, uid_pos, p_msk, n_msk

    scrH, scrW = obs.shape[2], obs.shape[3]
    dstH, dstW = obs_shape[1], obs_shape[2]

    obs_new = resize_img(scrH, scrW, dstH, dstW, obs, retimg)

    return obs_new, uid_pos, p_msk, n_msk


def process_action(action_list):
    action_s = np.zeros([len(action_list), 2], dtype=np.float32)
    for i, action in enumerate(action_list):
        action_s[i, :] = action[0].numpy()
    return action_s


def test_subp(process_id,
              log_root_path,
              shared_rollout,
              shared_ifdone,
              init_poi_value_s,
              ENV_CONF, Env,
              ):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    torch.set_num_threads(1)

    is_comm = CONF.get('method_name', '') == 'MAPPO_COMM'
    with torch.no_grad():
        np.random.seed(seed + process_id)
        torch.manual_seed(seed + process_id)
        sub_iter_counter = 0
        print("---------------------------->", process_id, "subp")
        test_log_path = os.path.join(log_root_path, 'test')
        sublog = SubLog(ENV_CONF, log_path=test_log_path, process_id=process_id)
        CONF['uav_num'] = ENV_CONF['uav_num']
        _mink, _maxk = comm_spectral_k_bounds(ENV_CONF['uav_num'])
        CONF['comm_min_clusters'], CONF['comm_max_clusters'] = _mink, _maxk
        local_ac_list = [Policy(uid) for uid in range(ENV_CONF['uav_num'])]

        for uavid in range(ENV_CONF['uav_num']):
            local_ac_list[uavid].eval()

        local_comm = None
        comm_group_labels_vol = None
        comm_group_npy_path = None
        if is_comm:
            z_dim = int(torch.tensor(CONF['M_size']).prod().item())
            _mx = CONF.get('comm_max_group_size', 0)
            local_comm = CommModule(
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
                node2edge_weight=CONF.get('hgcn_node2edge_weight', False))
            local_comm.eval()
            proc_dir = os.path.join(test_log_path, 'process_' + str(process_id))
            os.makedirs(proc_dir, exist_ok=True)
            comm_group_labels_vol = np.zeros(
                (ENV_CONF['uav_num'], ENV_CONF['max_step'], CONF['train_iter']),
                dtype=np.int32)
            comm_group_npy_path = os.path.join(proc_dir, 'comm_group_labels.npy')

        env = Env(CONF['obs_shape'][1:], CONF['hr_shape'])
        while sub_iter_counter < 1:
            while True:
                if shared_ifdone.value:
                    time.sleep(0.01)
                    continue
                if True:
                    # sync shared model to local
                    for uavid in range(ENV_CONF['uav_num']):
                        local_ac_list[uavid].load_state_dict(
                            torch.load(log_root_path + '/model_' + str(uavid) + '.pth',
                                        map_location=torch.device('cpu')))
                    if is_comm:
                        comm_path = log_root_path + '/comm.pth'
                        if os.path.exists(comm_path):
                            ckpt = torch.load(comm_path, map_location='cpu')
                            local_comm.load_state_dict(ckpt['state_dict'])
                            local_comm.set_groups(ckpt['groups'])
                    ################################## feed in sharestorage ####################################
                    st = env.reset(init_poi_value_s)

                    hr_obs = np.zeros(
                        [ENV_CONF['uav_num'], CONF['obs_shape'][0] - 1 + ENV_CONF['data_type_num'], *CONF['hr_shape']],
                        dtype=np.float32)
                    obs = np.zeros([ENV_CONF['uav_num'], CONF['obs_shape'][0], CONF['obs_range'], CONF['obs_range']],
                                   dtype=np.float32)
                    ux = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 0] / ENV_CONF['field_length'][0] *
                                  CONF['hr_shape'][0], dtype=np.long)
                    uy = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 1] / ENV_CONF['field_length'][1] *
                                  CONF['hr_shape'][1], dtype=np.long)
                    # -----------------------------
                    px = np.array(st['cur_peo_pos_value'][:, 0] / ENV_CONF['field_length'][0] *
                                  CONF['hr_shape'][0], dtype=np.long)
                    py = np.array(st['cur_peo_pos_value'][:, 1] / ENV_CONF['field_length'][1] *
                                  CONF['hr_shape'][1], dtype=np.long)
                    retimg = np.zeros([ENV_CONF['uav_num'], *CONF['obs_shape']], dtype=np.float32)

                    uid_pos = np.zeros([ENV_CONF['uav_num'], 2], dtype=np.long)
                    p_msk = np.zeros(
                        [ENV_CONF['uav_num'], 1, CONF['mtx_size'] * CONF['mtx_size'], *CONF['M_size']],
                        dtype=np.float32)
                    n_msk = np.ones([ENV_CONF['uav_num'], *CONF['M_size']])
                    obs, uid_pos, p_msk, n_msk = gen_obs(hr_obs, obs, ux, uy, env.blk_hr_grid_range,
                                                         CONF['hr_shape'], ENV_CONF['uav_num'],
                                                         st['cur_poi2hr_grid_range_channel'],
                                                         CONF['obs_shape'], px, py, retimg,
                                                         st['uav_types'],
                                                         st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                         :, 0],
                                                         st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                         :, 1],
                                                         uid_pos,
                                                         ENV_CONF['field_length'], CONF['M_size'],
                                                         p_msk, n_msk, CONF['mtx_size'], st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                         :, 2]
                                                         )
                    sublog.episode_info = []
                    sublog.episode_info.append([st, obs])
                    shared_rollout.reset()
                    shared_rollout.obs_s[0, 0].copy_(torch.tensor(obs, dtype=torch.float32))
                    shared_rollout.p_msk_s[0, 0].copy_(torch.tensor(p_msk, dtype=torch.float32))
                    shared_rollout.n_msk_s[0, 0].copy_(torch.tensor(n_msk, dtype=torch.float32))
                    last_action_np = np.zeros(
                        [ENV_CONF['uav_num'], CONF['action_space']], dtype=np.float32)
                    for step_id in range(ENV_CONF['max_step']):
                        value_list = []
                        action_list = []
                        action_log_prob_list = []
                        h_list = []

                        obs_all = shared_rollout.obs_s[step_id]
                        obs_all_flat = obs_all.view(obs_all.size(0), -1,
                                                    *CONF['obs_shape'][1:])

                        if is_comm:
                            comm_alpha = float(get_comm_alpha(test=True))
                            last_act_t = torch.tensor(last_action_np, dtype=torch.float32)
                            x_embed_list = []
                            z_list = []
                            for uid in range(ENV_CONF['uav_num']):
                                x_embed_uid, z_uid, h_uid = local_ac_list[uid].get_z(
                                    shared_rollout.obs_s[step_id, :, uid],
                                    shared_rollout.recurrent_hidden_states_s[step_id, :, uid],
                                    last_act_t[uid:uid + 1])
                                x_embed_list.append(x_embed_uid)
                                z_list.append(z_uid)
                                h_list.append(h_uid.unsqueeze(1))
                            z_all_t = torch.stack(z_list, dim=1)
                            h_comm, _ = local_comm(
                                z_all_t, update_groups=True,
                                force_cluster=(
                                    CONF.get('comm_cluster_at_slot0', True)
                                    and step_id == 0))
                            _gl = torch.tensor(
                                local_comm.get_group_labels(), dtype=torch.int8)
                            shared_rollout.group_labels_s[step_id, 0].copy_(_gl)
                            critic_use_comm = CONF.get('critic_use_comm', False)
                            h_comm_all = (h_comm * comm_alpha) if critic_use_comm else None
                            if critic_use_comm:
                                shared_rollout.h_comm_s[step_id, 0].copy_(
                                    (h_comm * comm_alpha).detach())
                            for uid in range(ENV_CONF['uav_num']):
                                val, act, alp = local_ac_list[uid].get_action_from_z(
                                    x_embed_list[uid], z_list[uid], h_comm[:, uid],
                                    obs_all=obs_all_flat, h_comm_all=h_comm_all, comm_alpha=comm_alpha)
                                value_list.append(val.unsqueeze(1))
                                action_list.append(act)
                                action_log_prob_list.append(alp.unsqueeze(1))
                        else:
                            for uid in range(ENV_CONF['uav_num']):
                                value, action, action_log_prob, h = local_ac_list[uid].get_action_s(
                                    shared_rollout.obs_s[step_id, :, uid],
                                    shared_rollout.recurrent_hidden_states_s[step_id, :, uid],
                                    shared_rollout.p_msk_s[step_id, :, uid],
                                    shared_rollout.n_msk_s[step_id, :, uid],
                                    obs_all=obs_all_flat,
                                )
                                value_list.append(torch.unsqueeze(value, 1))
                                action_list.append(action)
                                action_log_prob_list.append(torch.unsqueeze(action_log_prob, 1))
                                h_list.append(torch.unsqueeze(h, 1))

                        value_s = torch.cat(value_list, 1)
                        action_s = process_action(action_list)
                        action_s_log_prob = torch.cat(action_log_prob_list, 1)
                        h = torch.cat(h_list, 1)
                        st, done = env.step(action_s)
                        hr_obs = np.zeros(
                            [ENV_CONF['uav_num'], CONF['obs_shape'][0] - 1 + ENV_CONF['data_type_num'],
                             *CONF['hr_shape']],
                            dtype=np.float32)
                        obs = np.zeros(
                            [ENV_CONF['uav_num'], CONF['obs_shape'][0], CONF['obs_range'], CONF['obs_range']],
                            dtype=np.float32)
                        ux = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 0] / ENV_CONF['field_length'][0] *
                                      CONF['hr_shape'][0], dtype=np.long)
                        uy = np.array(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, 1] / ENV_CONF['field_length'][1] *
                                      CONF['hr_shape'][1], dtype=np.long)
                        px = np.array(st['cur_peo_pos_value'][:, 0] / ENV_CONF['field_length'][0] *
                                      CONF['hr_shape'][0], dtype=np.long)
                        py = np.array(st['cur_peo_pos_value'][:, 1] / ENV_CONF['field_length'][1] *
                                      CONF['hr_shape'][1], dtype=np.long)
                        retimg = np.zeros([ENV_CONF['uav_num'], *CONF['obs_shape']], dtype=np.float32)
                        uid_pos = np.zeros([ENV_CONF['uav_num'], 2], dtype=np.long)
                        p_msk = np.zeros([ENV_CONF['uav_num'], 1, CONF['mtx_size'] * CONF['mtx_size'],
                                          *CONF['M_size']],
                                         dtype=np.float32)
                        n_msk = np.ones([ENV_CONF['uav_num'], *CONF['M_size']])
                        obs, uid_pos, p_msk, n_msk = gen_obs(hr_obs, obs, ux, uy,
                                                             env.blk_hr_grid_range,
                                                             CONF['hr_shape'], ENV_CONF['uav_num'],
                                                             st['cur_poi2hr_grid_range_channel'],
                                                             CONF['obs_shape'], px, py, retimg,
                                                             st['uav_types'],
                                                             st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                             :, 0],
                                                             st['cur_uav_pos_energy_hit_mec_cec_charge'][
                                                             :, 1],
                                                             uid_pos,
                                                             ENV_CONF['field_length'],
                                                             CONF['M_size'],
                                                             p_msk, n_msk, CONF['mtx_size'], st[
                                                                 'cur_uav_pos_energy_hit_mec_cec_charge'][
                                                             :, 2]
                                                             )
                        # ----------------------------------------
                        sublog.episode_info.append([st, obs, done, action_s])
                        shared_rollout.insert(torch.tensor(obs, dtype=torch.float32),
                                              value_s,
                                              action_s_log_prob,
                                              torch.tensor(action_s, dtype=torch.float32),
                                              h, torch.tensor(p_msk, dtype=torch.float32),
                                              torch.tensor(n_msk, dtype=torch.float32))
                        # ----------------------------------------------
                    obs_all_last = shared_rollout.obs_s[-1]
                    obs_all_last_flat = obs_all_last.view(
                        obs_all_last.size(0), -1, *CONF['obs_shape'][1:])
                    _bootstrap_h_comm = None
                    if is_comm and CONF.get('critic_use_comm', False):
                        _t_last = ENV_CONF['max_step'] - 1
                        _bootstrap_h_comm = shared_rollout.h_comm_s[_t_last:_t_last + 1, 0]
                    next_value_list = []
                    for uid in range(ENV_CONF['uav_num']):
                        next_value = local_ac_list[uid].get_value_s(
                            obs_all_last_flat, h_comm_all=_bootstrap_h_comm)
                        next_value_list.append(torch.unsqueeze(next_value, 1))
                    next_value_s = torch.cat(next_value_list, 1)
                    shared_rollout.gen_reward(sublog.episode_info)

                    shared_rollout.compute_returns(next_value_s, use_gae=True, gamma=CONF['gamma'], tau=CONF['tau'])
                    ################################## sublog work ####################################
                    sublog.episode_total_reward_list[sub_iter_counter] = float(
                        shared_rollout.reward_s.sum().item())
                    sublog.gen_metrics_result(sub_iter_counter)
                    sublog.record_metrics_result()
                    if is_comm and comm_group_npy_path is not None:
                        _gl = shared_rollout.group_labels_s[:, 0, :].numpy().astype(
                            np.int32)
                        comm_group_labels_vol[:, :, sub_iter_counter] = _gl.T
                        np.save(comm_group_npy_path, comm_group_labels_vol)
                    if process_id == 0 and sub_iter_counter % 10 == 0:
                        sublog.record_trace_se(sub_iter_counter, env)
                    if sub_iter_counter % 50 == 0:
                        sublog.draw_ana(sub_iter_counter, shared_rollout.reward_s.numpy())
                    shared_ifdone.value = True
                    sub_iter_counter += 1
                    break
