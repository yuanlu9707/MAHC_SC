from .conf import *


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage:
    def __init__(self, ENV_CONF):
        self.ENV_CONF = ENV_CONF
        self.obs_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], *CONF['obs_shape']],
            dtype=torch.float32)
        self.value_s = torch.zeros([self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                   dtype=torch.float32)
        self.return_s = torch.zeros([self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                    dtype=torch.float32)

        self.reward_s = torch.zeros([self.ENV_CONF['max_step'], CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                    dtype=torch.float32)
        self.action_s_log_prob = torch.zeros([self.ENV_CONF['max_step'], CONF['env_num'], self.ENV_CONF['uav_num'], 1],
                                             dtype=torch.float32)
        self.action_s = torch.zeros(
            [self.ENV_CONF['max_step'], CONF['env_num'], self.ENV_CONF['uav_num'], CONF['action_space']],
            dtype=torch.float32)
        self.recurrent_hidden_states_s = torch.zeros(self.ENV_CONF['max_step'] + 1, CONF['env_num'],
                                                     self.ENV_CONF['uav_num'],
                                                     *CONF['M_size'], dtype=torch.float32)

        self.p_msk_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], 1,
             CONF['mtx_size'] * CONF['mtx_size'], *CONF['M_size']],
            dtype=torch.float32)
        self.n_msk_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, CONF['env_num'], self.ENV_CONF['uav_num'], *CONF['M_size']],
            dtype=torch.float32)

        # 每个时隙的分组标签轨迹，shape (max_step, env_num, n_agents)，int8
        self.group_labels_s = torch.zeros(
            [self.ENV_CONF['max_step'], CONF['env_num'], self.ENV_CONF['uav_num']],
            dtype=torch.int8)

        # 每个时隙的预测特征，shape (max_step, env_num, pred_out_d)
        # subp 以 detach 方式写入，actor/critic 更新时直接读取
        _pred_out_d = CONF.get('pred_out_d', 64)
        self.pred_feat_s = torch.zeros(
            [self.ENV_CONF['max_step'], CONF['env_num'], _pred_out_d],
            dtype=torch.float32)

        self._to_device()

    def _to_device(self):
        self.obs_s = self.obs_s.to(CONF['device'])
        self.value_s = self.value_s.to(CONF['device'])
        self.return_s = self.return_s.to(CONF['device'])

        self.reward_s = self.reward_s.to(CONF['device'])
        self.action_s_log_prob = self.action_s_log_prob.to(CONF['device'])
        self.action_s = self.action_s.to(CONF['device'])

        self.recurrent_hidden_states_s = self.recurrent_hidden_states_s.to(CONF['device'])

        self.p_msk_s = self.p_msk_s.to(CONF['device'])
        self.n_msk_s = self.n_msk_s.to(CONF['device'])
        self.pred_feat_s = self.pred_feat_s.to(CONF['device'])
        # group_labels_s 保留 CPU：仅用于 .cpu().numpy() 取索引，无需 GPU 计算

    def insert(self, shared_rollout, env_num):
        self.obs_s[:, env_num].copy_(shared_rollout.obs_s[:, 0])
        self.value_s[:, env_num].copy_(shared_rollout.value_s[:, 0])

        self.action_s[:, env_num].copy_(shared_rollout.action_s[:, 0])
        self.action_s_log_prob[:, env_num].copy_(shared_rollout.action_s_log_prob[:, 0])

        self.reward_s[:, env_num].copy_(shared_rollout.reward_s[:, 0])
        self.return_s[:, env_num].copy_(shared_rollout.return_s[:, 0])
        self.recurrent_hidden_states_s[:, env_num].copy_(shared_rollout.recurrent_hidden_states_s[:, 0])
        self.p_msk_s[:, env_num].copy_(shared_rollout.p_msk_s[:, 0])
        self.n_msk_s[:, env_num].copy_(shared_rollout.n_msk_s[:, 0])
        self.group_labels_s[:, env_num].copy_(shared_rollout.group_labels_s[:, 0])
        self.pred_feat_s[:, env_num].copy_(shared_rollout.pred_feat_s[:, 0])

    def minibatch_generator(self, advantage_s, uid):
        T, N = CONF['seq_len'], (self.ENV_CONF['max_step'] - CONF['seq_len'] + 1) * CONF['env_num']
        sampler = BatchSampler(SubsetRandomSampler(range(N)), CONF['mini_batch_size'], drop_last=False)
        obs_batch = []
        rhs_batch = []
        action_batch = []
        value_batch = []
        return_batch = []
        old_action_s_log_prob_batch = []
        adv_targ_batch = []
        p_msk_batch = []
        n_msk_batch = []
        obs_all_batch = []
        pred_feat_batch = []
        for start_ind in range(self.ENV_CONF['max_step'] - CONF['seq_len'] + 1):
            start = start_ind
            end = start + CONF['seq_len']

            obs_batch.append(self.obs_s[start:end, :, uid])
            action_batch.append(self.action_s[start:end, :, uid])
            value_batch.append(self.value_s[start:end, :, uid])
            return_batch.append(self.return_s[start:end, :, uid])
            old_action_s_log_prob_batch.append(self.action_s_log_prob[start:end, :, uid])
            adv_targ_batch.append(advantage_s[start:end, :, uid])
            rhs_batch.append(self.recurrent_hidden_states_s[start:start + 1, :, uid])
            p_msk_batch.append(self.p_msk_s[start:end, :, uid])
            n_msk_batch.append(self.n_msk_s[start:end, :, uid])
            obs_all_batch.append(self.obs_s[start:end, :, :])
            pred_feat_batch.append(self.pred_feat_s[start:end, :, :])

        obs_batch = torch.cat(obs_batch, 1)
        action_batch = torch.cat(action_batch, 1)
        value_batch = torch.cat(value_batch, 1)
        return_batch = torch.cat(return_batch, 1)
        old_action_s_log_prob_batch = torch.cat(old_action_s_log_prob_batch, 1)
        adv_targ_batch = torch.cat(adv_targ_batch, 1)
        rhs_batch = torch.cat(rhs_batch, 1).view(N, *CONF['M_size'])
        p_msk_batch = torch.cat(p_msk_batch, 1)
        n_msk_batch = torch.cat(n_msk_batch, 1)
        obs_all_batch = torch.cat(obs_all_batch, 1)  # (T, N, uav_num, C, H, W)
        pred_feat_batch = torch.cat(pred_feat_batch, 1)  # (T, N, pred_out_d)

        for indices in sampler:
            cur_mini_batch_size = len(indices)
            obs_mini_batch = _flatten_helper(T, cur_mini_batch_size, obs_batch[:, indices])
            action_mini_batch = _flatten_helper(T, cur_mini_batch_size, action_batch[:, indices])
            value_mini_batch = _flatten_helper(T, cur_mini_batch_size, value_batch[:, indices])
            return_mini_batch = _flatten_helper(T, cur_mini_batch_size, return_batch[:, indices])

            old_action_s_log_prob_mini_batch = _flatten_helper(T, cur_mini_batch_size,
                                                               old_action_s_log_prob_batch[:, indices])
            adv_targ_mini_batch = _flatten_helper(T, cur_mini_batch_size, adv_targ_batch[:, indices])
            rhs_mini_batch = rhs_batch[indices]
            p_msk_mini_batch = _flatten_helper(T, cur_mini_batch_size, p_msk_batch[:, indices])
            n_msk_mini_batch = _flatten_helper(T, cur_mini_batch_size, n_msk_batch[:, indices])

            obs_all_mini = _flatten_helper(T, cur_mini_batch_size, obs_all_batch[:, indices])
            # (T*mini_batch, uav_num, C, H, W) -> (T*mini_batch, uav_num*C, H, W)
            obs_all_mini_batch = obs_all_mini.view(obs_all_mini.size(0), -1, *CONF['obs_shape'][1:])
            pred_feat_mini_batch = _flatten_helper(
                T, cur_mini_batch_size, pred_feat_batch[:, indices])

            yield obs_mini_batch, action_mini_batch, value_mini_batch, return_mini_batch, \
                  old_action_s_log_prob_mini_batch, adv_targ_mini_batch, rhs_mini_batch, \
                  p_msk_mini_batch, n_msk_mini_batch, obs_all_mini_batch, pred_feat_mini_batch

    def comm_minibatch_generator(self, advantage_s):
        """MAPPO_COMM 专用: 同一时间窗口下所有智能体数据一起 yield。"""
        T = CONF['seq_len']
        uav_num = self.ENV_CONF['uav_num']
        num_windows = self.ENV_CONF['max_step'] - T + 1
        N = num_windows * CONF['env_num']
        sampler = BatchSampler(SubsetRandomSampler(range(N)),
                               CONF['mini_batch_size'], drop_last=False)

        padded_action = torch.cat(
            [torch.zeros(1, *self.action_s.shape[1:], device=self.action_s.device),
             self.action_s], dim=0)

        obs_b = {uid: [] for uid in range(uav_num)}
        act_b = {uid: [] for uid in range(uav_num)}
        val_b = {uid: [] for uid in range(uav_num)}
        ret_b = {uid: [] for uid in range(uav_num)}
        olp_b = {uid: [] for uid in range(uav_num)}
        adv_b = {uid: [] for uid in range(uav_num)}
        h_b = {uid: [] for uid in range(uav_num)}
        pa_b = {uid: [] for uid in range(uav_num)}
        obs_all_b = []
        # 分组标签：取每个 window 起始时隙的标签，shape list of (env_num, n_agents)
        gl_b = []
        # pred_feat 窗口：shape list of (T, env_num, pred_out_d)
        pf_b = []

        for start in range(num_windows):
            end = start + T
            for uid in range(uav_num):
                obs_b[uid].append(self.obs_s[start:end, :, uid])
                act_b[uid].append(self.action_s[start:end, :, uid])
                val_b[uid].append(self.value_s[start:end, :, uid])
                ret_b[uid].append(self.return_s[start:end, :, uid])
                olp_b[uid].append(self.action_s_log_prob[start:end, :, uid])
                adv_b[uid].append(advantage_s[start:end, :, uid])
                h_b[uid].append(self.recurrent_hidden_states_s[start:start + 1, :, uid])
                pa_b[uid].append(padded_action[start:end, :, uid])
            obs_all_b.append(self.obs_s[start:end, :, :])
            # 每个 window 取全部 T 个时隙的标签，shape (T, env_num, n_agents)
            gl_b.append(self.group_labels_s[start:end, :, :])
            # pred_feat：(T, env_num, pred_out_d)
            pf_b.append(self.pred_feat_s[start:end, :, :])

        for uid in range(uav_num):
            obs_b[uid] = torch.cat(obs_b[uid], 1)
            act_b[uid] = torch.cat(act_b[uid], 1)
            val_b[uid] = torch.cat(val_b[uid], 1)
            ret_b[uid] = torch.cat(ret_b[uid], 1)
            olp_b[uid] = torch.cat(olp_b[uid], 1)
            adv_b[uid] = torch.cat(adv_b[uid], 1)
            h_b[uid] = torch.cat(h_b[uid], 1).view(N, *CONF['M_size'])
            pa_b[uid] = torch.cat(pa_b[uid], 1)
        obs_all_b = torch.cat(obs_all_b, 1)
        # gl_b: list of num_windows 个 (T, env_num, n_agents)
        # cat 后 → (T, N, n_agents)，与其他数据维度对齐（N = num_windows * env_num）
        gl_b_cat = torch.cat(gl_b, dim=1)   # (T, N, n_agents)
        # pred_feat: (T, N, pred_out_d)
        pf_b_cat = torch.cat(pf_b, dim=1)   # (T, N, pred_out_d)

        for indices in sampler:
            m = len(indices)
            obs_uid = [_flatten_helper(T, m, obs_b[u][:, indices]) for u in range(uav_num)]
            action_uid = [_flatten_helper(T, m, act_b[u][:, indices]) for u in range(uav_num)]
            value_uid = [_flatten_helper(T, m, val_b[u][:, indices]) for u in range(uav_num)]
            return_uid = [_flatten_helper(T, m, ret_b[u][:, indices]) for u in range(uav_num)]
            old_lp_uid = [_flatten_helper(T, m, olp_b[u][:, indices]) for u in range(uav_num)]
            adv_uid = [_flatten_helper(T, m, adv_b[u][:, indices]) for u in range(uav_num)]
            h_uid = [h_b[u][indices] for u in range(uav_num)]
            prev_act_uid = [_flatten_helper(T, m, pa_b[u][:, indices]) for u in range(uav_num)]
            oa = _flatten_helper(T, m, obs_all_b[:, indices])
            obs_all_flat = oa.view(oa.size(0), -1, *CONF['obs_shape'][1:])
            # 分组标签：(T*m, n_agents) int8，与 z_all 完全对齐（每个样本都有自己的分组）
            group_labels_mini = _flatten_helper(T, m, gl_b_cat[:, indices])
            # pred_feat：(T*m, pred_out_d)，已 detach（写入 buffer 时已 detach）
            pred_feat_mini = _flatten_helper(T, m, pf_b_cat[:, indices])

            yield (obs_uid, action_uid, value_uid, return_uid,
                   old_lp_uid, adv_uid, h_uid, prev_act_uid, obs_all_flat,
                   group_labels_mini, pred_feat_mini)
