from .conf import *


class ShareRolloutStorage:
    def __init__(self, ENV_CONF):
        self.ENV_CONF = ENV_CONF
        self.obs_s = torch.zeros([self.ENV_CONF['max_step'] + 1, 1, self.ENV_CONF['uav_num'], *CONF['obs_shape']],
                                 dtype=torch.float32).share_memory_()
        self.value_s = torch.zeros([self.ENV_CONF['max_step'] + 1, 1, self.ENV_CONF['uav_num'], 1],
                                   dtype=torch.float32).share_memory_()
        self.return_s = torch.zeros([self.ENV_CONF['max_step'] + 1, 1, self.ENV_CONF['uav_num'], 1],
                                    dtype=torch.float32).share_memory_()

        self.reward_s = torch.zeros([self.ENV_CONF['max_step'], 1, self.ENV_CONF['uav_num'], 1],
                                    dtype=torch.float32).share_memory_()
        self.action_s_log_prob = torch.zeros([self.ENV_CONF['max_step'], 1, self.ENV_CONF['uav_num'], 1],
                                             dtype=torch.float32).share_memory_()
        self.action_s = torch.zeros([self.ENV_CONF['max_step'], 1, self.ENV_CONF['uav_num'], CONF['action_space']],
                                    dtype=torch.float32).share_memory_()
        self.step = mp.Value('l', 0)

        self.recurrent_hidden_states_s = torch.zeros(self.ENV_CONF['max_step'] + 1, 1, self.ENV_CONF['uav_num'],
                                                     *CONF['M_size'], dtype=torch.float32).share_memory_()
        self.p_msk_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, 1, self.ENV_CONF['uav_num'], 1, CONF['mtx_size'] * CONF['mtx_size'],
             *CONF['M_size']],
            dtype=torch.float32).share_memory_()
        self.n_msk_s = torch.zeros(
            [self.ENV_CONF['max_step'] + 1, 1, self.ENV_CONF['uav_num'], *CONF['M_size']],
            dtype=torch.float32).share_memory_()

        # 每个时隙的智能体分组标签，shape (max_step, 1, n_agents)
        # 值含义：agent i 属于第 group_labels_s[t,0,i] 组
        self.group_labels_s = torch.zeros(
            [self.ENV_CONF['max_step'], 1, self.ENV_CONF['uav_num']],
            dtype=torch.int8).share_memory_()

        # 每个时隙的预测特征，shape (max_step, 1, pred_out_d)
        # 由 subp 中 local_pred 推理后 detach 写入；0 时隙使用全零（冷启动）
        _pred_out_d = CONF.get('pred_out_d', 64)
        self.pred_feat_s = torch.zeros(
            [self.ENV_CONF['max_step'], 1, _pred_out_d],
            dtype=torch.float32).share_memory_()

          # MAPPO_COMM + critic_use_comm：每时隙 actor 路径上算出的 h_comm (B=1)，供 bootstrap 直接读取
        _hgcn_d = int(CONF.get('hgcn_out_dim', 64))
        self.h_comm_s = torch.zeros(
            [self.ENV_CONF['max_step'], 1, self.ENV_CONF['uav_num'], _hgcn_d],
            dtype=torch.float32).share_memory_()

        self.reset()

    def reset(self):
        self.step.value = 0

    def gen_reward(self, episode_info):
        for step in range(self.ENV_CONF['max_step']):
            st = episode_info[step][0]
            next_st = episode_info[step + 1][0]

            env_init_poi_value = st['env_init_poi_value']
            uav_pos_energy_hit_mec_cec_charge = st['cur_uav_pos_energy_hit_mec_cec_charge']
            # -------------------------------------
            uav_dc = st['cur_uav_dc']
            next_peo_visit_poi_time = next_st['cur_peo_visit_poi_time']
            next_uav_visit_poi_time = next_st['cur_uav_visit_poi_time']
            next_uav_dc = next_st['cur_uav_dc']
            next_uav_pos_energy_hit_mec_cec_charge = next_st['cur_uav_pos_energy_hit_mec_cec_charge']
            # -------------------------------------

            # f
            f = 0.0
            active_poi_id_list = np.nonzero(np.sum(env_init_poi_value, axis=-1))
            next_uav_peo_visit_poi_time = (next_peo_visit_poi_time + np.sum(next_uav_visit_poi_time, axis=0))[
                active_poi_id_list]
            square_of_sum = np.square(np.sum(next_uav_peo_visit_poi_time))
            sum_of_square = np.sum(np.square(next_uav_peo_visit_poi_time))
            if sum_of_square > self.ENV_CONF['min_value']:
                f = square_of_sum / sum_of_square / len(next_uav_peo_visit_poi_time)

            # uav_dc
            uav_dc = np.sum(next_uav_dc - uav_dc, axis=1)

            ec = (next_uav_pos_energy_hit_mec_cec_charge[:, 4] - uav_pos_energy_hit_mec_cec_charge[:, 4]) + (
                    next_uav_pos_energy_hit_mec_cec_charge[:, 5] - uav_pos_energy_hit_mec_cec_charge[:, 5])

            # hit
            hit = next_uav_pos_energy_hit_mec_cec_charge[:, 3] - uav_pos_energy_hit_mec_cec_charge[:, 3]

            charge = next_uav_pos_energy_hit_mec_cec_charge[:, 6] - uav_pos_energy_hit_mec_cec_charge[:, 6]
            clipped_charge = copy.deepcopy(charge)
            clipped_charge[charge < self.ENV_CONF['charge_min_factor'] * self.ENV_CONF['uav_init_energy']] = 0

            # reward
            reward_collect = (self.ENV_CONF['positive_factor'] * f * uav_dc) / (ec + self.ENV_CONF['min_value'])
            reward_collect_clipped = np.clip(reward_collect, -10, 10)
            penalty = - self.ENV_CONF['penalty_factor'] * hit
            reward_charge = self.ENV_CONF['charge_factor'] * clipped_charge / self.ENV_CONF['uav_init_energy']
            reward = reward_collect_clipped + penalty + reward_charge
            self.reward_s[step, 0] = torch.unsqueeze(torch.tensor(reward, dtype=torch.float32), 1)

    def insert(self, obs_s, value_s, action_s_log_prob, action_s, recurrent_hidden_states_s,
               p_msk_s, n_msk_s, pred_feat=None):
        """
        pred_feat : (1, pred_out_d) FloatTensor 或 None。
            当 actor_use_pred=True 时由 subp 传入（已 detach）。
        """
        self.obs_s[self.step.value + 1, 0].copy_(obs_s)
        self.value_s[self.step.value].copy_(value_s)
        self.action_s[self.step.value].copy_(action_s)
        self.action_s_log_prob[self.step.value].copy_(action_s_log_prob)
        self.recurrent_hidden_states_s[self.step.value + 1].copy_(recurrent_hidden_states_s)
        self.p_msk_s[self.step.value + 1].copy_(p_msk_s)
        self.n_msk_s[self.step.value + 1].copy_(n_msk_s)
        if pred_feat is not None:
            self.pred_feat_s[self.step.value, 0].copy_(pred_feat.squeeze(0))
        self.step.value = (self.step.value + 1) % self.ENV_CONF['max_step']

    def compute_returns(self, next_value_s, use_gae, gamma, tau):
        if use_gae:
            self.value_s[-1].copy_(next_value_s)
            gae = 0
            for step in reversed(range(self.reward_s.size(0))):
                delta = self.reward_s[step] + gamma * self.value_s[step + 1] - self.value_s[step]
                gae = delta + gamma * tau * gae
                self.return_s[step] = gae + self.value_s[step]
        else:
            self.return_s[-1] = next_value_s
            for step in reversed(range(self.reward_s.size(0))):
                self.return_s[step] = self.return_s[step + 1] * gamma + self.reward_s[step]
