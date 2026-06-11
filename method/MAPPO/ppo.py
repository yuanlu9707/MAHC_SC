from .conf import *
from .pred import pred_module_loss, build_pred_train_batch


def adjust_learning_rate(optimizer, lr, iter_id):
    lr = CONF['lr'] * CONF['decay_rate'] ** max(0, iter_id - CONF['decay_start_iter_id'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class PPO:
    """统一训练器，同时支持 MAPPO 和 MAPPO_COMM。

    MAPPO_COMM 时传入 comm_module:
      - 每个 UAV 的 actor / critic 各自独立更新；
      - 通信模块有自己的 optimizer，用所有 actor 平均 loss + 所有 critic 平均
        loss + 组一致性损失 + 注意力正则化损失。
    pred_feat:
      - 默认由主进程预训练后写入 pred_feat_timeseries.npy，subp 按 step_id 查表；
      - actor 更新时从 buffer 取 pred_feat（与采集时一致）；
      - MAPPO/MAPPO_CNN 且 critic_use_pred 时，Critic 前向同样使用该 pred_feat。
    """

    def __init__(self, ac_list, comm_module=None, pred_module=None):
        self.uav_num = len(ac_list)
        self.ac_list = ac_list
        self.comm = comm_module
        self.pred = pred_module
        self.lr_list = [CONF['lr'] for _ in range(self.uav_num)]
        self._use_pred = CONF.get('actor_use_pred', False)
        self._any_critic_use_pred = any(
            getattr(ac, '_critic_use_pred', False) for ac in self.ac_list)

        self.actor_optimizer_list = []
        self.critic_optimizer_list = []

        for ac in self.ac_list:
            if self.comm is not None:
                actor_params = (list(ac.actor_base.parameters())
                                + list(ac.action_head.parameters())
                                + list(ac.dist_dia.parameters()))
            else:
                actor_params = list(ac.actor_base.parameters())
                if hasattr(ac, 'action_head') and ac.action_head is not None:
                    actor_params += list(ac.action_head.parameters())
                actor_params += list(ac.dist_dia.parameters())
            self.actor_optimizer_list.append(
                optim.Adam(actor_params, lr=CONF['lr'], eps=CONF['eps'], weight_decay=1e-6))
            self.critic_optimizer_list.append(
                optim.Adam(list(ac.critic.parameters()),
                           lr=CONF['lr'], eps=CONF['eps'], weight_decay=1e-6))

        if self.comm is not None:
            self.comm_optimizer = optim.Adam(
                self.comm.parameters(), lr=CONF['lr'], eps=CONF['eps'], weight_decay=1e-6)

        if self.pred is not None:
            self.pred_optimizer = optim.Adam(
                self.pred.parameters(),
                lr=CONF.get('pred_lr', CONF.get('pred_lr_init', CONF['lr'])),
                eps=CONF['eps'], weight_decay=1e-6)
        else:
            self.pred_optimizer = None

    # ------------------------------------------------------------------
    def update(self, rollouts, iter_id):
        if self.comm is not None:
            return self._update_comm(rollouts, iter_id)
        return self._update_standard(rollouts, iter_id)

    # ------------------------------------------------------------------
    # 原始 MAPPO
    # ------------------------------------------------------------------
    def _update_standard(self, rollouts, iter_id):
        advantage_s = rollouts.return_s[:-1] - rollouts.value_s[:-1]
        advantage_s = (advantage_s - advantage_s.mean()) / (advantage_s.std() + 1e-5)

        value_loss_total = 0
        action_loss_total = 0
        dist_entropy_total = 0
        loss_total = 0
        sample_num = 0

        for _ in range(CONF['buffer_replay_time']):
            for uid in range(self.uav_num):
                data_generator = rollouts.minibatch_generator(advantage_s, uid)
                for sample_mini_batch in data_generator:
                    obs_mini_batch, action_mini_batch, value_mini_batch, return_mini_batch, \
                        old_action_s_log_prob_mini_batch, adv_targ_mini_batch, h_mini_batch, \
                        p_msk_mini_batch, n_msk_mini_batch, obs_all_mini_batch, \
                        pred_feat_mini_batch = sample_mini_batch

                    sample_num += action_mini_batch.size(0)

                    _pf = pred_feat_mini_batch if (
                        self._use_pred or self._any_critic_use_pred) else None
                    evl_value_s, dist_entropy_s, action_s_log_prob, h = \
                        self.ac_list[uid].evaluate_action_s(
                            obs_mini_batch, action_mini_batch, h_mini_batch,
                            p_msk_mini_batch, n_msk_mini_batch,
                            obs_all=obs_all_mini_batch, pred_feat=_pf)

                    ratio = torch.exp(action_s_log_prob - old_action_s_log_prob_mini_batch)
                    surr1 = ratio * adv_targ_mini_batch
                    surr2 = torch.clamp(ratio, 1.0 - CONF['clip_param'],
                                        1.0 + CONF['clip_param']) * adv_targ_mini_batch
                    action_loss = -torch.min(surr1, surr2).mean()
                    actor_loss = action_loss - dist_entropy_s * CONF['entropy_coef']

                    self.actor_optimizer_list[uid].zero_grad()
                    actor_loss.backward(retain_graph=True)
                    _ap = (list(self.ac_list[uid].actor_base.parameters())
                           + list(self.ac_list[uid].dist_dia.parameters()))
                    if hasattr(self.ac_list[uid], 'action_head') and self.ac_list[uid].action_head is not None:
                        _ap += list(self.ac_list[uid].action_head.parameters())
                    nn.utils.clip_grad_norm_(_ap, CONF['max_grad_norm'])
                    self.actor_optimizer_list[uid].step()

                    if CONF['use_clipped_value_loss']:
                        value_pred_clipped = value_mini_batch + \
                                             (evl_value_s - value_mini_batch).clamp(
                                                 -CONF['clip_param'], CONF['clip_param'])
                        value_losses = (evl_value_s - return_mini_batch).pow(2)
                        value_losses_clipped = (value_pred_clipped - return_mini_batch).pow(2)
                        value_loss = .5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(return_mini_batch, evl_value_s)

                    critic_loss = value_loss * CONF['value_loss_coef']

                    self.critic_optimizer_list[uid].zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.ac_list[uid].critic.parameters(),
                        CONF['max_grad_norm'])
                    self.critic_optimizer_list[uid].step()

                    value_loss_total += value_loss.item()
                    action_loss_total += action_loss.item()
                    dist_entropy_total += dist_entropy_s.item()
                    loss_total += (actor_loss.item() + critic_loss.item())

        value_loss_per_sample = value_loss_total / sample_num
        action_loss_per_sample = action_loss_total / sample_num
        dist_entropy_per_sample = dist_entropy_total / sample_num
        loss_per_sample = loss_total / sample_num

        for uid in range(self.uav_num):
            self.lr_list[uid] = adjust_learning_rate(
                optimizer=self.actor_optimizer_list[uid],
                lr=self.lr_list[uid], iter_id=iter_id)
            adjust_learning_rate(
                optimizer=self.critic_optimizer_list[uid],
                lr=self.lr_list[uid], iter_id=iter_id)

        return value_loss_per_sample, action_loss_per_sample, dist_entropy_per_sample, loss_per_sample

    # ------------------------------------------------------------------
    # MAPPO_COMM：各 agent 独立更新，comm 用平均 loss 更新
    # ------------------------------------------------------------------
    def _update_comm(self, rollouts, iter_id):
        advantage_s = rollouts.return_s[:-1] - rollouts.value_s[:-1]
        advantage_s = (advantage_s - advantage_s.mean()) / (advantage_s.std() + 1e-5)

        value_loss_total = 0
        action_loss_total = 0
        dist_entropy_total = 0
        loss_total = 0
        sample_num = 0

        for _ in range(CONF['buffer_replay_time']):
            data_generator = rollouts.comm_minibatch_generator(advantage_s)
            for batch in data_generator:
                (obs_uid, action_uid, value_uid, return_uid,
                 old_lp_uid, adv_uid, h_uid, prev_act_uid,
                 obs_all_mini_batch, group_labels_mini, pred_feat_mini) = batch
                comm_alpha = float(get_comm_alpha(iter_i=iter_id, test=False))

                B = obs_uid[0].size(0)
                sample_num += B * self.uav_num
                device = obs_uid[0].device

                # ===== Phase 1: 所有智能体前向 → z → comm → h_comm =====
                x_embed_list = []
                z_list = []
                for uid in range(self.uav_num):
                    x_embed, z, _ = self.ac_list[uid].get_z(
                        obs_uid[uid], h_uid[uid], prev_act_uid[uid])
                    x_embed_list.append(x_embed)
                    z_list.append(z)

                z_all = torch.stack(z_list, dim=1)               # (B, uav_num, z_dim)

                # ===== 样本对齐：B=T*m 条样本各自使用轨迹中记录的真实分组 =====
                # group_labels_mini 已在 CPU(int8)，np.unique 向量化求唯一配置，O(B log B)
                _gl_np = group_labels_mini.numpy()                # (B, n_agents) int8，已 CPU
                _unique_rows, _inverse = np.unique(
                    _gl_np, axis=0, return_inverse=True)          # K×n_agents, (B,)

                all_h_subs: list = []    # 各 config 的 h_sub (n_sub, n_agents, h_dim)
                all_idx_ts: list = []    # 各 config 的原始下标 (n_sub,) GPU LongTensor
                attention_reg    = torch.tensor(0.0, device=device)
                consistency_loss = torch.tensor(0.0, device=device)

                for _k in range(len(_unique_rows)):
                    _mask = np.where(_inverse == _k)[0]           # (n_sub,) numpy int64
                    self.comm.set_groups(
                        self.comm.labels_to_groups(
                            _unique_rows[_k].tolist()))
                    # from_numpy 零拷贝，to(device) 一次性传 GPU
                    _idx_t = torch.from_numpy(_mask).to(device)
                    # Detach communication input to block gradient path:
                    # h_comm -> z -> actor_base.
                    # _z_sub = z_all[_idx_t]
                    _z_sub = z_all[_idx_t].detach()               # (n_sub, n_agents, z_dim).detach() .detach() .detach() .detach() .detach() .detach()
                    _h_sub, _att_sub = self.comm(_z_sub, update_groups=False)
                    all_h_subs.append(_h_sub)
                    all_idx_ts.append(_idx_t)
                    _w = len(_mask) / B
                    attention_reg = attention_reg + _att_sub * _w
                    # consistency_loss：在各自正确分组下计算，加权平均
                    for _grp in self.comm.get_groups():
                        if len(_grp) > 1:
                            _gf = _h_sub[:, _grp, :]
                            consistency_loss = consistency_loss + (
                                (_gf - _gf.mean(dim=1, keepdim=True)) ** 2
                            ).mean() * _w

                # 恢复 batch 顺序：与 h_comm[_cat_idx]=_cat_h 数学等价；用 _cat_h[_inv] 避免
                # 原地 scatter 在 grad(comm_total, comm)+分 agent backward 时的 autograd 版本冲突。
                _cat_h = torch.cat(all_h_subs, dim=0)              # (B, n_agents, h_dim)
                _cat_idx = torch.cat(all_idx_ts, dim=0)            # 第 i 行落到 j=_cat_idx[i]
                _inv = torch.empty(B, dtype=torch.long, device=device)
                _inv[_cat_idx] = torch.arange(B, device=device, dtype=torch.long)
                h_comm = _cat_h[_inv]                              # (B, n_agents, h_dim)

                critic_use_comm = CONF.get('critic_use_comm', False)
                h_comm_all = (h_comm * comm_alpha) if critic_use_comm else None

                # ===== Phase 2: 各智能体计算 actor_loss / critic_loss（保留计算图）=====
                # pred_feat_mini: (T*m, pred_out_d) 已 detach；actor 与（COMM 下）critic 用 pred 时都传入
                _pf = pred_feat_mini if (
                    self._use_pred or self._any_critic_use_pred) else None

                actor_losses = []
                critic_losses = []
                for uid in range(self.uav_num):
                    evl_value, dist_entropy, action_log_prob = \
                        self.ac_list[uid].evaluate_from_z(
                            x_embed_list[uid], z_list[uid], h_comm[:, uid],
                            action_uid[uid], obs_all=obs_all_mini_batch,
                            h_comm_all=h_comm_all, pred_feat=_pf, comm_alpha=comm_alpha)

                    ratio = torch.exp(action_log_prob - old_lp_uid[uid])
                    surr1 = ratio * adv_uid[uid]
                    surr2 = torch.clamp(ratio, 1.0 - CONF['clip_param'],
                                        1.0 + CONF['clip_param']) * adv_uid[uid]
                    action_loss = -torch.min(surr1, surr2).mean()
                    actor_loss = action_loss - dist_entropy * CONF['entropy_coef']
                    actor_losses.append(actor_loss)

                    if CONF['use_clipped_value_loss']:
                        vp_clipped = value_uid[uid] + (evl_value - value_uid[uid]).clamp(
                            -CONF['clip_param'], CONF['clip_param'])
                        vl = (evl_value - return_uid[uid]).pow(2)
                        vl_c = (vp_clipped - return_uid[uid]).pow(2)
                        value_loss = 0.5 * torch.max(vl, vl_c).mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(return_uid[uid], evl_value)
                    critic_loss = value_loss * CONF['value_loss_coef']
                    critic_losses.append(critic_loss)

                    value_loss_total += value_loss.item()
                    action_loss_total += action_loss.item()
                    dist_entropy_total += dist_entropy.item()

                # ===== Phase 3~5：comm 梯度预取 + backward/step =====
                # critic_use_comm=False：critic 不经 h_comm→actor，A/B 等价；
                # critic_use_comm=True：先完成全部 actor+critic backward，再 step，否则
                # actor.step() 后 critic.backward 报 version mismatch。

                # ----- a) 在任何 step 之前，先用 autograd.grad 预取 comm 的梯度并缓存； -----
                avg_actor_loss = sum(actor_losses) / self.uav_num
                avg_critic_loss = sum(critic_losses) / self.uav_num
                # consistency_loss 已在上方按各样本真实分组计算

                # Communication module is optimized by actor objective only
                # (plus communication regularizers), excluding critic loss.
                comm_total = (avg_actor_loss
                              + CONF['lambda_consistency'] * consistency_loss
                              + CONF['lambda_attention'] * attention_reg)
                if critic_use_comm:
                    comm_total = comm_total + avg_critic_loss

                comm_params = list(self.comm.parameters())
                g_comm_raw = torch.autograd.grad(
                    comm_total, comm_params, retain_graph=True, allow_unused=True)
                g_comm_stash = [None if g is None else g.detach().clone()
                                for g in g_comm_raw]

                _actor_params = lambda uid: (
                    list(self.ac_list[uid].actor_base.parameters())
                    + list(self.ac_list[uid].action_head.parameters())
                    + list(self.ac_list[uid].dist_dia.parameters()))

                if critic_use_comm:
                    for uid in range(self.uav_num):
                        self.actor_optimizer_list[uid].zero_grad()
                        self.critic_optimizer_list[uid].zero_grad()
                    for uid in range(self.uav_num):
                        actor_losses[uid].backward(retain_graph=True)
                    for uid in range(self.uav_num - 1):
                        critic_losses[uid].backward(retain_graph=True)
                    if self.uav_num > 0:
                        critic_losses[-1].backward()
                    for uid in range(self.uav_num):
                        nn.utils.clip_grad_norm_(_actor_params(uid), CONF['max_grad_norm'])
                        self.actor_optimizer_list[uid].step()
                    for uid in range(self.uav_num):
                        nn.utils.clip_grad_norm_(
                            self.ac_list[uid].critic.parameters(),
                            CONF['max_grad_norm'])
                        self.critic_optimizer_list[uid].step()
                else:
                    # ----- b) actor：先全部 backward，再全部 step -----
                    for uid in range(self.uav_num):
                        self.actor_optimizer_list[uid].zero_grad()
                    for uid in range(self.uav_num):
                        actor_losses[uid].backward(retain_graph=True)
                    for uid in range(self.uav_num):
                        nn.utils.clip_grad_norm_(_actor_params(uid), CONF['max_grad_norm'])
                        self.actor_optimizer_list[uid].step()
                    # ----- c) critic：先全部 backward，再全部 step -----
                    for uid in range(self.uav_num):
                        self.critic_optimizer_list[uid].zero_grad()
                    for uid in range(self.uav_num - 1):
                        critic_losses[uid].backward(retain_graph=True)
                    if self.uav_num > 0:
                        critic_losses[-1].backward()
                    for uid in range(self.uav_num):
                        nn.utils.clip_grad_norm_(
                            self.ac_list[uid].critic.parameters(),
                            CONF['max_grad_norm'])
                        self.critic_optimizer_list[uid].step()

                # ----- d) 将缓存梯度写回 comm.parameters().grad，再做 comm.step() -----
                self.comm_optimizer.zero_grad()
                for p, g in zip(comm_params, g_comm_stash):
                    if g is not None:
                        p.grad = g
                nn.utils.clip_grad_norm_(self.comm.parameters(), CONF['max_grad_norm'])
                self.comm_optimizer.step()

                loss_total += (sum(l.item() for l in actor_losses)
                               + sum(l.item() for l in critic_losses)
                               + comm_total.item())

        sample_num = max(sample_num, 1)
        value_loss_per_sample = value_loss_total / sample_num
        action_loss_per_sample = action_loss_total / sample_num
        dist_entropy_per_sample = dist_entropy_total / sample_num
        loss_per_sample = loss_total / sample_num

        for uid in range(self.uav_num):
            self.lr_list[uid] = adjust_learning_rate(
                self.actor_optimizer_list[uid], self.lr_list[uid], iter_id)
            adjust_learning_rate(
                self.critic_optimizer_list[uid], self.lr_list[uid], iter_id)
        adjust_learning_rate(self.comm_optimizer, self.lr_list[0], iter_id)

        return value_loss_per_sample, action_loss_per_sample, dist_entropy_per_sample, loss_per_sample

    # ------------------------------------------------------------------
    # 预测网络独立更新（每 episode 调用一次）
    # ------------------------------------------------------------------
    def update_pred(self, peo_region_id, region_stats):
        """训练 PredModule（每个 RL iter 在 PPO 更新之后调用；与 peo_pos 全时隙一致）。

        Parameters
        ----------
        peo_region_id : (T, P) int64  numpy，来自 peo_pos 的行人区域 ID
        region_stats  : (T, K, 3) float32  numpy，对应区域统计量

        Returns
        -------
        loss_dict : dict，各分项损失值（用于日志）
        """
        if self.pred is None or self.pred_optimizer is None:
            return {}

        device = next(self.pred.parameters()).device
        L = CONF.get('pred_L', 5)

        (peo_id_seq, reg_seq, time_ids, cur_count,
         peo_id_next, reg_stats_next) = build_pred_train_batch(
            peo_region_id, region_stats, L, device=device)

        self.pred.train()
        self.pred_optimizer.zero_grad()

        pred_feat, peo_logits, flow_pred, peo_feat, reg_feat = self.pred(
            peo_id_seq, reg_seq, time_ids, cur_count)
        peo_id_cur = peo_id_seq[:, -1, :].long()

        loss, loss_dict = pred_module_loss(
            peo_logits, flow_pred, peo_id_next, reg_stats_next, cur_count,
            peo_feat=peo_feat, reg_feat=reg_feat, peo_id_current=peo_id_cur)

        loss.backward()
        nn.utils.clip_grad_norm_(self.pred.parameters(), CONF['max_grad_norm'])
        self.pred_optimizer.step()
        self.pred.eval()

        return loss_dict
