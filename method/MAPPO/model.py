import torch
import torch.nn.functional as F
import numpy as np
from .distributions import *
from .cubic_map import *


class Policy(nn.Module):
    """MAPPO Policy：每个智能体持有独立的 Actor 和集中式 Critic。

    - Actor: 输入自身观测，输出动作分布
    - Critic: 输入所有智能体观测拼接，输出 V 值
    """

    def __init__(self, uid):
        super(Policy, self).__init__()
        self.hidden_size = CONF['hidden_size']
        uav_num = CONF['uav_num']

        method = CONF.get('method_name', 'MAPPO')
        if method == 'MAPPO':
            self.actor_base = ActorBase(CONF['obs_shape'][0], self.hidden_size, uid)
        elif method == 'MAPPO_CNN':
            self.actor_base = ActorCNN(CONF['obs_shape'][0], self.hidden_size, uid)
        elif method == 'MAPPO_CNN_GRU':
            self.actor_base = ActorCNN_GRU(CONF['obs_shape'][0], self.hidden_size, uid)
        elif method == 'MAPPO_COMM':
            self.actor_base = ActorCOMM(CONF['obs_shape'][0], self.hidden_size, uid)
        else:
            self.actor_base = ActorBase(CONF['obs_shape'][0], self.hidden_size, uid)

        init_ah = lambda m: init(m, nn.init.orthogonal_,
                                 lambda x: nn.init.constant_(x, 0),
                                 nn.init.calculate_gain('relu'))
        ac_hid = int(CONF.get('hidden_size_ac', 128))

        if method == 'MAPPO_COMM':
            z_dim = int(torch.tensor(CONF['M_size']).prod().item())
            x_embed_dim = int(CONF['hidden_size_gru'])
            hgcn_out_dim = CONF['hgcn_out_dim']
            use_pred = CONF.get('actor_use_pred', False)
            pred_out_d = CONF.get('pred_out_d', 64) if use_pred else 0
            self._cat_fe_mlp = bool(CONF.get('cat_fe_mlp', True))
            self._cat_fe_cnn256 = bool(CONF.get('cat_fe_cnn256', True))
            self._cat_fe_gru = bool(CONF.get('cat_fe_gru', True))
            self._actor_use_pred = use_pred
            self.comm_gate = nn.Parameter(torch.tensor(0.0))
            # [z || h_comm (|| pred_feat)] → hidden_size_ac → action(2)
            in_dim = hgcn_out_dim + pred_out_d
            if self._cat_fe_cnn256 or self._cat_fe_mlp:
                in_dim += x_embed_dim
            if self._cat_fe_gru:
                in_dim += z_dim
            self.action_head = nn.Sequential(
                init_ah(nn.Linear(in_dim, ac_hid)),
                nn.ReLU(),
            )
            self.dist_dia = DiagGaussian(ac_hid, 2) ##128--2
        elif method in ('MAPPO', 'MAPPO_CNN', 'MAPPO_CNN_GRU'):
            use_pred = CONF.get('actor_use_pred', False)
            pred_out_d = CONF.get('pred_out_d', 64) if use_pred else 0
            self._actor_use_pred = use_pred
            # ActorBase / ActorCNN 末层均输出 hidden_size；再 (± pred_feat) → hidden_size_ac → action(2)
            in_dim = self.hidden_size + (pred_out_d if use_pred else 0)
            if method == 'MAPPO_CNN_GRU':
                in_dim = int(torch.tensor(CONF['M_size']).prod().item()) + (pred_out_d if use_pred else 0)
            self.action_head = nn.Sequential(
                init_ah(nn.Linear(in_dim, ac_hid)),
                nn.ReLU(),
            )
            self.dist_dia = DiagGaussian(ac_hid, 2)
        else:
            self._actor_use_pred = False
            self._cat_fe_mlp = False
            self._cat_fe_cnn256 = False
            self._cat_fe_gru = False
            self.action_head = None
            self.dist_dia = DiagGaussian(self.hidden_size, 2)

        self._critic_use_pred = bool(CONF.get('critic_use_pred', False)) and method in (
            'MAPPO', 'MAPPO_CNN', 'MAPPO_CNN_GRU', 'MAPPO_COMM')
        _critic_pf_d = CONF.get('pred_out_d', 64) if self._critic_use_pred else 0

        if method == 'MAPPO':
            self.critic = CentralizedCritic_MLP(
                uav_num, self.hidden_size, pred_feat_dim=_critic_pf_d)
        elif method == 'MAPPO_COMM' and CONF.get('critic_use_comm', False):
            comm_dim = CONF['hgcn_out_dim'] * uav_num
            self.critic = CentralizedCritic(
                uav_num, self.hidden_size, comm_dim=comm_dim, pred_feat_dim=_critic_pf_d)
        elif method in ('MAPPO_CNN', 'MAPPO_CNN_GRU'):
            self.critic = CentralizedCritic(
                uav_num, self.hidden_size, comm_dim=0, pred_feat_dim=_critic_pf_d)
        elif method == 'MAPPO_COMM':
            # 未开 critic_use_comm 时仍可用 critic + 预测特征（无 h_comm 支路）
            self.critic = CentralizedCritic(
                uav_num, self.hidden_size, comm_dim=0, pred_feat_dim=_critic_pf_d)
        else:
            self.critic = CentralizedCritic(uav_num, self.hidden_size, comm_dim=0, pred_feat_dim=0)

    # ---------- 通用接口 (MAPPO / CNN / CNN_GRU) ----------
    def get_action_s(self, obs_s, h, p_msk, n_msk, obs_all=None, pred_feat=None):
        actor_feature_s, rhs_h_s = self.actor_base(obs_s, h, p_msk, n_msk)
        if self.action_head is not None:
            if self._actor_use_pred and pred_feat is not None:
                if pred_feat.dim() == 1:
                    pred_feat = pred_feat.unsqueeze(0)
                pred_feat = pred_feat.to(device=actor_feature_s.device, dtype=actor_feature_s.dtype)
                actor_feature_s = self.action_head(
                    torch.cat([actor_feature_s, pred_feat], dim=-1))
            else:
                actor_feature_s = self.action_head(actor_feature_s)
        dist_dia = self.dist_dia(actor_feature_s)
        action_dia = dist_dia.sample()
        action_log_probs_dia = dist_dia.log_probs(action_dia)

        if obs_all is not None:
            value_s = self._critic_forward(obs_all, pred_feat=pred_feat)
        else:
            value_s = torch.zeros(obs_s.size(0), 1, device=obs_s.device)

        return value_s, action_dia, action_log_probs_dia, rhs_h_s

    def get_value_s(self, obs_all, pred_feat=None, h_comm_all=None):
        """Bootstrap 等场景需与训练一致：MAPPO_COMM 且 critic_use_comm 时传入 h_comm_all。"""
        return self._critic_forward(
            obs_all, h_comm_all=h_comm_all, pred_feat=pred_feat)

    def evaluate_action_s(self, obs_s, action_s, h, p_msk, n_msk, obs_all=None,
                          pred_feat=None):
        actor_feature_s, rhs_h_s = self.actor_base(obs_s, h, p_msk, n_msk)
        if self.action_head is not None:
            if self._actor_use_pred and pred_feat is not None:
                if pred_feat.dim() == 1:
                    pred_feat = pred_feat.unsqueeze(0)
                pred_feat = pred_feat.to(device=actor_feature_s.device, dtype=actor_feature_s.dtype)
                actor_feature_s = self.action_head(
                    torch.cat([actor_feature_s, pred_feat], dim=-1))
            else:
                actor_feature_s = self.action_head(actor_feature_s)
        dist_dia = self.dist_dia(actor_feature_s)
        action_log_probs_dia = dist_dia.log_probs(action_s)
        dist_entropy_dia = dist_dia.entropy().mean()

        if obs_all is not None:
            value_s = self._critic_forward(obs_all, pred_feat=pred_feat)
        else:
            value_s = torch.zeros(obs_s.size(0), 1, device=obs_s.device)

        return value_s, dist_entropy_dia, action_log_probs_dia, rhs_h_s

    def _prep_pred_feat_for_critic(self, pred_feat, obs_all):
        """与 obs_all 同 batch，供 Critic 拼接；None 时用零向量。"""
        B = obs_all.size(0)
        d = CONF.get('pred_out_d', 64)
        dev, dt = obs_all.device, obs_all.dtype
        if pred_feat is None:
            return torch.zeros(B, d, device=dev, dtype=dt)
        if pred_feat.dim() == 1:
            pred_feat = pred_feat.unsqueeze(0)
        if pred_feat.size(0) == 1 and B > 1:
            pred_feat = pred_feat.expand(B, -1)
        return pred_feat.to(device=dev, dtype=dt)

    def _critic_forward(self, obs_all, h_comm_all=None, pred_feat=None):
        kw = {}
        if h_comm_all is not None:
            kw['h_comm_all'] = h_comm_all
        if self._critic_use_pred:
            kw['pred_feat'] = self._prep_pred_feat_for_critic(pred_feat, obs_all)
        return self.critic(obs_all, **kw)

    # ---------- MAPPO_COMM 专用接口 ----------
    def get_z(self, obs_s, h, last_action):
        """Phase 1: obs → CNN → concat last_action → embed → GRU → z"""
        return self.actor_base(obs_s, h, last_action)

    def get_action_from_z(self, x_embed, z, h_comm, obs_all=None, h_comm_all=None,
                          pred_feat=None, comm_alpha=1.0):
        """Phase 3: [z, h_comm (, pred_feat)] → action_head → DiagGaussian → action

        h_comm_all: (B, uav_num, hgcn_out_dim) 当 critic_use_comm=True 时传入。
        pred_feat : (B, pred_out_d) 当 actor_use_pred=True 时传入。
        """
        # gate = torch.sigmoid(self.comm_gate)
        # h_comm = gate * h_comm
        # #
        # h_comm = h_comm * comm_alpha
        # if h_comm_all is not None:
        #     h_comm_all = h_comm_all * comm_alpha

        parts = []
        if self._cat_fe_cnn256 or self._cat_fe_mlp:
            parts.append(x_embed)
        if self._cat_fe_gru:
            parts.append(z)
        if self._actor_use_pred and pred_feat is not None:
            parts.append(pred_feat)
        parts.append(h_comm)
        zh = torch.cat(parts, dim=-1)
        actor_feature = self.action_head(zh)
        dist_dia = self.dist_dia(actor_feature)
        action_dia = dist_dia.sample()
        action_log_probs_dia = dist_dia.log_probs(action_dia)
        if obs_all is not None:
            value_s = self._critic_forward(
                obs_all, h_comm_all=h_comm_all, pred_feat=pred_feat)
        else:
            value_s = torch.zeros(z.size(0), 1, device=z.device)
        return value_s, action_dia, action_log_probs_dia

    def evaluate_from_z(self, x_embed, z, h_comm, action_s, obs_all=None, h_comm_all=None,
                        pred_feat=None, comm_alpha=1.0):
        """训练时: [z, h_comm (, pred_feat)] → action_head → DiagGaussian → log_prob & entropy

        h_comm_all: (B, uav_num, hgcn_out_dim) 当 critic_use_comm=True 时传入。
        pred_feat : (B, pred_out_d) 当 actor_use_pred=True 时传入（从 buffer 取出，已 detach）。
        """
        # gate = torch.sigmoid(self.comm_gate)
        # h_comm = gate * h_comm
        # #
        # h_comm = h_comm * comm_alpha
        # if h_comm_all is not None:
        #     h_comm_all = h_comm_all * comm_alpha

        parts = []
        if self._cat_fe_cnn256 or self._cat_fe_mlp:
            parts.append(x_embed)
        if self._cat_fe_gru:
            parts.append(z)
        if self._actor_use_pred and pred_feat is not None:
            parts.append(pred_feat)
        parts.append(h_comm)
        zh = torch.cat(parts, dim=-1)
        actor_feature = self.action_head(zh)
        dist_dia = self.dist_dia(actor_feature)
        action_log_probs_dia = dist_dia.log_probs(action_s)
        dist_entropy_dia = dist_dia.entropy().mean()
        if obs_all is not None:
            value_s = self._critic_forward(
                obs_all, h_comm_all=h_comm_all, pred_feat=pred_feat)
        else:
            value_s = torch.zeros(z.size(0), 1, device=z.device)
        return value_s, dist_entropy_dia, action_log_probs_dia


class CentralizedCritic_MLP(nn.Module):
    """集中式 Critic：输入所有智能体观测拼接，展平后经两层 MLP 输出 V 值。

    当 pred_feat_dim > 0 时：仅对观测展平向量做 MLP，再与 pred_feat 拼接后送入 value_head。
    """

    def __init__(self, uav_num, hidden_size, pred_feat_dim=0):
        super(CentralizedCritic_MLP, self).__init__()
        self.uav_num = uav_num
        self.pred_feat_dim = int(pred_feat_dim)
        obs_in_dim = (CONF['obs_shape'][0] * CONF['obs_shape'][1] * CONF['obs_shape'][2]
                      * uav_num)

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.mlp = nn.Sequential(
            init_(nn.Linear(obs_in_dim, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, hidden_size)),
            nn.ReLU(),
        )
        init_v = lambda m: init(m,
                                nn.init.orthogonal_,
                                lambda x: nn.init.constant_(x, 0))
        v_in = hidden_size + self.pred_feat_dim if self.pred_feat_dim > 0 else hidden_size
        self.value_head = nn.Sequential(
            init_(nn.Linear(v_in, hidden_size)),
            nn.ReLU(),
            init_v(nn.Linear(hidden_size, 1)),
        )
        self.train()

    def forward(self, obs_all, h_comm_all=None, pred_feat=None):
        """
        obs_all: (B, C*uav_num, H, W) 或 (B, uav_num, C, H, W)
        pred_feat: (B, pred_feat_dim)，pred_feat_dim>0 时由 Policy 传入或补零
        返回: (B, 1)
        """
        del h_comm_all
        x = obs_all.view(obs_all.size(0), -1)
        h = self.mlp(x)
        if self.pred_feat_dim > 0:
            if pred_feat is None:
                pred_feat = torch.zeros(
                    h.size(0), self.pred_feat_dim, device=h.device, dtype=h.dtype)
                print('pred_feat is None')
            else:
                if pred_feat.dim() == 1:
                    pred_feat = pred_feat.unsqueeze(0)
                pred_feat = pred_feat.to(device=h.device, dtype=h.dtype)
            h = torch.cat([h, pred_feat], dim=-1)
        return self.value_head(h)

class CentralizedCritic(nn.Module):
    """集中式 Critic：输入所有智能体观测拼接，经 CNN+MLP 输出 V 值。

    当 comm_dim > 0 时，额外接收所有智能体的 h_comm 拼接向量并与 CNN
    输出级联后送入 FC 层。

    CNN 展平后先经线性层压缩到与构造参数 hidden_size 相同维度，
    再与 comm / pred_feat 拼接。comm 侧先将 (B, uav_num*hgcn_out_dim) 线性压到 64 再拼接。
    """

    def __init__(self, uav_num, hidden_size, comm_dim=0, pred_feat_dim=0):
        super(CentralizedCritic, self).__init__()
        self.uav_num = uav_num
        self.comm_dim = comm_dim  # 展平后的原始维度；>0 时用 comm_proj 压到 _CRITIC_COMM_EMBED
        self.pred_feat_dim = int(pred_feat_dim)
        self._comm_cat_dim = (64 if comm_dim > 0 else 0)
        in_channels = CONF['obs_shape'][0] * uav_num

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels, 64, 8, stride=4, padding=4)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 5, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 4, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels,
                                CONF['obs_shape'][1], CONF['obs_shape'][2])
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.cnn_embed = nn.Sequential(
            init_(nn.Linear(cnn_out_dim, hidden_size)),
            nn.ReLU(),
        )

        if comm_dim > 0:
            self.comm_proj = nn.Sequential(
                init_(nn.Linear(comm_dim, self._comm_cat_dim)),
                nn.ReLU(),
            )
        else:
            self.comm_proj = None

        init_fc = lambda m: init(m,
                                 nn.init.orthogonal_,
                                 lambda x: nn.init.constant_(x, 0))

        fc_in = hidden_size + self._comm_cat_dim + self.pred_feat_dim
        if fc_in == hidden_size:
            self.fc = nn.Sequential(
                init_fc(nn.Linear(hidden_size, 1)),
            )
        else:
            self.fc = nn.Sequential(
                init_(nn.Linear(fc_in, hidden_size)),
                nn.ReLU(),
                init_fc(nn.Linear(hidden_size, 1)),
            )
        self.train()

    def forward(self, obs_all, h_comm_all=None, pred_feat=None):
        """
        obs_all:     (B, C*uav_num, H, W) 或 (B, uav_num, C, H, W)
        h_comm_all:  (B, uav_num, hgcn_out_dim) 或 None；comm_dim>0 且为 None 时
                     拼接与 comm 分支同维的零向量，与 fc 输入维一致。
        pred_feat:   (B, pred_feat_dim)，pred_feat_dim>0 时由 Policy 传入或补零
        返回: (B, 1)
        """
        if obs_all.dim() == 5:
            obs_all = obs_all.view(obs_all.size(0), -1,
                                   obs_all.size(-2), obs_all.size(-1))
        x = self.cnn_embed(self.cnn(obs_all))
        if self.comm_dim > 0:
            if h_comm_all is not None:
                h_flat = h_comm_all.view(h_comm_all.size(0), -1)
                x = torch.cat([x, self.comm_proj(h_flat)], dim=-1)
            else:
                x = torch.cat([x,torch.zeros(x.size(0), self._comm_cat_dim, device=x.device, dtype=x.dtype)],dim=-1)
                print('h_comm_all is None')
        if self.pred_feat_dim > 0:
            if pred_feat is None:
                pred_feat = torch.zeros(
                    x.size(0), self.pred_feat_dim, device=x.device, dtype=x.dtype)
                print('pred_feat is None')
            else:
                if pred_feat.dim() == 1:
                    pred_feat = pred_feat.unsqueeze(0)
                pred_feat = pred_feat.to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, pred_feat], dim=-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Actor backbone 变体
# ---------------------------------------------------------------------------

class ActorBase(nn.Module):
    """MLP backbone：观测展开后经两层 MLP。"""

    def __init__(self, input_channel_num, hidden_size, uid):
        super(ActorBase, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.uid = uid
        self.hidden_size = hidden_size
        in_dim = input_channel_num * CONF['obs_shape'][1] * CONF['obs_shape'][2]
        self.mlp = nn.Sequential(
            init_(nn.Linear(in_dim, hidden_size*2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size*2, hidden_size)),
            nn.ReLU(),
        )
        self.train()

    def forward(self, obs_s, rhs_h_s, p_msk, n_msk):
        x = obs_s.view(obs_s.size(0), -1)
        actor_feature_s = self.mlp(x)
        batch_size = obs_s.size(0)
        rhs_hc_s = torch.zeros(batch_size, *CONF['M_size'],
                               device=obs_s.device, dtype=obs_s.dtype)
        return actor_feature_s, rhs_hc_s


class ActorCNN(nn.Module):
    """CNN backbone：观测经 CNN + 线性层。"""

    def __init__(self, input_channel_num, hidden_size, uid):
        super(ActorCNN, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.uid = uid
        self.hidden_size = hidden_size
        self.main = nn.Sequential(
            init_(nn.Conv2d(input_channel_num, 32, 8, stride=4, padding=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 5, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 4, stride=1, padding=1)),
            nn.ReLU(),
        )
        init_fc = lambda m: init(m,
                                 nn.init.orthogonal_,
                                 lambda x: nn.init.constant_(x, 0))
        self.fc = init_fc(nn.Linear(32 * 3 * 3, hidden_size))
        self.train()

    def forward(self, obs_s, rhs_h_s, p_msk, n_msk):
        x = self.main(obs_s)
        x = x.flatten(1)
        actor_feature_s = torch.relu(self.fc(x))
        batch_size = obs_s.size(0)
        rhs_hc_s = torch.zeros(batch_size, *CONF['M_size'],
                               device=obs_s.device, dtype=obs_s.dtype)
        return actor_feature_s, rhs_hc_s


class ActorCNN_GRU(nn.Module):
    """CNN+GRU backbone。"""

    def __init__(self, input_channel_num, hidden_size, uid):
        super(ActorCNN_GRU, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.uid = uid
        self.hidden_size = hidden_size
        self.main = nn.Sequential(
            init_(nn.Conv2d(input_channel_num, 32, 8, stride=4, padding=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 5, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 4, stride=1, padding=1)),
            nn.ReLU(),
        )
        self.gru_input_size = int(CONF['hidden_size_gru'])
        self.gru_hidden_size = int(torch.tensor(CONF['M_size']).prod().item())
        self.gru = nn.GRUCell(self.gru_input_size, self.gru_hidden_size)
        init_fc = lambda m: init(m,
                                 nn.init.orthogonal_,
                                 lambda x: nn.init.constant_(x, 0))
        self.embed_fc = init_fc(nn.Linear(32 * 3 * 3, self.gru_input_size))
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.actor_fc = init_(nn.Linear(self.gru_hidden_size, hidden_size))
        self.train()

    def forward(self, obs_s, rhs_h_s, p_msk, n_msk):
        x = self.main(obs_s)
        x_flat = x.flatten(1)
        x_embed = self.embed_fc(x_flat)
        if x_embed.size(0) == rhs_h_s.size(0):
            h_flat = rhs_h_s.view(rhs_h_s.size(0), -1)
            h_new = self.gru(x_embed, h_flat)
            z = h_new
            rhs_hc_s = h_new.view(rhs_h_s.size(0), *CONF['M_size'])
        else:
            N = rhs_h_s.size(0)
            T = x_embed.size(0) // N
            x_embed_seq = x_embed.view(T, N, -1)
            h = rhs_h_s.view(N, -1)
            outputs = []
            for t in range(T):
                h = self.gru(x_embed_seq[t], h)
                outputs.append(h)
            z = torch.stack(outputs, dim=0).view(T * N, -1)
            rhs_hc_s = h.view(N, *CONF['M_size'])
        return z, rhs_hc_s

class ActorCOMM(nn.Module):
    """CNN+GRU backbone for MAPPO_COMM.

    obs → CNN → flatten → concat(last_action) → embed_fc → GRU → z
    z 直接输出（不经过 actor_fc），后续由 Policy.action_head 处理。
    """

    def __init__(self, input_channel_num, hidden_size, uid):
        super(ActorCOMM, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.uid = uid
        self.hidden_size = hidden_size
        self.cnn_out_dim = 32 * 3 * 3  # 288
        self.main = nn.Sequential(
            init_(nn.Conv2d(input_channel_num, 32, 8, stride=4, padding=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 5, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 32, 4, stride=1, padding=1)),
            nn.ReLU(),
        )
        self.gru_input_size = int(CONF['hidden_size_gru'])
        self.gru_hidden_size = int(torch.tensor(CONF['M_size']).prod().item())
        self.gru = nn.GRUCell(self.gru_input_size, self.gru_hidden_size)
        init_fc = lambda m: init(m,
                                 nn.init.orthogonal_,
                                 lambda x: nn.init.constant_(x, 0))
        self.embed_fc = init_fc(nn.Linear(self.cnn_out_dim + CONF['action_space'],
                                          self.gru_input_size))
        if CONF['cat_fe_mlp']==True:
            mlp_in_dim = input_channel_num * CONF['obs_shape'][1] * CONF['obs_shape'][2]
            self.mlp = nn.Sequential(
                init_(nn.Linear(mlp_in_dim, hidden_size * 2)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size * 2, hidden_size)),
                nn.ReLU(),
            )
        self.gru_fc = init_(nn.Linear(self.gru_input_size, self.gru_hidden_size)) ###fc替换GRU，256到128到分组和通信

        self.train()

    def forward(self, obs_s, rhs_h_s, last_action):
        """
        Args:
            obs_s:       (B, C, H, W)  或 (T*N, C, H, W) 训练时
            rhs_h_s:     (B, *M_size) 或 (N, *M_size)    GRU 初始隐藏态
            last_action: (B, action_space) 或 (T*N, action_space)
        Returns:
            z:       (B, z_dim) 或 (T*N, z_dim)    GRU 输出特征
            rhs_hc_s:(B, *M_size) 或 (N, *M_size)  更新后的隐藏态
        """
        x = self.main(obs_s)
        x_flat = x.flatten(1)
        x_cat = torch.cat([x_flat, last_action], dim=-1)
        x_embed = self.embed_fc(x_cat)

        if CONF['use_gru'] == True:
            if x_embed.size(0) == rhs_h_s.size(0):
                h_flat = rhs_h_s.view(rhs_h_s.size(0), -1)
                h_new = self.gru(x_embed, h_flat)
                z = h_new
                rhs_hc_s = h_new.view(rhs_h_s.size(0), *CONF['M_size'])
            else:
                N = rhs_h_s.size(0)
                T = x_embed.size(0) // N
                x_embed_seq = x_embed.view(T, N, -1)
                h = rhs_h_s.view(N, -1)
                outputs = []
                for t in range(T):
                    h = self.gru(x_embed_seq[t], h)
                    outputs.append(h)
                z = torch.stack(outputs, dim=0).view(T * N, -1)
                rhs_hc_s = h.view(N, *CONF['M_size'])
        else:
            z = x_embed   #256输入通信
            batch_size = obs_s.size(0)
            rhs_hc_s = torch.zeros(batch_size, *CONF['M_size'], device=obs_s.device, dtype=obs_s.dtype)

        if CONF['cat_fe_mlp'] == True:
            x1 = obs_s.view(obs_s.size(0), -1)
            x_embed = self.mlp(x1)

        return x_embed, z, rhs_hc_s ##x_embed在cat_fe_cnn256或者cat_fe_mlp时才拼接


# ---------------------------------------------------------------------------
# HGCN 通信模块（超图卷积 + 动态谱聚类）
# ---------------------------------------------------------------------------

###注意力
# class HGCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads=1, node2edge_weight=False):
#         super(HGCNLayer, self).__init__()
#
#         # 保留接口，避免外层 HGCN 传参时报错；本版本不使用 num_heads/node2edge_weight
#         self.num_heads = num_heads
#         self.node2edge_weight = node2edge_weight
#
#         # HyperAttnLayer 中的共享特征投影 P
#         self.proj = nn.Linear(in_dim, in_dim, bias=False)
#
#         # 拼接注意力向量 a，对 [x_i P || e_m] 打分
#         self.attn_vec = nn.Parameter(torch.empty(2 * in_dim))
#         nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))
#
#         self.leaky_relu = nn.LeakyReLU(0.2)
#
#         # 输出映射
#         self.out_proj = nn.Linear(in_dim, out_dim, bias=False)
#         self.skip_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
#
#         self.attention_weights = None   # detach，供监控/可视化
#         self._att_w = None              # 带梯度，供损失回传
#
#     def forward(self, x, hypergraph):
#         """
#         HyperAttn-style 超图注意力前向：
#         节点→超边均值聚合 + 超边→节点拼接注意力分发。
#
#         x:          (B, n_agents, in_dim)
#         hypergraph: (B, n_groups, n_agents)
#                     H[b,e,v]=1 表示节点 v 属于超边 e
#
#         return:     (B, n_agents, out_dim)
#         """
#         B, n_groups, n_agents = hypergraph.shape
#
#         # ===== Step 1: 节点特征共享投影 =====
#         # X' = X P
#         x_proj = self.proj(x)  # (B, n_agents, in_dim)
#
#         # ===== Step 2: 节点 → 超边均值聚合 =====
#         # E = D_e^{-1} H X'
#         edge_degree = hypergraph.sum(dim=-1).clamp(min=1e-6)
#         # (B, n_groups)
#
#         edge_feat = torch.bmm(hypergraph, x_proj)
#         # (B, n_groups, in_dim)
#
#         edge_feat = edge_feat * edge_degree.pow(-1.0).unsqueeze(-1)
#         # (B, n_groups, in_dim)
#
#         # ===== Step 3: 计算节点-超边拼接注意力 =====
#         # 对节点 u 和超边 m 计算 a^T [x_u' || e_m]
#         x_expand = x_proj.unsqueeze(1).expand(B, n_groups, n_agents, -1)
#         # (B, n_groups, n_agents, in_dim)
#
#         edge_expand = edge_feat.unsqueeze(2).expand(B, n_groups, n_agents, -1)
#         # (B, n_groups, n_agents, in_dim)
#
#         concat = torch.cat([x_expand, edge_expand], dim=-1)
#         # (B, n_groups, n_agents, 2*in_dim)
#
#         scores = (concat * self.attn_vec).sum(dim=-1)
#         # (B, n_groups, n_agents)
#
#         scores = self.leaky_relu(scores)
#
#         # 只允许节点关注自己所属的超边
#         scores = scores.masked_fill(hypergraph == 0, -1e9)
#
#         # 对每个节点，在其所属超边维度上做 softmax
#         beta = torch.softmax(scores.transpose(1, 2), dim=-1)
#         # (B, n_agents, n_groups)
#
#         beta = torch.nan_to_num(beta, nan=0.0)
#
#         self.attention_weights = beta.detach()
#         self._att_w = beta
#
#         # ===== Step 4: 超边 → 节点注意力聚合 =====
#         # x_att[u] = sum_m beta[u,m] * edge_feat[m]
#         x_att = torch.bmm(beta, edge_feat)
#         # (B, n_agents, in_dim)
#
#         # ===== Step 5: 输出映射 =====
#         # X_out = F.relu(self.out_proj(x_att))  # (B, N, F_out)
#         X_out = F.relu(self.out_proj(x_att)+self.skip_proj(x))  # (B, N, F_out)
#         return X_out

####原双分支
class HGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2, node2edge_weight=False):
        super(HGCNLayer, self).__init__()
        self.feature_transform = nn.Linear(in_dim, in_dim)
        self.node2edge_weight = node2edge_weight
        self._node2edge_att_w = None   # 带梯度，供损失回传
        if node2edge_weight:
            # 节点→超边注意力：a^T [x_v || x_e_init]，输入 2*in_dim，输出标量
            self.node2edge_attn = nn.Linear(2 * in_dim, 1, bias=False)
            self.leaky_relu = nn.LeakyReLU(0.2)
            # 可学习的残差融合门控 (初始化为 0, sigmoid 后为 0.5)
            self.edge_fuse_gate = nn.Parameter(torch.tensor(0.0))
        # 超边→节点注意力（query=节点，key/value=超边特征）
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.linear = nn.Linear(in_dim, out_dim)
        self.attention_weights = None   # 超边→节点注意力权重（detach，供监控/可视化）
        self._att_w = None              # 超边→节点注意力权重（带梯度，供损失回传）

    def forward(self, x, hypergraph):
        """
        超图卷积前向：节点→超边聚合 + 超边→节点分发 + 组内注意力 + 残差融合。

        x:          (B, n_agents, in_dim)
        hypergraph: (B, n_groups, n_agents)  — 关联矩阵，H[b,e,v]=1 表示节点v属于超边e
        """
        B, n_agents, _ = x.shape
        n_groups = hypergraph.size(1)
        H_T = hypergraph.transpose(1, 2)                              # (B, n_agents, n_groups)

        # --- 归一化系数 ---
        D_inv_sqrt = (1.0 / hypergraph.sum(dim=1).clamp(min=1e-6)).sqrt()  # (B, n_agents)

        # --- 特征变换 ---
        x_norm = x * D_inv_sqrt.unsqueeze(-1)                         # D^{-1/2} 归一化
        x_t = F.relu(self.feature_transform(x_norm))                  # (B, n_agents, in_dim)

        # ===== 步骤 1：节点 → 超边（组内聚合）=====
        if self.node2edge_weight:
            # 超边初始均值特征（用于注意力参考）
            B_count = hypergraph.sum(dim=2).clamp(min=1e-6)           # (B, n_groups)
            edge_mean = torch.bmm(hypergraph, x_t) / B_count.unsqueeze(-1)
            # (B, n_groups, in_dim)

            # 广播拼接 [x_v || x_e] 计算每条超边内各节点的注意力分数
            x_expand    = x_t.unsqueeze(1).expand(B, n_groups, n_agents, -1)
            edge_expand = edge_mean.unsqueeze(2).expand(B, n_groups, n_agents, -1)
            scores = self.leaky_relu(
                self.node2edge_attn(
                    torch.cat([x_expand, edge_expand], dim=-1)
                )
            ).squeeze(-1)                                             # (B, n_groups, n_agents)

            # 掩码：不属于本超边的节点置 -inf，再 softmax
            scores = scores.masked_fill(hypergraph == 0, -1e9)
            alpha = torch.softmax(scores, dim=-1)                    # (B, n_groups, n_agents)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            self._node2edge_att_w = alpha                            # 带梯度

            # edge_feat = 均值（保底） + 注意力加权（拔高），参考 hgcn1 融合方案
            edge_attn = torch.bmm(alpha, x_t)                        # (B, n_groups, in_dim)
            # 使用可学习门控进行平滑融合
            gate = torch.sigmoid(self.edge_fuse_gate)
            edge_feat = (1 - gate) * edge_mean + gate * edge_attn
        else:
            # 原方案：均匀 B^{-1} 归一化聚合
            B_inv = 1.0 / hypergraph.sum(dim=2).clamp(min=1e-6)      # (B, n_groups)
            edge_feat = torch.bmm(hypergraph, x_t) * B_inv.unsqueeze(-1)
            # (B, n_groups, in_dim)

        # ===== 步骤 2：超边 → 节点（直接分发）=====
        node_agg = torch.bmm(H_T, edge_feat) * D_inv_sqrt.unsqueeze(-1)
        # (B, n_agents, in_dim)

        # ===== 步骤 3：超边 → 节点注意力（query=节点，key/value=超边特征）=====
        attn_mask_3d = (H_T == 0).float() * -1e9                     # (B, n_agents, n_groups)
        num_heads = self.attention.num_heads
        attn_mask = attn_mask_3d.unsqueeze(1).expand(
            B, num_heads, n_agents, n_groups
        ).reshape(B * num_heads, n_agents, n_groups)

        x_att, att_w = self.attention(
            x_t, edge_feat, edge_feat,
            attn_mask=attn_mask)
        self.attention_weights = att_w.detach()
        self._att_w = att_w

        # ===== 步骤 4：残差融合 + 输出 =====
        return F.relu(self.linear(x_att + node_agg))


class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, num_heads=1,
                 node2edge_weight=False):
        super(HGCN, self).__init__()
        self.layers = nn.ModuleList()
        kw = dict(num_heads=num_heads, node2edge_weight=node2edge_weight)
        self.layers.append(HGCNLayer(in_dim, hidden_dim, **kw))
        for _ in range(num_layers - 2):
            self.layers.append(HGCNLayer(hidden_dim, hidden_dim, **kw))
        self.layers.append(HGCNLayer(hidden_dim, out_dim, **kw))

    def forward(self, x, hypergraph):
        for layer in self.layers:
            x = layer(x, hypergraph)
        return x


class DynamicSpectralClustering:
    """基于 z 特征的谱聚类。"""

    def __init__(self, min_clusters, max_clusters, n_agents,
                 min_group_size=1, max_group_size=None):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_agents = n_agents
        self.min_group_size = int(max(1, min_group_size))
        # None 或 <=0 表示不限制上限（等价于 n_agents）
        self.max_group_size = max_group_size
        self.current_groups = None

    def _split_group(self, z, g, max_sz):
        """将过大组拆成若干子组，每组至多 max_sz（在特征空间上 KMeans 递归）。"""
        g = list(g)
        if len(g) <= max_sz:
            return [g]
        from sklearn.cluster import KMeans

        k = int(np.ceil(len(g) / max_sz))
        k = min(max(k, 2), len(g))
        sub = z[g]
        lab = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(sub)
        buckets = [[] for _ in range(k)]
        for a, lb in zip(g, lab):
            buckets[lb].append(a)
        out = []
        for part in buckets:
            if not part:
                continue
            out.extend(self._split_group(z, part, max_sz))
        return out

    def _split_large_groups(self, z_agents, groups, max_sz):
        if max_sz >= self.n_agents:
            return [list(x) for x in groups if x]
        out = []
        for g in groups:
            if not g:
                continue
            out.extend(self._split_group(z_agents, list(g), max_sz))
        return out if out else [list(range(self.n_agents))]

    def _merge_small_groups(self, z_agents, groups, min_sz):
        """反复将人数 < min_sz 的组合并到质心最近的邻组。"""
        groups = [list(g) for g in groups if g]
        if len(groups) <= 1:
            return groups
        while True:
            sizes = [len(g) for g in groups]
            if not sizes or min(sizes) >= min_sz:
                break
            i = int(np.argmin(sizes))
            if len(groups[i]) >= min_sz:
                break
            best_j, best_d = None, float('inf')
            ci = z_agents[groups[i]].mean(axis=0)
            for j, gj in enumerate(groups):
                if j == i:
                    continue
                cj = z_agents[gj].mean(axis=0)
                d = float(np.linalg.norm(ci - cj))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is None:
                break
            groups[best_j].extend(groups[i])
            del groups[i]
        return groups

    def _enforce_group_sizes(self, z_agents, groups):
        """对谱聚类结果做后处理：同时满足每组人数 [min_sz, max_sz] 与组数 [min_k, max_k]。

        两类约束存在矛盾时（例如 min_sz * min_k > n_agents），尽力而为：
        Phase 1：先满足人数约束（merge / split）；
        Phase 2：再调整组数（过多则合并最近对，过少则拆最大组）。
        """
        g = [list(x) for x in groups if x]
        if not g:
            return [list(range(self.n_agents))]

        max_eff = self.max_group_size
        max_sz = min(int(max_eff), self.n_agents) if (max_eff and max_eff > 0) else self.n_agents
        min_sz = max(1, min(self.min_group_size, self.n_agents))
        if min_sz > max_sz:
            min_sz = max_sz
        min_k = max(1, self.min_clusters)
        max_k = max(1, self.max_clusters)
        if min_k > max_k:
            min_k = max_k

        # 快速路径：两类约束均已满足
        if (min_k <= len(g) <= max_k
                and all(min_sz <= len(grp) <= max_sz for grp in g)):
            return g

        # ---- Phase 1：人数约束 ----
        for _ in range(30):
            prev = [tuple(sorted(x)) for x in g]
            g = self._merge_small_groups(z_agents, g, min_sz)
            g = self._split_large_groups(z_agents, g, max_sz)
            if [tuple(sorted(x)) for x in g] == prev:
                break

        # ---- Phase 2a：组数过多 → 合并最近组对（优先不违反 max_sz）----
        while len(g) > max_k and len(g) > 1:
            best_i, best_j, best_d = None, None, float('inf')
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if len(g[i]) + len(g[j]) > max_sz:
                        continue   # 合并后超 max_sz，跳过
                    ci = z_agents[g[i]].mean(axis=0)
                    cj = z_agents[g[j]].mean(axis=0)
                    d = float(np.linalg.norm(ci - cj))
                    if d < best_d:
                        best_d, best_i, best_j = d, i, j
            if best_i is None:
                # 找不到满足 max_sz 的对，放宽约束取距离最近对
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        ci = z_agents[g[i]].mean(axis=0)
                        cj = z_agents[g[j]].mean(axis=0)
                        d = float(np.linalg.norm(ci - cj))
                        if d < best_d:
                            best_d, best_i, best_j = d, i, j
            if best_i is None:
                break
            g[best_i] = g[best_i] + g[best_j]
            del g[best_j]

        # ---- Phase 2b：组数过少 → 拆最大组（拆成两半），直到满足 min_k ----
        while len(g) < min_k:
            i_max = int(np.argmax([len(x) for x in g]))
            if len(g[i_max]) < 2:
                break   # 每组只有 1 人，无法再拆
            # 拆目标大小：让拆出的子组不超过 max_sz，且能真正拆成 ≥2 份
            split_sz = max(1, min(max_sz, len(g[i_max]) - 1))
            sub = self._split_group(z_agents, g[i_max], split_sz)
            if len(sub) <= 1:
                break
            g = g[:i_max] + sub + g[i_max + 1:]

        return g if g else [list(range(self.n_agents))]

    def cluster(self, z_agents):
        """
        z_agents: (n_agents, z_dim)  numpy array
        """
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score

        # 无人机数太少，无法有效分组，退回全体一组
        if self.n_agents <= 2:
            return [list(range(self.n_agents))]

        # 自适应近邻数：约取智能体数的一半，上限不超过 n_agents-1
        n_neighbors = max(1, min(self.n_agents // 2, self.n_agents - 1))

        best_score = -1
        best_labels = None
        for k in range(self.min_clusters, self.max_clusters + 1):
            sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
                                    n_neighbors=n_neighbors + 1)
            labels = sc.fit_predict(z_agents)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(z_agents, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
        if best_labels is None:
            best_labels = np.zeros(self.n_agents, dtype=int)

        n_clusters = int(best_labels.max()) + 1
        groups = [[] for _ in range(n_clusters)]
        for i in range(self.n_agents):
            groups[best_labels[i]].append(i)
        groups = [g for g in groups if g]
        return self._enforce_group_sizes(z_agents, groups)

    def update_groups(self, z_agents, stability_threshold):
        new_groups = self.cluster(z_agents)
        if self.current_groups is None:
            self.current_groups = new_groups
            return True, new_groups

        moved = self._count_moved(self.current_groups, new_groups)
        if moved == 0:
            return False, self.current_groups
        if moved / self.n_agents < stability_threshold:
            return False, self.current_groups

        self.current_groups = new_groups
        return True, new_groups

    def _count_moved(self, old_groups, new_groups):
        old_map = {a: frozenset(g) for g in old_groups for a in g}
        new_map = {a: frozenset(g) for g in new_groups for a in g}
        return sum(1 for a in range(self.n_agents) if old_map.get(a) != new_map.get(a))

    def get_current_groups(self):
        if self.current_groups is None:
            return [list(range(self.n_agents))]
        return self.current_groups


class CommModule(nn.Module):
    """通信模块：动态谱聚类分组 + HGCN 超图卷积。

    所有智能体共享同一个 CommModule 实例。
    """

    def __init__(self, n_agents, z_dim, hgcn_hidden_dim, hgcn_out_dim,
                 hgcn_num_layers, min_clusters, max_clusters,
                 stability_threshold, clustering_interval,
                 min_group_size=1, max_group_size=None,
                 hgcn_mha_num_heads=1, node2edge_weight=False):
        super(CommModule, self).__init__()
        self.n_agents = n_agents
        self.z_dim = z_dim
        # z 直接进入 HGCN （in_dim=z_dim）
        # z(256) -> proj(128) 再进入 HGCN
        self.comm_in_dim = hgcn_hidden_dim * 2 #hgcn_hidden_dim*2=128
        self.in_proj = nn.Linear(z_dim, self.comm_in_dim)
        self.norm = nn.LayerNorm(self.comm_in_dim)
        self.out_norm = nn.LayerNorm(hgcn_out_dim)
        self.hgcn = HGCN(
            self.comm_in_dim, hgcn_hidden_dim, hgcn_out_dim, hgcn_num_layers,
            num_heads=hgcn_mha_num_heads, node2edge_weight=node2edge_weight) #128输入/64输入
        # self.comm_gate = nn.Sequential(
        #     nn.Linear(self.comm_in_dim + hgcn_out_dim, hgcn_out_dim),
        #     nn.Sigmoid()
        # )
        self.clustering = DynamicSpectralClustering(
            min_clusters, max_clusters, n_agents,
            min_group_size=min_group_size, max_group_size=max_group_size)
        self.clustering_interval = clustering_interval
        self.stability_threshold = stability_threshold
        self._groups = [list(range(n_agents))]
        self._steps_since_cluster = 0
        # 本次 forward 是否发生了分组变化（供外部日志使用）
        self._groups_changed = False

    def forward(self, z_all, update_groups=False, force_cluster=False):
        """
        z_all: (B, n_agents, z_dim)
        update_groups: 为 True 时按 clustering_interval 尝试重聚类。
        force_cluster: 为 True 时本步立刻聚类并重置间隔计数（用于每回合时隙 0）。

        Returns:
            h_comm       (B, n_agents, hgcn_out_dim)
            attention_reg scalar tensor，KL 散度正则化损失（带梯度，可直接用于回传）
        """
        z_all = self.in_proj(z_all)  # 128输入通信，但是如果detach输入通信，那这个映射不会被更新了，所以放到通信里。如果不加detach
        z_all = self.norm(z_all)
        z_all = F.gelu(z_all)

        self._groups_changed = False
        if update_groups:
            if force_cluster:
                z_np = z_all[0].detach().cpu().numpy()
                changed, new_groups = self.clustering.update_groups(
                    z_np, self.stability_threshold)
                if changed:
                    self._groups = new_groups
                    self._groups_changed = True
                self._steps_since_cluster = 0
            else:
                self._steps_since_cluster += 1
                if self._steps_since_cluster >= self.clustering_interval:
                    z_np = z_all[0].detach().cpu().numpy()
                    changed, new_groups = self.clustering.update_groups(
                        z_np, self.stability_threshold)
                    if changed:
                        self._groups = new_groups
                        self._groups_changed = True
                    self._steps_since_cluster = 0

        B = z_all.size(0)
        device = z_all.device
        H = self._create_hypergraph(B, device)
        # z_all = self.in_proj(z_all)
        h_comm = self.hgcn(z_all, H)
        h_comm = self.out_norm(h_comm) ###输出正则化
        # gate = self.comm_gate(torch.cat([z_all, h_comm], dim=-1))
        # h_comm = gate * h_comm

        # 计算注意力正则化损失（带梯度，att_w 来自本次 forward，无需额外显存）
        attention_reg = torch.tensor(0.0, device=device)
        # for layer in self.hgcn.layers:
        #     if hasattr(layer, '_att_w') and layer._att_w is not None:
        #         w_soft = torch.softmax(layer._att_w + 1e-8, dim=-1)
        #         uniform = torch.ones_like(w_soft) / w_soft.size(-1)
        #         attention_reg = attention_reg + F.kl_div(
        #             w_soft.log(), uniform, reduction='batchmean')

        return h_comm, attention_reg

    def _create_hypergraph(self, batch_size, device):
        """构建关联矩阵 H，shape (batch_size, n_groups, n_agents)。
        用 expand 避免重复分配显存：H_single(1,G,N) 在 batch 维度零拷贝展开。
        """
        n_groups = len(self._groups)
        H_single = torch.zeros(1, n_groups, self.n_agents, device=device)
        for i, group in enumerate(self._groups):
            H_single[0, i, group] = 1.0
        # expand 是零拷贝视图；torch.bmm 在 PyTorch≥1.10 正确处理 stride-0 batch 维
        return H_single.expand(batch_size, -1, -1)

    def get_groups(self):
        return [g[:] for g in self._groups]

    def get_group_labels(self):
        """将分组转换为标签数组，如 [[0,1],[2,3],[4]] → [0,0,1,1,2]。"""
        labels = [0] * self.n_agents
        for gid, members in enumerate(self._groups):
            for agent in members:
                labels[agent] = gid
        return labels

    def set_groups(self, groups):
        self._groups = [g[:] for g in groups]

    @staticmethod
    def labels_to_groups(labels):
        """将标签数组转换回分组列表，如 [0,0,1,1,2] → [[0,1],[2,3],[4]]。"""
        n_groups = int(max(labels)) + 1
        groups = [[] for _ in range(n_groups)]
        for agent, gid in enumerate(labels):
            groups[int(gid)].append(agent)
        return [g for g in groups if g]

    def get_attention_weights(self):
        """返回 HGCN 各层的注意力权重列表。"""
        weights = []
        for layer in self.hgcn.layers:
            if layer.attention_weights is not None:
                weights.append(layer.attention_weights)
        return weights if weights else None
