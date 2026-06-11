"""行人时空预测模块 (PredModule)

数据流（当前默认）
----------------
1) 使用 peo_pos.npy，shape (max_step+1, P, 3)，与 reset + 100 步 env 对齐。
2) 主进程 pretrain_pred_module：每 epoch 用全部 N=max_step 个滑动窗口（对应 100 个 RL 时隙）算损失并更新。
3) compute_pred_feat_timeseries：一次前向得到 (max_step, d_pred)，写入 log 目录 pred_feat_timeseries.npy。
4) subp 每时隙 step_id 直接索引该数组，不再跑 PredModule。
"""
import math

from .conf import *
from .C2I import UniMobI2CLoss, UniMobC2ILoss


# ─────────────────────────────────────────────────────────────────────────────
# 离线预处理
# ─────────────────────────────────────────────────────────────────────────────

def compute_peo_region_id(peo_pos, field_length, K_cells):
    """将行人坐标映射到区域 ID（行主序展开）。

    Parameters
    ----------
    peo_pos     : (T, P, >=2) float，单位 meter
    field_length: [W, H]      float，单位 meter
    K_cells     : int，每维度格子数；总区域 K = K_cells²

    Returns
    -------
    peo_region_id : (T, P) int64，取值 [0, K_cells² - 1]
    """
    W, H = field_length[0], field_length[1]
    x = np.clip((peo_pos[:, :, 0] / W * K_cells).astype(np.int64), 0, K_cells - 1)
    y = np.clip((peo_pos[:, :, 1] / H * K_cells).astype(np.int64), 0, K_cells - 1)
    return x * K_cells + y           # (T, P)


def compute_region_stats(peo_region_id, K):
    """由行人区域 ID 序列计算区域统计量。

    Parameters
    ----------
    peo_region_id : (T, P) int64
    K             : int，总区域数

    Returns
    -------
    region_stats : (T, K, 3) float32，各维含义 [count, inflow, outflow]
    """
    T, P = peo_region_id.shape
    stats = np.zeros((T, K, 3), dtype=np.float32)
    for t in range(T):
        np.add.at(stats[t, :, 0], peo_region_id[t], 1.0)
    for t in range(1, T):
        moved = peo_region_id[t] != peo_region_id[t - 1]
        np.add.at(stats[t, :, 1], peo_region_id[t,  moved], 1.0)   # inflow
        np.add.at(stats[t, :, 2], peo_region_id[t - 1, moved], 1.0)  # outflow
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 在线数据窗口缓冲（subp 内使用）
# ─────────────────────────────────────────────────────────────────────────────

class PredDataBuffer:
    """维护过去 L+1 时隙的行人区域数据，供 PredModule 推理使用。"""

    def __init__(self, L, P, K, field_length, K_cells):
        self.L = L
        self.P = P
        self.K = K
        self.K_cells = K_cells
        self.W, self.H = field_length[0], field_length[1]

        self._peo_ids  = np.zeros((L + 1, P), dtype=np.int64)     # 滑动窗口
        self._reg_stat = np.zeros((L + 1, K, 3), dtype=np.float32)
        self._t_ids    = np.zeros(L + 1, dtype=np.int64)
        self._filled   = 0

    def update(self, peo_pos, step_id):
        """更新缓冲区，推入当前时隙的行人位置。

        peo_pos  : (P, 2) float，单位 meter
        step_id  : int，当前时隙索引
        """
        # 向左滑动，腾出最后一槽
        self._peo_ids  = np.roll(self._peo_ids,  -1, axis=0)
        self._reg_stat = np.roll(self._reg_stat, -1, axis=0)
        self._t_ids    = np.roll(self._t_ids,    -1)

        # 计算当前时隙区域 ID
        cx = np.clip((peo_pos[:, 0] / self.W * self.K_cells).astype(np.int64), 0, self.K_cells - 1)
        cy = np.clip((peo_pos[:, 1] / self.H * self.K_cells).astype(np.int64), 0, self.K_cells - 1)
        curr_ids = cx * self.K_cells + cy              # (P,)

        self._peo_ids[-1]  = curr_ids
        self._t_ids[-1]    = step_id
        self._filled       = min(self._filled + 1, self.L + 1)

        # 计算当前时隙区域统计量
        cur_stat = np.zeros((self.K, 3), dtype=np.float32)
        np.add.at(cur_stat[:, 0], curr_ids, 1.0)
        if self._filled > 1:
            prev_ids = self._peo_ids[-2]
            moved = curr_ids != prev_ids
            np.add.at(cur_stat[:, 1], curr_ids[moved],  1.0)   # inflow
            np.add.at(cur_stat[:, 2], prev_ids[moved],  1.0)   # outflow
        self._reg_stat[-1] = cur_stat

    def get_tensors(self, device='cpu'):
        """返回 PredModule.forward 所需的张量（batch=1）。

        Returns
        -------
        peo_id_seq  : (1, L+1, P)     LongTensor
        reg_seq     : (1, L+1, K, 3)  FloatTensor
        time_ids    : (1, L+1)         LongTensor
        cur_count   : (1, K)           FloatTensor
        """
        peo_id_seq = torch.from_numpy(self._peo_ids ).unsqueeze(0).to(device)     # (1,L+1,P)
        reg_seq    = torch.from_numpy(self._reg_stat).unsqueeze(0).to(device)     # (1,L+1,K,3)
        time_ids   = torch.from_numpy(self._t_ids   ).unsqueeze(0).to(device)     # (1,L+1)
        cur_count  = reg_seq[:, -1, :, 0]                                         # (1,K)
        return peo_id_seq, reg_seq, time_ids, cur_count


# ─────────────────────────────────────────────────────────────────────────────
# GCRU（图卷积 + GRU）
# ─────────────────────────────────────────────────────────────────────────────

class GCRUCell(nn.Module):
    """空间注意力 + GRU 门控的单时间步单元。

    空间维度：多头自注意力（等价于全连接图的注意力传播）。
    时间维度：GRU 门控更新隐藏状态。
    """
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.spatial = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.Wz = nn.Linear(2 * d, d)
        self.Wr = nn.Linear(2 * d, d)
        self.Wh = nn.Linear(2 * d, d)

    def forward(self, x, h):
        """
        x, h : (B, N, d)
        Returns h_new : (B, N, d)
        """
        xs, _ = self.spatial(x, x, x)           # 空间聚合
        xs = self.norm(xs + x)  # 残差 + norm
        xh    = torch.cat([xs, h], dim=-1)
        z     = torch.sigmoid(self.Wz(xh))
        r     = torch.sigmoid(self.Wr(xh))
        h_new = torch.tanh(self.Wh(torch.cat([xs, r * h], dim=-1)))
        return (1 - z) * h + z * h_new


class GCRU(nn.Module):
    """多时间步 GCRU：在 (L+1) 个时隙上循环，输出最后时隙的节点特征。"""

    def __init__(self, in_d, hidden_d, n_heads=4):
        super().__init__()
        self.in_proj  = nn.Linear(in_d, hidden_d)
        self.cell     = GCRUCell(hidden_d, n_heads)
        self.hidden_d = hidden_d

    def forward(self, x_seq):
        """
        x_seq : (B, Lp1, N, in_d)
        Returns : (B, N, hidden_d)
        """
        B, Lp1, N, _ = x_seq.shape
        x = self.in_proj(x_seq)                              # (B, Lp1, N, hidden_d)
        h = torch.zeros(B, N, self.hidden_d, device=x.device, dtype=x.dtype)
        for t in range(Lp1):
            h = self.cell(x[:, t], h)
        return h                                              # (B, N, hidden_d)


# ─────────────────────────────────────────────────────────────────────────────
# PredModule
# ─────────────────────────────────────────────────────────────────────────────

class PredModule(nn.Module):
    """行人-区域时空预测模块。

    输入（当前时隙 t 及过去 L 时隙）
    ---------------------------------
    peo_id_seq  : (B, L+1, P)        行人区域 ID（LongTensor）
    reg_seq     : (B, L+1, K, 3)     区域 [count, inflow, outflow]（FloatTensor；reg_in 另拼接时间嵌入与固定区域 ID 嵌入）
    time_ids    : (B, L+1)            时间索引（LongTensor）
    cur_count   : (B, K)              当前时隙区域人数（FloatTensor）

    输出
    ----
    pred_feat   : (B, d_pred)         供 Actor 动作头使用的预测特征
    peo_logits  : (B, P, K)           行人区域预测 logits（用于交叉熵损失）
    flow_pred   : (B, K, 2)           区域入流/出流预测，直接预测归一化的（与 reg_in 输入尺度一致）
    peo_feat    : (B, P, d)           I2C/C2I 用：见 CONF['i2c_c2i_use_normalize']（True：投影+L2；False：GCRU 原特征）
    reg_feat    : (B, K, d)           同上

    T_max       : 时间嵌入 nn.Embedding 行数，须大于窗口内任意 time_id；训练侧传入
                  peo_pos 时间维长度，即 max_step+1（与 ENV_CONF 一致）。
    """

    def __init__(self, P, K, d=64, d_pred=64, L=5, T_max=101, n_heads=2):
        super().__init__()
        self.P = P
        self.K = K
        self.d = d
        self.L = L

        # ── 嵌入层 ─────────────────────────────────────────────────────────
        self.time_embed    = nn.Embedding(T_max, d // 2)
        self.reg_id_embed  = nn.Embedding(K, d // 2)      # 行人所在区域 / 区域行 k 的固定 ID（共用表）

        self.peo_in = nn.Linear(d, d)                     # (t_emb||region_emb) → d
        self.reg_in = nn.Linear(d // 2 + d // 2 + 3, d)  # time || region_id_emb || stats → d

        # ── 时空编码（GCRU）──────────────────────────────────────────────
        self.peo_gcru = GCRU(d, d, n_heads)
        self.reg_gcru = GCRU(d, d, n_heads)

        # ── I2C/C2I：可选投影 + L2（i2c_c2i_use_normalize=False 时不创建、不用）──
        self._i2c_c2i_use_normalize = bool(CONF.get('i2c_c2i_use_normalize', True))
        if self._i2c_c2i_use_normalize:
            self.peo_align_proj = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            self.reg_align_proj = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        else:
            self.peo_align_proj = None
            self.reg_align_proj = None

        # ── 交叉注意力（Pre-LN + 输入门控残差；见 forward 注释）──────────
        self.norm_peo = nn.LayerNorm(d)
        self.norm_reg = nn.LayerNorm(d)
        self.peo_cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.reg_cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.peo_cross_gate = nn.Linear(d, d)
        self.reg_cross_gate = nn.Linear(d, d)

        # ── 预测头 ───────────────────────────────────────────────────────
        self.peo_head  = nn.Linear(d, K)   # (B, P, d) → (B, P, K)
        self.flow_head = nn.Linear(d, 2)   # (B, K, d) → (B, K, 2)

        # ── pred_feat 聚合 MLP ──────────────────────────────────────────
        self.peo_pred_mlp = nn.Sequential(nn.Linear(K, d // 2), nn.ReLU())
        self.reg_pred_mlp = nn.Sequential(nn.Linear(3, d // 2), nn.ReLU())
        self._pred_feat_use_cross_attn = bool(CONF.get('pred_feat_use_cross_attn', False))
        if self._pred_feat_use_cross_attn:
            self.peo_cross_to_fp = nn.Sequential(
                nn.Linear(d, d // 2), nn.ReLU())
            self.reg_cross_to_fp = nn.Sequential(
                nn.Linear(d, d // 2), nn.ReLU())
        else:
            self.peo_cross_to_fp = None
            self.reg_cross_to_fp = None
        self.pred_feat_mlp = nn.Sequential(
            nn.Linear(d // 2 + d // 2, d_pred),
            nn.ReLU(),
            nn.Linear(d_pred, d_pred),
        )

    def forward(self, peo_id_seq, reg_seq, time_ids, cur_count):
        B, Lp1, P = peo_id_seq.shape
        K = reg_seq.shape[2]
        # 区域人数 / 入流 / 出流：单时隙、单区域上界均为 P，缩放到约 [0,1] 再送入 reg_in
        _reg_scale = max(float(self.P), 1.0)
        reg_seq_norm = reg_seq / _reg_scale

        # ── 嵌入 ────────────────────────────────────────────────────────
        t_emb    = self.time_embed(time_ids)                         # (B, L+1, d//2)
        rid_emb  = self.reg_id_embed(peo_id_seq)                     # (B, L+1, P, d//2)

        t_peo    = t_emb.unsqueeze(2).expand(-1, -1, P, -1)          # (B, L+1, P, d//2)
        peo_seq  = self.peo_in(torch.cat([t_peo, rid_emb], dim=-1))  # (B, L+1, P, d)

        t_reg = t_emb.unsqueeze(2).expand(-1, -1, K, -1)               # (B, L+1, K, d//2)
        reg_row_ids = torch.arange(K, device=reg_seq.device, dtype=torch.long).view(1, 1, K).expand(
            B, Lp1, K)
        reg_rid_emb = self.reg_id_embed(reg_row_ids)                   # (B, L+1, K, d//2)，第 k 列为区域 k
        reg_seq_ = self.reg_in(torch.cat([t_reg, reg_rid_emb, reg_seq_norm], dim=-1))  # (B, L+1, K, d)

        # ── GCRU ────────────────────────────────────────────────────────
        peo_feat = self.peo_gcru(peo_seq)   # (B, P, d)
        reg_feat = self.reg_gcru(reg_seq_)  # (B, K, d)

        # ── 交叉注意力（Pre-LN + 门控残差）───────────────────────────────
        # Pre-LN：稳定训练、与 Transformer 常见写法一致；门控由原始 GCRU 输出计算，
        # g=sigmoid(Linear(x))，输出为 x + g * MHA(LN·)，避免仅 g*out 时在 g→0 时丢光信息。
        peo_n = self.norm_peo(peo_feat)
        reg_n = self.norm_reg(reg_feat)

        # ── I2C/C2I 输入特征（主干仍始终用未投影的 GCRU 特征做交叉注意力）──────
        if self._i2c_c2i_use_normalize:
            peo_align = F.normalize(self.peo_align_proj(peo_feat), dim=-1, eps=1e-8)
            reg_align = F.normalize(self.reg_align_proj(reg_feat), dim=-1, eps=1e-8)
        else:
            peo_align = peo_n
            reg_align = reg_n

        peo_attn, _ = self.peo_cross(peo_n, reg_n, reg_n)              # (B, P, d)
        reg_attn, _ = self.reg_cross(reg_n, peo_n, peo_n)             # (B, K, d)
        g_peo = torch.sigmoid(self.peo_cross_gate(peo_feat))
        g_reg = torch.sigmoid(self.reg_cross_gate(reg_feat))
        peo_cross = peo_feat + g_peo * peo_attn
        reg_cross = reg_feat + g_reg * reg_attn

        # ── 预测输出 ────────────────────────────────────────────────────
        peo_logits = self.peo_head(peo_cross)    # (B, P, K)
        # flow_head：在归一化 reg 特征后直接回归归一化的入流/出流（不再对输出除 P）
        flow_pred = self.flow_head(reg_cross)   # (B, K, 2)，与 reg_seq/P 同尺度

        # 组装 K×3 供 pred_feat MLP：守恒 count_norm = cur_norm + in_norm - out_norm
        cur_count_norm = cur_count / _reg_scale
        count_pred_norm = cur_count_norm + flow_pred[..., 0] - flow_pred[..., 1]
        reg_pred_3 = torch.cat(
            [count_pred_norm.unsqueeze(-1), flow_pred], dim=-1)   # (B, K, 3)

        # ── pred_feat ───────────────────────────────────────────────────
        if self._pred_feat_use_cross_attn:
            # 与 peo_pred_mlp/reg_pred_mlp 同结构尺度：d→d//2+ReLU，再对 P / K 维 mean
            peo_fp = self.peo_cross_to_fp(peo_cross).mean(dim=1)       # (B, d//2)
            reg_fp = self.reg_cross_to_fp(reg_cross).mean(dim=1)       # (B, d//2)
        else:
            peo_fp = self.peo_pred_mlp(peo_logits).mean(dim=1)         # (B, d//2)
            reg_fp = self.reg_pred_mlp(reg_pred_3).mean(dim=1)         # (B, d//2)
        pred_feat = self.pred_feat_mlp(
            torch.cat([peo_fp, reg_fp], dim=-1))                        # (B, d_pred)

        return pred_feat, peo_logits, flow_pred, peo_align, reg_align


# ─────────────────────────────────────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────────────────────────────────────

def _chebyshev_region_dist_matrix(K_cells, device, dtype=torch.float32):
    """区域 ID = cx * K_cells + cy 时，两区域切比雪夫距离 max(|Δcx|, |Δcy|)。Shape (K, K)."""
    K = K_cells * K_cells
    ids = torch.arange(K, device=device, dtype=torch.long)
    cx = ids // K_cells
    cy = ids % K_cells
    dcx = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs()
    dcy = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs()
    return torch.maximum(dcx, dcy).to(dtype)


def mask_peo_logits_by_current_region(
        peo_logits, peo_id_current, K_cells, penalty_scale, max_dist=2.0):
    """对 (B,P,K) logits 做软掩码：与当前区域切比雪夫距离 > max_dist 的候选区减去惩罚。

    penalty_scale * relu(dist - max_dist)；横/竖/斜一步距离均为 1（切比雪夫）。
    """
    if penalty_scale <= 0:
        return peo_logits
    B, P, K = peo_logits.shape
    if K != K_cells * K_cells:
        return peo_logits
    dist_m = _chebyshev_region_dist_matrix(K_cells, peo_logits.device, peo_logits.dtype)
    cur = peo_id_current.long().clamp(0, K - 1)
    dist_bpk = dist_m[cur, :]
    md = float(max_dist)
    pen = penalty_scale * F.relu(dist_bpk - md)
    return peo_logits - pen


def pred_module_loss(peo_logits, flow_pred, peo_id_next, reg_stats_next, cur_count,
                     peo_feat=None, reg_feat=None, peo_id_current=None):
    """计算预测模块的联合损失。

    Parameters
    ----------
    peo_logits    : (B, P, K)   行人区域预测 logits
    flow_pred     : (B, K, 2)   归一化后的 [inflow, outflow]（≈ 真值 / P）
    peo_id_next   : (B, P)      LongTensor，下时隙真实行人区域 ID
    reg_stats_next: (B, K, 3)   FloatTensor，下时隙真实区域统计 [count, inflow, outflow]
    cur_count     : (B, K)      当前时隙区域人数（与 forward 一致），用于质量守恒人数预测
    peo_feat      : (B, P, d)   I2C/C2I 输入（与 forward 第 4 返回值一致；是否投影+L2 由 i2c_c2i_use_normalize）
    reg_feat      : (B, K, d)   同上
    peo_id_current: (B, P)      LongTensor，当前时隙真实行人区域 ID（窗口最后一帧）；
                                use_peo_logits_mask 时用于空间软掩码（切比雪夫距 > max_dist 惩罚）。

    ②③ 为归一化域 MSE；与 CE 量级可能差数量级，用 CONF['pred_loss_w_*'] 加权（见 conf.py）。

    Returns
    -------
    loss_total : scalar tensor
    loss_dict  : dict（各分项，供日志使用）
    """
    B, P, K = peo_logits.shape
    _scale = max(float(P), 1.0)

    logits_peo = peo_logits
    if (CONF.get('use_peo_logits_mask', False) and peo_id_current is not None):
        _kc = int(CONF.get('region_cell_num', 6))
        _ps = float(CONF.get('pred_peo_logits_mask_penalty', 1.0))
        _md = float(CONF.get('pred_peo_logits_mask_max_dist', 2))
        logits_peo = mask_peo_logits_by_current_region(
            peo_logits, peo_id_current.long(), _kc, _ps, max_dist=_md)

    # ① 行人区域预测：交叉熵（与汇聚人数均用 logits_peo）
    loss_ce = F.cross_entropy(
        logits_peo.view(B * P, K),
        peo_id_next.view(B * P))

    # ② flow_pred 已为归一化流量；守恒人数也在归一化域，与真值 /P 直接 MSE
    flow_true = reg_stats_next[:, :, 1:3] / _scale   # (B, K, 2)
    count_true_norm = reg_stats_next[:, :, 0] / _scale
    count_pred_norm = cur_count / _scale + flow_pred[..., 0] - flow_pred[..., 1]
    pred_stack = torch.cat([flow_pred, count_pred_norm.unsqueeze(-1)], dim=-1)
    true_stack = torch.cat([flow_true, count_true_norm.unsqueeze(-1)], dim=-1)
    loss_flow = F.mse_loss(pred_stack, true_stack)
    # # RMSE
    # rmse_raw = torch.sqrt(loss_flow) * _scale
    #
    # mae_norm = torch.abs(pred_stack - true_stack).mean()
    # mae_raw = mae_norm * _scale
    # print("RMSE:", rmse_raw.item())
    # print("MAE:", mae_raw.item())
    #
    # # 1. 展平
    # logits_flat = logits_peo.view(B * P, K)
    # labels_flat = peo_id_next.view(B * P)
    #
    # # 2. Top-1 Accuracy
    # pred_top1 = logits_flat.argmax(dim=-1)  # [B*P]
    # acc1 = (pred_top1 == labels_flat).float().mean()
    #
    # # 3. Top-3 Accuracy
    # top3 = torch.topk(logits_flat, 3, dim=-1).indices  # [B*P, 3]
    # acc3 = (top3 == labels_flat.unsqueeze(-1)).any(dim=-1).float().mean()
    #
    # print("Acc@1:", acc1.item())
    # print("Acc@3:", acc3.item())

    # ③ 行人区域概率汇聚人数 vs 真实人数：同尺度归一化后 MSE
    peo_prob  = torch.softmax(logits_peo, dim=-1)  # (B, P, K)
    agg_count = peo_prob.sum(dim=1)                # (B, K)
    loss_agg_count = F.mse_loss(agg_count / _scale, count_true_norm)

    w_ce = float(CONF.get('pred_loss_w_ce', 1.0))
    w_flow = float(CONF.get('pred_loss_w_flow', 1.0))
    w_agg_count = float(CONF.get('pred_loss_w_agg_count', 1.0))
    loss_total = w_ce * loss_ce + w_flow * loss_flow + w_agg_count * loss_agg_count

    loss_dict = {
        'ce': loss_ce.item(),
        'flow': loss_flow.item(),
        'agg_count': loss_agg_count.item(),
        'ce_w': (w_ce * loss_ce).item(),
        'flow_w': (w_flow * loss_flow).item(),
        'count_w': (w_agg_count * loss_agg_count).item(),
    }

    if (peo_feat is not None and reg_feat is not None
            and peo_id_current is not None):
        peo_id_cur = peo_id_current.long()
        loss_i2c = UniMobI2CLoss()(peo_feat, reg_feat, peo_id_cur)
        loss_c2i = UniMobC2ILoss(
            temperature=float(CONF.get('pred_c2i_temperature', 0.1)))(
            peo_feat, reg_feat, peo_id_cur)
        w_i2c = float(CONF.get('pred_loss_w_i2c', 1.0))
        w_c2i = float(CONF.get('pred_loss_w_c2i', 1.0))
        loss_total = loss_total + w_i2c * loss_i2c + w_c2i * loss_c2i
        loss_dict['i2c'] = loss_i2c.item()
        loss_dict['c2i'] = loss_c2i.item()
        loss_dict['i2c_w'] = (w_i2c * loss_i2c).item()
        loss_dict['c2i_w'] = (w_c2i * loss_c2i).item()

    return loss_total, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# 训练数据批次构建（主进程使用）
# ─────────────────────────────────────────────────────────────────────────────

def _build_pred_windows_numpy(peo_region_id, region_stats, L):
    """滑动窗口堆叠为 (N, L+1, ...)，N = T-1。"""
    T, P = peo_region_id.shape
    K    = region_stats.shape[1]
    Lp1  = L + 1
    N    = T - 1

    peo_id_arr = np.zeros((N, Lp1, P), dtype=np.int64)
    reg_arr    = np.zeros((N, Lp1, K, 3), dtype=np.float32)
    t_arr      = np.zeros((N, Lp1), dtype=np.int64)

    for i in range(N):
        t = i
        t_start = max(0, t - L)
        window = range(t_start, t + 1)
        wlen = len(window)
        pad = Lp1 - wlen
        for k, ts in enumerate(window):
            peo_id_arr[i, pad + k] = peo_region_id[ts]
            reg_arr[i, pad + k] = region_stats[ts]
            t_arr[i, pad + k] = ts
        if pad > 0:
            for k in range(pad):
                peo_id_arr[i, k] = peo_region_id[t_start]
                reg_arr[i, k] = region_stats[t_start]
                t_arr[i, k] = t_start

    cur_count_arr = region_stats[:T - 1, :, 0]
    return peo_id_arr, reg_arr, t_arr, cur_count_arr


def build_pred_window_inputs_batch(peo_region_id, region_stats, L, device='cpu'):
    """仅模型输入：batch 维 N=T-1，对应 RL step_id = 0..N-1。"""
    peo_id_arr, reg_arr, t_arr, cur_count_arr = _build_pred_windows_numpy(
        peo_region_id, region_stats, L)
    return (
        torch.from_numpy(peo_id_arr).to(device),
        torch.from_numpy(reg_arr).to(device),
        torch.from_numpy(t_arr).to(device),
        torch.from_numpy(cur_count_arr).to(device),
    )


def build_pred_train_batch(peo_region_id, region_stats, L, device='cpu'):
    """由全时隙数据构建用于训练 PredModule 的批次（滑动窗口）。

    Parameters
    ----------
    peo_region_id : (T, P) int64 numpy（建议 T = max_step+1，来自 peo_pos.npy）
    region_stats  : (T, K, 3) float32 numpy
    L             : int，历史时隙数

    Returns
    -------
    peo_id_seq    : (T-1, L+1, P)     LongTensor
    reg_seq       : (T-1, L+1, K, 3)  FloatTensor
    time_ids      : (T-1, L+1)         LongTensor
    cur_count     : (T-1, K)           FloatTensor
    peo_id_next   : (T-1, P)           LongTensor
    reg_stats_next: (T-1, K, 3)        FloatTensor
    """
    T, P = peo_region_id.shape
    peo_id_arr, reg_arr, t_arr, cur_count_arr = _build_pred_windows_numpy(
        peo_region_id, region_stats, L)

    peo_id_next_arr = peo_region_id[1:]
    reg_stats_next_arr = region_stats[1:]

    return (
        torch.from_numpy(peo_id_arr).to(device),
        torch.from_numpy(reg_arr).to(device),
        torch.from_numpy(t_arr).to(device),
        torch.from_numpy(cur_count_arr).to(device),
        torch.from_numpy(peo_id_next_arr).to(device),
        torch.from_numpy(reg_stats_next_arr).to(device),
    )


def _pred_pretrain_lr_for_epoch(ep, epochs, lr_init, lr_late, warmup_epochs, use_cosine):
    """预训练第 ep 轮（0-based）学习率。warmup 内恒定 lr_init；之后余弦降至 lr_late。"""
    if not use_cosine or ep < warmup_epochs:
        return float(lr_init)
    n_cos = epochs - warmup_epochs
    if n_cos <= 1:
        return float(lr_late)
    # t ∈ [0, 1]，从首轮余弦对齐 lr_init，末轮对齐 lr_late
    t = (ep - warmup_epochs) / float(n_cos - 1)
    t = min(1.0, max(0.0, t))
    return float(
        lr_late + (lr_init - lr_late) * 0.5 * (1.0 + math.cos(math.pi * t)))


def pretrain_pred_module(pred_module, peo_region_id, region_stats,
                         epochs=None, device=None, L=None, lr=None, log_every=20):
    """离线预训练：每个 epoch 在全部 N=T-1 个窗口上做一次梯度更新。"""
    epochs = int(epochs if epochs is not None else CONF.get('pred_pretrain_epochs', 200))
    device = device or CONF['device']
    L = L if L is not None else CONF.get('pred_L', 5)
    lr_init = float(
        lr if lr is not None else CONF.get('pred_lr_init', CONF.get('pred_lr', CONF['lr'])))
    lr_late = float(CONF.get('pred_lr_late', 1e-5))
    use_cos = bool(CONF.get('pred_pretrain_cosine_schedule', False))
    warmup = int(CONF.get('pred_pretrain_warmup_epochs', 3000))

    opt = optim.Adam(
        pred_module.parameters(), lr=lr_init, eps=CONF['eps'], weight_decay=1e-6)

    pred_module.train()
    last_dict = {}
    for ep in range(epochs):
        cur_lr = _pred_pretrain_lr_for_epoch(
            ep, epochs, lr_init, lr_late, warmup, use_cos)
        for g in opt.param_groups:
            g['lr'] = cur_lr

        (peo_id_seq, reg_seq, time_ids, cur_count,
         peo_id_next, reg_stats_next) = build_pred_train_batch(
            peo_region_id, region_stats, L, device=device)

        opt.zero_grad()
        _pf, peo_logits, flow_pred, peo_feat, reg_feat = pred_module(
            peo_id_seq, reg_seq, time_ids, cur_count)
        peo_id_cur = peo_id_seq[:, -1, :].long()
        loss, last_dict = pred_module_loss(
            peo_logits, flow_pred, peo_id_next, reg_stats_next, cur_count,
            peo_feat=peo_feat, reg_feat=reg_feat, peo_id_current=peo_id_cur)
        loss.backward()
        nn.utils.clip_grad_norm_(pred_module.parameters(), CONF['max_grad_norm'])
        opt.step()

        if log_every and (ep + 1) % log_every == 0:
            print(f'[pred pretrain] epoch {ep + 1}/{epochs} lr={cur_lr:.2e} loss={loss.item():.4f} '
                  f'ce={last_dict.get("ce", 0):.4f} flow={last_dict.get("flow", 0):.5f} '
                  f'agg_count={last_dict.get("agg_count", 0):.5f} '
                  f'i2c={last_dict.get("i2c", 0):.4f} c2i={last_dict.get("c2i", 0):.4f}')

    pred_module.eval()
    return last_dict


@torch.no_grad()
def compute_pred_feat_timeseries(pred_module, peo_region_id, region_stats,
                                 device=None, L=None):
    """对 step_id=0..T-2 各窗口前向，得到 (T-1, d_pred) numpy float32。"""
    device = device or CONF['device']
    L = L if L is not None else CONF.get('pred_L', 5)
    pred_module.eval()
    peo_id_seq, reg_seq, time_ids, cur_count = build_pred_window_inputs_batch(
        peo_region_id, region_stats, L, device=device)
    pred_forward_t0 = time.perf_counter()
    pred_feat, _, _, _, _ = pred_module(peo_id_seq, reg_seq, time_ids, cur_count)
    pred_forward_elapsed = time.perf_counter() - pred_forward_t0
    pred_steps = int(pred_feat.shape[0])
    pred_avg_s = pred_forward_elapsed / max(pred_steps, 1)
    # print(
    #     f"[pred forward time] steps={pred_steps} "
    #     f"avg_s={pred_avg_s:.6f} avg_ms={pred_avg_s * 1000.0:.3f} "
    #     f"total_s={pred_forward_elapsed:.6f}"
    # )
    return pred_feat.detach().cpu().numpy().astype(np.float32)

