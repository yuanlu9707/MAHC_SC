from util import *

# log_root_path = '../' + __file__.split('/')[-4] + '_log'
# method_name = __file__.split('/')[-2]
log_root_path = '../' + 'log'
# method_name 默认值，与 main.py 中 args.method_name 一致；main 启动时会覆盖为 main 中的 method_name
method_name = 'MAPPO_COMM'  # 'MAPPO', 'MAPPO_CNN', 'MAPPO_CNN_GRU', 'MAPPO_COMM'
CONF = {
    'root_path': log_root_path,
    'lr': 2.5e-4,
    'eps': 1e-5,
    'alpha': 0.99,
    'gamma': 0.99,
    'tau': 0.95,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.1,
    'max_grad_norm': 0.5,
    'env_num': 8,
    'mini_batch_size': 400,
    'buffer_replay_time': 4,
    'clip_param': 0.1,
    'train_iter': 10000,
    'test_num': 50,
    'use_clipped_value_loss': True,
    'decay_rate': 0.9995,
    'decay_start_iter_id': 3000,
    'obs_shape': [4, 20, 20],
    'obs_range': 20,
    'hr_shape': [400, 400],
    'action_space': 2,
    'hidden_size': 256,
    'hidden_size_gru': 256, ###GRU输入也就是CNN输出
    'cat_fe_mlp': False, ##优先级大于cat_fe_cnn256
    'use_gru': False,  ###不用gru了，此时z=x_embed，cat_fe_cnn256和cat_fe_gru哪个为true都一样
    'cat_fe_cnn256': True,
    'cat_fe_gru': False,
    'hidden_size_ac': 128, #输出头128--action
    'device': 'cuda:0',
    'method_name': method_name,
    'dvd_size': 1 / 1,
    'dvd_num': 1,
    'seq_len': 5,
    # 'M_size': [16, 16, 16],  # Z, X, Y
    'M_size': [8, 8, 4],  # Z, X, Y ###GRU输出,通信输入维度
    'mtx_size': 3,  # X' (Y')

    # ---- (预测模块) ----
    'actor_use_pred': True,
    # 仅 method_name 为 MAPPO / MAPPO_CNN 时生效：集中式 Critic 拼接 pred_feat 再估计 V
    'critic_use_pred': True,
    'region_cell_num': 6,          # 每维度格子数；总区域 K = region_cell_num²       ##KAIST6 purdue4
    'pred_d': 64,                  # PredModule 内部特征维度
    'pred_out_d': 64,              # pred_feat 输出维度（拼接到 action_head）
    'pred_L': 5,                   # 历史时隙数（含当前共 L+1 帧）
    'pred_n_heads': 2,             # GCRU / 交叉注意力头数
    'pred_lr_init': 1e-3,           # 预测网学习率上限；RL 每 iter 的 update_pred 也用它（见 ppo）
    'pred_lr_late': 1e-5,           # 预训练余弦阶段的最小学习率
    # True：预训练前 pred_pretrain_warmup_epochs 轮固定 pred_lr_init，之后余弦退火至 pred_lr_late；False：全程 pred_lr_init
    'pred_pretrain_cosine_schedule': False,
    'pred_pretrain_warmup_epochs': 3000,
    'pred_pretrain_epochs': 3000,   # 使用 peo_pos.npy 离线预训练 PredModule 的轮数（每轮 100 个时隙窗口）
    # True：每个 RL iter 后 update_pred 并重写 pred_feat_timeseries.npy；False：仅用预训练后的固定权重与首轮生成的特征表
    'pred_update_each_iter': True,
    # 须 pred_update_each_iter=True：RL iter=0 时 pred LR=预训练末轮 LR；iter 0..pred_rl_cosine_iters-1 余弦降至 pred_rl_lr_floor；iter≥pred_rl_cosine_iters 起不再 update_pred、不再重写 pred_feat_timeseries（与 train_iter 无关，可设为 5000）
    'pred_rl_cosine_lr_then_freeze': True,  #True
    'pred_rl_cosine_iters': 10000,
    'pred_rl_lr_floor': 1e-5,
    # 预测模块总损失 = w_ce*CE + w_flow*MSE_flow + w_count*MSE_count；CE 常为 O(1)~O(log K)，归一化 MSE 常为 O(1e-2) 量级，可增大 w_flow/w_count 平衡梯度
    'pred_loss_w_ce': 1.0,
    'pred_loss_w_flow': 10000.0,
    'pred_loss_w_agg_count': 10000.0,
    # I2C/C2I：True=投影头 + L2 后送入 UniMob；False=直接用 GCRU 输出（与早期实现一致，不建投影层）
    'pred_loss_w_i2c': 1,
    'pred_loss_w_c2i': 1,
    'pred_c2i_temperature': 0.1,
    'i2c_c2i_use_normalize': False,  #False
    # True：行人区域 logits 按当前格到候选格的切比雪夫距离软惩罚（距离 > max_dist 部分每多 1 格减 penalty）
    'use_peo_logits_mask': True,  #True
    'pred_peo_logits_mask_max_dist': 2,       # 切比雪夫距离 ≤ 该整数不罚；超出部分 × penalty
    'pred_peo_logits_mask_penalty': 1.0,
    # True: pred_feat 由 peo_cross/reg_cross 经 MLP 池化；False: 仍用 peo_logits 与 reg_pred_3（损失始终用 logits/flow）
    'pred_feat_use_cross_attn': True,

    # ---- MAPPO_COMM (HGCN 通信模块) ----
    'critic_use_comm': False,  #def=Flase # True: critic 额外输入所有智能体 h_comm，comm 更新含 avg_critic_loss
    'hgcn_hidden_dim': 64,
    'hgcn_out_dim': 64,           ##KAIST64 purdue64or32
    # 'use_gate_out_comm': True,
    'hgcn_num_layers': 1, #def=1 ####最少是2，即便等于1
    # 每层 HGCN 内 MultiheadAttention 的头数；须整除 hgcn_hidden_dim（如 64 可用 1,2,4,8,...）
    'hgcn_MHA_num_heads': 1, #def=2
    'hgcn_node2edge_weight': False, #def=Flase
    # comm_min_clusters / comm_max_clusters 在 train/subp 里按 uav_num 由 comm_spectral_k_bounds 写入
    # 'comm_min_clusters': 2,
    # 'comm_max_clusters': 3,
    'comm_stability_threshold': 0.2, #def=0.3
    'comm_clustering_interval': 10,  #def=10
    # True: 每个 env 回合在时隙 step_id==0 强制谱聚类一次，再按 interval 计数（5→约 0,5,10,...）
    'comm_cluster_at_slot0': True, #def=True
    # 谱聚类后对分组做后处理：每组人数范围；1 / 0 分别表示不限制下限 / 上限
    'comm_min_group_size': 1, #def=1
    'comm_max_group_size': 4, #def=4
    'lambda_consistency': 1,
    'lambda_attention': 0, #0.01
}

def comm_spectral_k_bounds(n_agents):
    """谱聚类搜索簇数 k 的范围：min_k = ceil(n/4)，max_k = floor(n/2)；保证 min_k <= max_k >= 1。"""
    n = int(n_agents)
    min_k = (n + 3) // 4   # ceil(n/4)
    max_k = n // 2         # floor(n/2)
    min_k = max(1, min_k)
    max_k = max(1, max_k)
    if min_k > max_k:
        min_k = max_k
    return min_k, max_k


def get_comm_alpha(iter_i=None, test=False): #KAIST不加门控和系数还可以，pur64可加系数，pur32可加门控和系数
    start_alpha = 0.1
    max_alpha = 0.5
    warmup_iters = 2000
    ramp_iters = 4000

    # start_alpha = 0.1
    # max_alpha = 0.1
    # warmup_iters = 10000
    # ramp_iters = 0

    # start_alpha = 0
    # max_alpha = 0.1
    # warmup_iters = 5000
    # ramp_iters = 0

    if test:
        return max_alpha

    if iter_i is None:
        return max_alpha

    if iter_i < warmup_iters:
        return start_alpha
    elif iter_i < warmup_iters + ramp_iters:
        progress = (iter_i - warmup_iters) / ramp_iters
        return start_alpha + progress * (max_alpha - start_alpha)
    else:
        return max_alpha
