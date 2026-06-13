import torch
import torch.nn as nn
import torch.nn.functional as F

class I2RLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, U, R, loc):
        """
        I2R (Individual-to-Collective) 对齐损失
        目标: 属于同一区域的用户特征聚合后，应与该区域的特征相似。
        """
        B, N, D = U.shape
        K = R.shape[1]

        # 1. 准备聚合容器
        # user_loc_expanded: [B, N, D] -> 用于 scatter 的索引
        loc_expanded = loc.unsqueeze(-1).expand(-1, -1, D)

        # 2. 聚合 (Scatter Add)
        # 将用户特征加到对应的区域上
        # U_sum: [B, K, D]
        U_sum = torch.zeros(B, K, D, device=U.device)
        U_sum.scatter_add_(1, loc_expanded, U)

        # 3. 计数 (统计每个区域有多少人)
        # counts: [B, K, 1]
        counts = torch.zeros(B, K, 1, device=U.device)
        counts.scatter_add_(1, loc.unsqueeze(-1), torch.ones(B, N, 1, device=U.device))

        # 4. 掩码 (Omega): 找出"有人"的区域
        mask = (counts.squeeze(-1) > 0)  # [B, K]

        # [安全检查] 如果整个Batch所有区域都没人(极罕见)，直接返回0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=U.device, requires_grad=True)

        # 5. 求平均聚合特征
        # 加上 1e-8 防止空区域除以 0 (虽然会被 mask 掉，但为了计算图稳定)
        U_agg = U_sum / (counts + 1e-8)

        # 6. 计算余弦相似度损失
        # 只取 mask 为 True 的部分进行计算，节省资源且避免 NaN
        # 只提取有效区域进行计算，效率最高
        U_agg_valid = U_agg[mask]  # [Valid_Count, D]
        R_valid = R[mask]          # [Valid_Count, D]

        # F.cosine_similarity 内部会自动做 Normalize
        cos_sim = F.cosine_similarity(U_agg_valid, R_valid, dim=-1)

        # Loss = 1 - cosine (目标是让 cosine 趋近 1)
        loss = 1.0 - cos_sim.mean()

        return loss


class R2ILoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, u_features, r_features, user_loc_indices):
        """
        计算 R2I (Collective-to-Individual) 对齐损失
        逻辑: 以区域(Region)为锚点(Anchor)，拉近该区域内的用户(Positive Samples)，
             推远所有其他用户(Negative Samples implied in All Samples).

        参数:
            u_features: [B, N, D] 用户特征 (Samples)
            r_features: [B, K, D] 区域特征 (Anchors)
            user_loc_indices: [B, N] 用户所在的区域ID GT (范围 0 ~ K-1)

        返回:
            loss: 标量
        """
        B, N, D = u_features.shape
        K = r_features.shape[1]

        # 1. 特征归一化 (UniMob 要求使用 Cosine Similarity)
        # 对应原代码中的 torch.div(matmul, temp) 前的标准化
        # 归一化使得 dot product 等价于 cosine similarity
        u_norm = F.normalize(u_features, dim=-1)  # [B, N, D]
        r_norm = F.normalize(r_features, dim=-1)  # [B, K, D]

        # 2. 计算相似度矩阵 [B, K, N]
        # Anchor: Region (K), Sample: User (N)
        # 对应原代码中的 anchor_dot_contrast
        # 目标: [B, K, N] -> 每个区域 k 与每个用户 n 的相似度
        # 使用 bmm: (B, K, D) @ (B, D, N) -> (B, K, N)
        logits = torch.bmm(r_norm, u_norm.transpose(1, 2)) / self.temperature

        # 3. 构建正样本掩码 Mask [B, K, N]
        # 如果 loc[b, n] == k，则 mask[b, k, n] = 1
        # 对应原代码中的 mask = torch.eq(...)
        # loc[b, n] == k 表示用户 n 属于区域 k (正样本)

        # [B, 1, N]
        loc_expanded = user_loc_indices.unsqueeze(1)
        # [1, K, 1]
        region_ids = torch.arange(K, device=u_features.device).view(1, K, 1)

        # pos_mask: [B, K, N] (True 表示是正样本)
        pos_mask = (loc_expanded == region_ids)

        # 4. 确定有效区域 (Valid Regions / Omega)
        # 只有包含至少一个用户的区域才计算 Loss，否则分母为0或无意义
        # 只有确实包含至少一个用户的区域，才能作为 Anchor 计算 Loss
        # valid_region_mask: [B, K]
        valid_region_mask = pos_mask.any(dim=-1)

        # [安全检查] 如果整个Batch没有任何有效区域(极罕见)，直接返回0
        if valid_region_mask.sum() == 0:
            return torch.tensor(0.0, device=u_features.device, requires_grad=True)

        # 5. 计算 Loss (LogSumExp Trick): LogSumExp(All) - LogSumExp(Pos)
        # 这比原代码的 exp_logits.sum 更加数值稳定，防止溢出
        # 公式: - log( sum(exp(pos)) / sum(exp(all)) )
        #     = log(sum(exp(all))) - log(sum(exp(pos)))

        # (A) LogSumExp(All Samples) -> 分母
        # 对用户维度 N 求 LSE -> [B, K]
        lse_all = torch.logsumexp(logits, dim=-1)

        # (B) LogSumExp(Positive Samples) -> 分子
        # 技巧: 把非正样本的位置设为 -inf，这样 exp(-inf)=0，不会影响求和
        neg_inf = torch.zeros_like(logits).fill_(float('-inf'))
        logits_pos = torch.where(pos_mask, logits, neg_inf)

        # 对用户维度 N 求 LSE -> [B, K]
        lse_pos = torch.logsumexp(logits_pos, dim=-1)

        # (C) 计算每个区域的 Loss
        loss_per_region = lse_all - lse_pos

        # 6. 平均 Loss
        # 对应原代码中的 mean_log_prob_pos.mean()
        # 只对有效区域(有人区域)求平均
        final_loss = loss_per_region[valid_region_mask].mean()

        return final_loss


# ================= 验证代码 =================
if __name__ == "__main__":
    B, N, K, D = 2, 5, 3, 4
    model = R2ILoss(temperature=0.1)
    model1 = I2RLoss()

    # 模拟数据
    U = torch.randn(B, N, D, requires_grad=True)
    R = torch.randn(B, K, D, requires_grad=True)
    # 模拟 Loc:
    # Batch 0: 区域0有两个用户(0,1)，区域1有一个(2)，区域2有两个(3,4) -> 所有区域都有效
    # Batch 1: 用户全部在区域0 -> 区域1,2无效(空)
    loc = torch.tensor([[0, 0, 1, 2, 2],
                        [0, 0, 0, 0, 0]]).long()

    loss = model(U, R, loc)
    loss1 = model1(U, R, loc)

    print("Final R2I Loss:", loss.item())
    print("Final I2R Loss:", loss1.item())

    loss.backward()
    print("Backward pass successful.")

    
