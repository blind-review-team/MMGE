import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def infonce_align(embed1, embed2, temperature=0.2):
    """
    InfoNCE对比学习对齐方法，当前GUME中使用的方法

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征
        temperature: 温度参数，控制相似度分布的平滑程度

    Returns:
        对比学习损失
    """
    embed1 = F.normalize(embed1, dim=1, p=2)
    embed2 = F.normalize(embed2, dim=1, p=2)

    # 计算正样本对的得分
    pos_score = (embed1 * embed2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    # 计算所有样本对的得分
    ttl_score = torch.matmul(embed1, embed2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

    # 计算对比损失
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def mse_align(embed1, embed2):
    """
    MSE/L2损失对齐，直接最小化两个模态特征之间的欧氏距离

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征

    Returns:
        MSE损失
    """
    return torch.mean((embed1 - embed2) ** 2)


def cosine_align(embed1, embed2):
    """
    余弦相似度对齐，最大化特征间的余弦相似度

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征

    Returns:
        余弦损失（负余弦相似度的均值）
    """
    embed1 = F.normalize(embed1, dim=-1)
    embed2 = F.normalize(embed2, dim=-1)
    return -torch.mean(torch.sum(embed1 * embed2, dim=1))


def kl_align(embed1, embed2):
    """
    KL散度对齐，让两个特征分布之间的KL散度最小化

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征

    Returns:
        KL散度损失
    """
    log_softmax1 = F.log_softmax(embed1, dim=1)
    softmax2 = F.softmax(embed2, dim=1)
    return F.kl_div(log_softmax1, softmax2, reduction='batchmean')


def js_align(embed1, embed2):
    """
    JS散度对齐，计算两个分布之间的JS散度

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征

    Returns:
        JS散度损失
    """
    log_softmax1 = F.log_softmax(embed1, dim=1)
    log_softmax2 = F.log_softmax(embed2, dim=1)
    softmax1 = F.softmax(embed1, dim=1)
    softmax2 = F.softmax(embed2, dim=1)

    # 计算平均分布
    m = 0.5 * (softmax1 + softmax2)
    m_log = torch.log(m + 1e-8)

    # JS散度 = 0.5 * (KL(P||M) + KL(Q||M))
    kl_p_m = F.kl_div(m_log, softmax1, reduction='batchmean')
    kl_q_m = F.kl_div(m_log, softmax2, reduction='batchmean')
    return 0.5 * (kl_p_m + kl_q_m)


def mmd_align(embed1, embed2, kernel='rbf', sigma=1.0):
    """
    MMD (最大平均差异) 对齐，最小化两个分布在再生核希尔伯特空间中的距离

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征
        kernel: 核函数类型，'rbf'或'linear'
        sigma: 高斯核的参数

    Returns:
        MMD损失
    """
    if kernel == 'linear':
        x1x1 = torch.mm(embed1, embed1.t())
        x2x2 = torch.mm(embed2, embed2.t())
        x1x2 = torch.mm(embed1, embed2.t())
        return torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)

    elif kernel == 'rbf':
        n = embed1.size(0)
        m = embed2.size(0)

        x1x1 = torch.mm(embed1, embed1.t())
        x1 = torch.diag(x1x1).unsqueeze(1).expand(n, n)
        x2 = x1.t()

        x1x1 = x1 + x2 - 2 * x1x1
        x1x1 = torch.exp(-x1x1 / (2 * sigma ** 2))

        x2x2 = torch.mm(embed2, embed2.t())
        x1 = torch.diag(x2x2).unsqueeze(1).expand(m, m)
        x2 = x1.t()

        x2x2 = x1 + x2 - 2 * x2x2
        x2x2 = torch.exp(-x2x2 / (2 * sigma ** 2))

        x1x2 = torch.mm(embed1, embed2.t())
        x1 = torch.sum(embed1 ** 2, dim=1).unsqueeze(1).expand(n, m)
        x2 = torch.sum(embed2 ** 2, dim=1).unsqueeze(0).expand(n, m)

        x1x2 = x1 + x2 - 2 * x1x2
        x1x2 = torch.exp(-x1x2 / (2 * sigma ** 2))

        return torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)


def clip_align(embed1, embed2, temperature=0.2):
    """
    CLIP式对齐，使用文本-图像对的配对信息，通过交叉熵损失使匹配对的相似度高于不匹配对

    Args:
        embed1: 第一个模态的特征 (如图像特征)
        embed2: 第二个模态的特征 (如文本特征)
        temperature: 温度参数

    Returns:
        CLIP对比损失
    """
    embed1 = F.normalize(embed1, dim=1)
    embed2 = F.normalize(embed2, dim=1)

    # 计算相似度矩阵
    logits = torch.matmul(embed1, embed2.t()) / temperature

    # 假设相同索引的特征是匹配的对
    labels = torch.arange(logits.size(0), device=logits.device)

    # 计算双向交叉熵损失
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    return (loss_i + loss_t) / 2.0


def reg_align(embed1, embed2, reg_weight=0.1):
    """
    自正则化对齐，通过添加正则化项鼓励两个模态特征满足某些结构约束

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征
        reg_weight: 正则化权重

    Returns:
        带正则项的对齐损失
    """
    # 基础相似度损失
    sim_loss = cosine_align(embed1, embed2)

    # 鼓励特征有更大的方差（更具有辨别性）
    var_loss = -(torch.var(embed1, dim=0).mean() + torch.var(embed2, dim=0).mean())

    # 整合基础损失和正则化损失
    return sim_loss + reg_weight * var_loss


class AdversarialAligner(nn.Module):
    """
    对抗性对齐模型，使用判别器区分不同模态，并训练特征提取器欺骗判别器
    """

    def __init__(self, feature_dim, hidden_dim=64):
        super(AdversarialAligner, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, embed1, embed2):
        """
        前向传播，预测特征来自哪个模态

        Args:
            embed1: 第一个模态的特征
            embed2: 第二个模态的特征

        Returns:
            判别器损失和生成器损失
        """
        # 真实标签：embed1为1，embed2为0
        batch_size = embed1.size(0)
        real_labels = torch.ones(batch_size, 1, device=embed1.device)
        fake_labels = torch.zeros(batch_size, 1, device=embed1.device)

        # 判别器预测
        pred_real = self.discriminator(embed1)
        pred_fake = self.discriminator(embed2)

        # 判别器损失
        d_loss_real = F.binary_cross_entropy(pred_real, real_labels)
        d_loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # 生成器损失：尝试欺骗判别器
        g_loss = F.binary_cross_entropy(pred_fake, real_labels)

        return d_loss, g_loss


def attention_align(query, key, value):
    """
    注意力机制对齐，使用注意力机制让一个模态特征关注另一个模态的相关部分

    Args:
        query: 查询特征 (如文本特征)
        key: 键特征 (如图像特征)
        value: 值特征 (通常与key相同)

    Returns:
        加权后的特征表示和注意力权重
    """
    # 计算注意力分数
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # 缩放注意力分数
    attention_scores = attention_scores / np.sqrt(key.size(-1))

    # 应用softmax获得注意力权重
    attention_weights = F.softmax(attention_scores, dim=-1)

    # 计算加权特征
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


def cca_align(embed1, embed2, reg=1e-4):
    """
    CCA (典型相关分析) 对齐，最大化两个视图之间的互相关

    Args:
        embed1: 第一个模态的特征 [batch_size, dim1]
        embed2: 第二个模态的特征 [batch_size, dim2]
        reg: 正则化参数

    Returns:
        负相关损失（越小表示相关性越高）
    """
    # 中心化
    embed1 = embed1 - embed1.mean(dim=0, keepdim=True)
    embed2 = embed2 - embed2.mean(dim=0, keepdim=True)

    # 计算协方差矩阵
    c1 = torch.matmul(embed1.t(), embed1) / (embed1.size(0) - 1) + reg * torch.eye(embed1.size(1), device=embed1.device)
    c2 = torch.matmul(embed2.t(), embed2) / (embed2.size(0) - 1) + reg * torch.eye(embed2.size(1), device=embed2.device)
    c12 = torch.matmul(embed1.t(), embed2) / (embed1.size(0) - 1)

    # 计算协方差矩阵的平方根的逆
    c1_inv_sqrt = torch.inverse(torch.matrix_power(c1, 1 / 2))
    c2_inv_sqrt = torch.inverse(torch.matrix_power(c2, 1 / 2))

    # 计算CCA矩阵
    cca_matrix = torch.matmul(torch.matmul(c1_inv_sqrt, c12), c2_inv_sqrt)

    # 使用奇异值分解计算典型相关系数
    _, s, _ = torch.svd(cca_matrix)

    # 负相关损失（最大化相关性等价于最小化负相关）
    return -torch.mean(s)


# 组合多种对齐方法
def combined_align(embed1, embed2, weights=None):
    """
    组合多种对齐方法

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征
        weights: 各种对齐方法的权重字典

    Returns:
        组合损失
    """
    if weights is None:
        weights = {
            'infonce': 1.0,
            'cosine': 0.5,
            'mse': 0.3,
            'kl': 0.2,
            'reg': 0.1
        }

    total_loss = 0

    if 'infonce' in weights:
        total_loss += weights['infonce'] * infonce_align(embed1, embed2)

    if 'cosine' in weights:
        total_loss += weights['cosine'] * cosine_align(embed1, embed2)

    if 'mse' in weights:
        total_loss += weights['mse'] * mse_align(embed1, embed2)

    if 'kl' in weights:
        total_loss += weights['kl'] * kl_align(embed1, embed2)

    if 'reg' in weights:
        total_loss += weights['reg'] * reg_align(embed1, embed2)

    return total_loss


# 组合InfoNCE和注意力机制的对齐方法
def infonce_attention_align(embed1, embed2, temperature=0.2, infonce_weight=0.7, attention_weight=0.3):
    """
    组合InfoNCE对比学习和注意力机制进行特征对齐

    Args:
        embed1: 第一个模态的特征
        embed2: 第二个模态的特征
        temperature: 温度参数
        infonce_weight: InfoNCE损失的权重
        attention_weight: 注意力机制损失的权重

    Returns:
        组合对齐损失
    """
    # 1. InfoNCE对比学习损失
    infonce_loss = infonce_align(embed1, embed2, temperature)

    # 2. 注意力机制对齐
    # 让第一个模态关注第二个模态
    output1, weights1 = attention_align(embed1, embed2, embed2)
    # 让第二个模态关注第一个模态
    output2, weights2 = attention_align(embed2, embed1, embed1)

    # 计算注意力对齐后特征与原始特征的一致性损失
    consistency_loss1 = F.mse_loss(output1, embed1)
    consistency_loss2 = F.mse_loss(output2, embed2)
    attn_loss = (consistency_loss1 + consistency_loss2) / 2.0

    # 组合损失
    combined_loss = infonce_weight * infonce_loss + attention_weight * attn_loss

    return combined_loss
