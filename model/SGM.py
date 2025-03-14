import torch.nn as nn
import torch.utils.data
import warnings
import torch

warnings.filterwarnings("ignore", message="Numerical issues were encountered ")


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


eps = 1e-8


def sinkhorn(M, r, c, iteration):
    p = torch.softmax(M, dim=-1)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iteration):
        u = r / ((p * v.unsqueeze(-2)).sum(-1) + eps)
        v = c / ((p * u.unsqueeze(-1)).sum(-2) + eps)
    p = p * u.unsqueeze(-1) * v.unsqueeze(-2)
    return p


def sink_algorithm(M, dustbin, iteration):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    r = torch.ones([M.shape[0], M.shape[1] - 1], device='cuda')
    r = torch.cat([r, torch.ones([M.shape[0], 1], device='cuda') * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1], device='cuda')
    c = torch.cat([c, torch.ones([M.shape[0], 1], device='cuda') * M.shape[2]], dim=-1)
    p = sinkhorn(M, r, c, iteration)
    return p


def seeding(nn_index1, nn_index2, x1, x2, topk, match_score, confbar, nms_radius, use_mc=True, test=False):
    # apply mutual check before nms
    if use_mc:
        mask_not_mutual = nn_index2.gather(dim=-1, index=nn_index1) != torch.arange(nn_index1.shape[1], device='cuda')
        match_score[mask_not_mutual] = -1
    # NMS
    pos_dismat1 = ((x1.norm(p=2, dim=-1) ** 2).unsqueeze_(-1) + (x1.norm(p=2, dim=-1) ** 2).unsqueeze_(-2) - 2 * (
            x1 @ x1.transpose(1, 2))).abs_().sqrt_()
    x2 = x2.gather(index=nn_index1.unsqueeze(-1).expand(-1, -1, 2), dim=1)
    pos_dismat2 = ((x2.norm(p=2, dim=-1) ** 2).unsqueeze_(-1) + (x2.norm(p=2, dim=-1) ** 2).unsqueeze_(-2) - 2 * (
            x2 @ x2.transpose(1, 2))).abs_().sqrt_()
    radius1, radius2 = nms_radius * pos_dismat1.mean(dim=(1, 2), keepdim=True), nms_radius * pos_dismat2.mean(
        dim=(1, 2), keepdim=True)
    nms_mask = (pos_dismat1 >= radius1) & (pos_dismat2 >= radius2)
    mask_not_local_max = (match_score.unsqueeze(-1) >= match_score.unsqueeze(-2)) | nms_mask
    mask_not_local_max = ~(mask_not_local_max.min(dim=-1).values)
    match_score[mask_not_local_max] = -1

    # confidence bar
    match_score[match_score < confbar] = -1
    mask_survive = match_score > 0
    if test:
        topk = min(mask_survive.sum(dim=1)[0] + 2, topk)
    _, topindex = torch.topk(match_score, topk, dim=-1)  # b*k
    seed_index1, seed_index2 = topindex, nn_index1.gather(index=topindex, dim=-1)
    return seed_index1, seed_index2


class PointCN(nn.Module):
    def __init__(self, channels, out_channels):
        nn.Module.__init__(self)
        self.shot_cut = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x) + self.shot_cut(x)


class attention_propagantion(nn.Module):

    def __init__(self, channel, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channel // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channel, channel, kernel_size=1), nn.Conv1d(
            channel, channel, kernel_size=1), \
            nn.Conv1d(channel, channel, kernel_size=1)
        self.mh_filter = nn.Conv1d(channel, channel, kernel_size=1)
        self.cat_filter = nn.Sequential(nn.Conv1d(2 * channel, 2 * channel, kernel_size=1),
                                        nn.SyncBatchNorm(2 * channel), nn.ReLU(),
                                        nn.Conv1d(2 * channel, channel, kernel_size=1))

    def forward(self, desc1, desc2, weight_v=None):
        # desc1(q) attend to desc2(k,v)
        batch_size = desc1.shape[0]
        query, key, value = self.query_filter(desc1).view(batch_size, self.head, self.head_dim, -1), self.key_filter(
            desc2).view(batch_size, self.head, self.head_dim, -1), \
            self.value_filter(desc2).view(batch_size, self.head, self.head_dim, -1)
        if weight_v is not None:
            value = value * weight_v.view(batch_size, 1, 1, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim=-1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        desc1_new = desc1 + self.cat_filter(torch.cat([desc1, add_value], dim=1))
        return desc1_new


class hybrid_block(nn.Module):
    # def __init__(self, channel, head):
    def __init__(self, num_support, feature_dim: int, layer_names: list):
        nn.Module.__init__(self)
        self.head = head
        self.channel = channel
        self.attention_block_down = attention_propagantion(channel, head)
        self.cluster_filter = nn.Sequential(nn.Conv1d(2 * channel, 2 * channel, kernel_size=1),
                                            nn.SyncBatchNorm(2 * channel), nn.ReLU(),
                                            nn.Conv1d(2 * channel, 2 * channel, kernel_size=1))
        self.cross_filter = attention_propagantion(channel, head)
        self.confidence_filter = PointCN(2 * channel, 1)
        self.attention_block_self = attention_propagantion(channel, head)
        self.attention_block_up = attention_propagantion(channel, head)

    def forward(self, desc1, desc2, seed_index1, seed_index2):
        cluster1, cluster2 = desc1.gather(dim=-1, index=seed_index1.unsqueeze(1).expand(-1, self.channel, -1)), \
            desc2.gather(dim=-1, index=seed_index2.unsqueeze(1).expand(-1, self.channel, -1))

        # pooling
        cluster1, cluster2 = self.attention_block_down(cluster1, desc1), self.attention_block_down(cluster2, desc2)
        concate_cluster = self.cluster_filter(torch.cat([cluster1, cluster2], dim=1))
        # filtering
        cluster1, cluster2 = self.cross_filter(concate_cluster[:, :self.channel], concate_cluster[:, self.channel:]), \
            self.cross_filter(concate_cluster[:, self.channel:], concate_cluster[:, :self.channel])
        cluster1, cluster2 = self.attention_block_self(cluster1, cluster1), self.attention_block_self(cluster2,
                                                                                                      cluster2)
        # unpooling
        seed_weight = self.confidence_filter(torch.cat([cluster1, cluster2], dim=1))
        seed_weight = torch.sigmoid(seed_weight).squeeze(1)
        desc1_new, desc2_new = self.attention_block_up(desc1, cluster1, seed_weight), self.attention_block_up(desc2,
                                                                                                              cluster2,
                                                                                                              seed_weight)
        return desc1_new, desc2_new, seed_weight


if __name__ == '__main__':
    model = hybrid_block(channel=3, head=8)
    model.eval()
    print(model)
    # input = torch.randn(40, 3, 3, 3)
    # y = model(input)
    # print(y.size())

"""
在原始superglue代码中AttentionalGNN是这样进行模型的调用的
def __init__(self, feature_dim: int, num_heads: int):
因此
self.gnn = AttentionalGNN(feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

在gia中进行修改
参照CSA_block.py中对AttentionalGNN中的形参进行修改
def __init__(self, num_support, feature_dim: int, layer_names: list):
因此
CSA_block_att = CSA_block.AttentionalGNN(num_supports, emb_size, ['cross'] * config['num_generation']).to(GPU)

在sgmnet网络中若要进行修改
def __init__(self, channel, head):
肯定是要修改掉的
在保证原数据类型统一的前提下
因此
def __init__(self, num_support, feature_dim: int, layer_names: list):
SGM_block_att = SGM.hybrid_block(num_supports, emb_size, ['cross'] * config['num_generation']).to(GPU)

"""

# class hybrid_block(nn.Module):
#     def __init__(self, channel, head):
#     # def __init__(self, num_support, feature_dim: int, layer_names: list):
#         nn.Module.__init__(self)
#         self.head = head
#         self.channel = channel
#         self.attention_block_down = attention_propagantion(channel, head)
#         self.cluster_filter = nn.Sequential(nn.Conv1d(2 * channel, 2 * channel, kernel_size=1),
#                                             nn.SyncBatchNorm(2 * channel), nn.ReLU(),
#                                             nn.Conv1d(2 * channel, 2 * channel, kernel_size=1))
#         self.cross_filter = attention_propagantion(channel, head)
#         self.confidence_filter = PointCN(2 * channel, 1)
#         self.attention_block_self = attention_propagantion(channel, head)
#         self.attention_block_up = attention_propagantion(channel, head)
#
#     def forward(self, desc1, desc2, seed_index1, seed_index2):
#         cluster1, cluster2 = desc1.gather(dim=-1, index=seed_index1.unsqueeze(1).expand(-1, self.channel, -1)), \
#             desc2.gather(dim=-1, index=seed_index2.unsqueeze(1).expand(-1, self.channel, -1))
#
#         # pooling
#         cluster1, cluster2 = self.attention_block_down(cluster1, desc1), self.attention_block_down(cluster2, desc2)
#         concate_cluster = self.cluster_filter(torch.cat([cluster1, cluster2], dim=1))
#         # filtering
#         cluster1, cluster2 = self.cross_filter(concate_cluster[:, :self.channel], concate_cluster[:, self.channel:]), \
#             self.cross_filter(concate_cluster[:, self.channel:], concate_cluster[:, :self.channel])
#         cluster1, cluster2 = self.attention_block_self(cluster1, cluster1), self.attention_block_self(cluster2,
#                                                                                                       cluster2)
#         # unpooling
#         seed_weight = self.confidence_filter(torch.cat([cluster1, cluster2], dim=1))
#         seed_weight = torch.sigmoid(seed_weight).squeeze(1)
#         desc1_new, desc2_new = self.attention_block_up(desc1, cluster1, seed_weight), self.attention_block_up(desc2,
#                                                                                                               cluster2,
#                                                                                                               seed_weight)
#         return desc1_new, desc2_new, seed_weight
