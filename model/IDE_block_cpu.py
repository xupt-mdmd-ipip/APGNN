'''Yang, Ling, et al. 
"Dpgn: Distribution propagation graph network for few-shot learning." 
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.utils.checkpoint as cp

class PointSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Point Similarity (see paper 3.2.1) Vp_(l-1) -> Ep_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(PointSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vp_last_gen, ep_last_gen, distance_metric):
        """
        Forward method of Point Similarity
        :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
        :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
                 l2 version of node similarities
        """

        vp_i = vp_last_gen.unsqueeze(2)
        vp_j = torch.transpose(vp_i, 1, 2)
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j) ** 2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(
            ep_last_gen.get_device())
        # ep_last_gen *= diagonal_mask
        ep_last_gen = torch.mul(ep_last_gen, diagonal_mask)  # 修改
        ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)
        ep_ij = F.normalize(ep_ij.squeeze(1) * ep_last_gen, p=1, dim=-1) * ep_last_gen_sum
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(ep_last_gen.get_device())
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        node_similarity_l2 = -torch.sum(vp_similarity, 3)
        return ep_ij, node_similarity_l2


class P2DAgg(nn.Module):
    def __init__(self, in_c, out_c):
        """
        P2D Aggregation (see paper 3.2.1) Ep_(l) -> Vd_(l)
        :param in_c: number of input channel for the fc layer
        :param out_c:number of output channel for the fc layer
        """
        super(P2DAgg, self).__init__()
        # add the fc layer
        self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True),
                                             nn.LeakyReLU()])
        self.out_c = out_c

    def forward(self, point_edge, distribution_node):
        """
        Forward method of P2D Aggregation
        :param point_edge: current generation's edge feature of point graph, Ep_(l)
        :param distribution_node: last generation's node feature of distribution graph, Ed_(l-1)
        :return: current generation's node feature of distribution graph, Vd_(l)
        """
        meta_batch = point_edge.size(0)
        num_sample = point_edge.size(1)
        distribution_node = torch.cat([point_edge[:, :, :self.out_c], distribution_node], dim=2)
        distribution_node = distribution_node.view(meta_batch * num_sample, -1)
        self.p2d_transform.cuda()
        distribution_node = self.p2d_transform(distribution_node)
        distribution_node = distribution_node.view(meta_batch, num_sample, -1)
        return distribution_node


class DistributionSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Distribution Similarity (see paper 3.2.2) Vd_(l) -> Ed_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(DistributionSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vd_curr_gen, ed_last_gen, distance_metric):
        """
        Forward method of Distribution Similarity
        :param vd_curr_gen: current generation's node feature of distribution graph, Vd_(l)
        :param ed_last_gen: last generation's edge feature of distribution graph, Ed_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l)
        """
        vd_i = vd_curr_gen.unsqueeze(2)
        vd_j = torch.transpose(vd_i, 1, 2)
        if distance_metric == 'l2':
            vd_similarity = (vd_i - vd_j) ** 2
        elif distance_metric == 'l1':
            vd_similarity = torch.abs(vd_i - vd_j)
        trans_similarity = torch.transpose(vd_similarity, 1, 3)
        ed_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(
            ed_last_gen.get_device())
        # ed_last_gen *= diagonal_mask
        ed_last_gen = torch.mul(ed_last_gen, diagonal_mask)  # 修改
        ed_last_gen_sum = torch.sum(ed_last_gen, -1, True)
        ed_ij = F.normalize(ed_ij.squeeze(1) * ed_last_gen, p=1, dim=-1) * ed_last_gen_sum
        diagonal_reverse_mask = torch.eye(vd_curr_gen.size(1)).unsqueeze(0).to(ed_last_gen.get_device())
        ed_ij += (diagonal_reverse_mask + 1e-6)
        ed_ij /= torch.sum(ed_ij, dim=2).unsqueeze(-1)

        return ed_ij


class D2PAgg(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        D2P Aggregation (see paper 3.2.2) Ed_(l) -> Vp_(l+1)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(D2PAgg, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.point_node_transform = nn.Sequential(*layer_list)

    def forward(self, distribution_edge, point_node):
        """
        Forward method of D2P Aggregation
        :param distribution_edge: current generation's edge feature of distribution graph, Ed_(l)
        :param point_node: last generation's node feature of point graph, Vp_(l-1)
        :return: current generation's node feature of point graph, Vp_(l)
        """
        # get size
        meta_batch = point_node.size(0)
        num_sample = point_node.size(1)

        # get eye matrix (batch_size x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(distribution_edge.get_device())

        # set diagonal as zero and normalize
        edge_feat = F.normalize(distribution_edge * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat, point_node)

        node_feat = torch.cat([point_node, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))
        node_feat = node_feat.transpose(1, 2).squeeze(-1)

        return node_feat


class DPGN(nn.Module):
    def __init__(self, num_generations, dropout, num_support_sample, num_sample, point_metric, distribution_metric,
                 emb_size):
        """
        DPGN model
        :param num_generations: number of total generations
        :param dropout: dropout rate
        :param num_support_sample: number of support sample
        :param num_sample: number of sample
        :param loss_indicator: indicator of what losses are using
        :param point_metric: metric for distance in point graph
        :param distribution_metric: metric for distance in distribution graph
        """
        super(DPGN, self).__init__()
        self.generation = num_generations
        self.dropout = dropout
        self.num_support_sample = num_support_sample
        self.num_sample = num_sample
        self.point_metric = point_metric
        self.distribution_metric = distribution_metric
        P_Sim = PointSimilarity(emb_size, emb_size, dropout=self.dropout)
        self.add_module('initial_edge', P_Sim)
        for l in range(self.generation):
            P2D = P2DAgg(2 * num_support_sample, num_support_sample)
            self.add_module('point2distribution_generation_{}'.format(l), P2D)
            self.add_module('point_sim_generation_{}'.format(l), P_Sim)

    def forward(self, middle_node, point_node, distribution_node, distribution_edge, point_edge):
        """
        Forward method of DPGN
        :param middle_node: feature extracted from second last layer of Embedding Network
        :param point_node: feature extracted from last layer of Embedding Network
        :param distribution_node: initialized nodes of distribution graph
        :param distribution_edge: initialized edges of distribution graph
        :param point_edge: initialized edge of point graph
        :return: classification result
                 instance_similarity
                 distribution_similarity
        """
        distribution_nodes= []
        point_edge, _ = self._modules['initial_edge'](middle_node, point_edge, self.point_metric)
        for l in range(self.generation):
            ## print('这是point_node：', point_node.size())
            point_edge, node_similarity_l2 = self._modules['point_sim_generation_{}'.format(l)](point_node, point_edge,
                                                                                                self.point_metric)
            distribution_node = self._modules['point2distribution_generation_{}'.format(l)](point_edge,
                                                                                            distribution_node)
            distribution_nodes.append(distribution_node)
        return distribution_nodes











'''AGNN'''
class PointSimilarity_Pre(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Point Similarity (see paper 3.2.1) Vp_(l-1) -> Ep_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(PointSimilarity_Pre, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vp_last_gen, ep_last_gen, distance_metric):
        """
        Forward method of Point Similarity
        :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
        :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
                 l2 version of node similarities
        """

        vp_i = vp_last_gen.unsqueeze(2)
        vp_j = torch.transpose(vp_i, 1, 2)
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j) ** 2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(
            ep_last_gen.device)
        # ep_last_gen *= diagonal_mask
        ep_last_gen = torch.mul(ep_last_gen, diagonal_mask)  # 修改
        ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)
        ep_ij = F.normalize(ep_ij.squeeze(1) * ep_last_gen, p=1, dim=-1) * ep_last_gen_sum

        # ep_ij = F.normalize(ep_ij.squeeze(1), p=1, dim=-1)
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(ep_last_gen.device)
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        node_similarity_l2 = -torch.sum(vp_similarity, 3)
        return ep_ij, node_similarity_l2

class PointSimilarity2(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0, layer=1):
        """
        Point Similarity (see paper 3.2.1) Vp_(l-1) -> Ep_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(PointSimilarity2, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

        # flexible sparsity ratio
        self.ratio = 0.1 * layer

    def forward(self, vp_last_gen, ep_last_gen, distance_metric):
        """
        Forward method of Point Similarity
        :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
        :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
                 l2 version of node similarities
        """

        vp_i = vp_last_gen.unsqueeze(2)
        vp_j = torch.transpose(vp_i, 1, 2)
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j) ** 2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))


        # normalization
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(
            ep_last_gen.device)
        # ep_last_gen *= diagonal_mask
        ep_last_gen = torch.mul(ep_last_gen, diagonal_mask)#修改
        ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)

        # layer memory attention (edge memory)
        ep_ij = ep_ij.squeeze(1) * ep_last_gen

        # neighbor attention via sparsity
        kval = int(vp_last_gen.shape[1] * (1.0 - self.ratio))
        topk, indices = torch.topk(ep_ij, kval, dim=2, largest=True)
        mask = torch.zeros(*ep_ij.shape, device=ep_last_gen.device)
        mask = mask.scatter(1, indices, 1)  # mark valueable nodes
        ep_ij = ep_ij * mask  # only keep the weights of these nodes

        ep_ij = F.normalize(ep_ij, p=1, dim=-1) * ep_last_gen_sum
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).cuda().to(ep_last_gen.device)
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        node_similarity_l2 = -torch.sum(vp_similarity, 3)
        return ep_ij, node_similarity_l2

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        return attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, dropout=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k):
        # # print(q.shape)
        # # q = q.reshape(-1)
        # # q = q.reshape(160,128)
        # d_k, n_head = self.d_k, self.n_head
        # sz_b, len_q, _ = q.size()
        # sz_b, len_k, _ = k.size()
        #
        # sz_b = torch.Tensor(sz_b)
        # len_q = torch.Tensor(len_q)
        # n_head = torch.Tensor(n_head)
        # d_k = torch.Tensor(d_k)
        # print(sz_b.shape, len_q.shape, n_head.shape, d_k.shape)
        # residual = q
        #
        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # '''修改'''
        # # q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        # # k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)
        #
        # q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        #
        # attn = self.attention(q, k)
        #
        # return attn
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk

        attn = self.attention(q, k)

        return attn


class ScaledDotProductAttention2(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention2(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention2(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output




class AGNN(nn.Module):
    def __init__(self, num_generations, dropout, num_support_sample, num_sample, loss_indicator, point_metric,emb_size):
        """
        AGNN model
        :param num_generations: number of total generations
        :param dropout: dropout rate
        :param num_support_sample: number of support sample
        :param num_sample: number of sample
        :param loss_indicator: indicator of what losses are using
        :param point_metric: metric for distance in the graph

        """
        super(AGNN, self).__init__()
        self.generation = num_generations
        self.dropout = dropout
        self.num_support_sample = num_support_sample
        self.num_sample = num_sample
        self.loss_indicator = loss_indicator
        self.point_metric = point_metric

        self.fusion = nn.Conv2d(2,1, kernel_size=(1,1), stride=(1,1))
        # self.slf_attn = MultiHeadAttention(1, 64, 64, dropout=0.5)
        self.slf_attn = MultiHeadAttention(1, emb_size, emb_size, dropout=0.5)

        P_Sim = PointSimilarity_Pre(emb_size, emb_size, dropout=self.dropout)
        #dim和add_两个参数得研究一下
        self.dim = emb_size+640    #emb_size=64
        add_dim = 16
        self.add_module('initial_edge', P_Sim)
        for l in range(self.generation):
            D2P = D2PAgg(self.dim*2, add_dim, dropout=self.dropout if l < self.generation-1 else 0.0)
            P_Sim = PointSimilarity2(self.dim, self.dim, dropout=self.dropout if l < self.generation-1 else 0.0, layer=l)
            self.add_module('distribution2point_generation_{}'.format(l), D2P)
            self.add_module('point_sim_generation_{}'.format(l), P_Sim)
            self.dim = self.dim + add_dim


    def forward(self, middle_node, point_node, lab_vec, point_edge, tr_label):
        #print('这是tr_label',tr_label)
        """
        Forward method of AGNN
        :param middle_node: feature extracted from second last layer of Embedding Network
        :param point_node: feature extracted from last layer of Embedding Network
        :param point_edge: initialized edge of point graph
        :return: classification result
                 instance_similarity
        """

        point_similarities = []
        node_similarities_l2 = []
        point_nodes = []
        point_edge, _ = self._modules['initial_edge'](middle_node, point_edge, self.point_metric)

        # Node self-attention via a transformer block
        # print('这是point_node.size：',point_node.size())
        att = self.slf_attn(point_node,point_node)

        # Label initialization
        # print('这是tr_label.size1：', tr_label.size())
        # tr_label=np.resize(tr_label)
        # print('这是tr_label.size2：', tr_label.size())
        [b, nk, c] = tr_label.size()
        tr_label = tr_label.reshape(-1).unsqueeze(1)   #b*1*NK
        # print('这是tr_label.size2：', tr_label.size())
        one_hot = torch.zeros((b*nk,320),device=point_node.device)#将5全都改成了320
        tr_label = tr_label.to(torch.int64)
        one_hot.scatter_(-2,tr_label,-2)#将1改为了-2
        one_hot_fin = one_hot.reshape(b,nk,320)
        zero_pad = torch.zeros((b, 320,320),device= point_node.device).fill_(1.0/320) # zero-init or avg-init
        # zero_pad = torch.zeros((b, 5, 5),device= point_node.device).fill_(0.0) # zero-init or avg-init
        # print('one_hot_fin的size', one_hot_fin.size())
        # print('zero_pad的size', zero_pad.size())

        #Label self-attenion
        lab_new = torch.cat([one_hot_fin,zero_pad], dim=2)#将dim设置成2后 att_l从320,640,640变成了320,320,320
        # print('lab_new的size', lab_new.size())
        lab_t2 = torch.transpose(lab_new, 1, 2)
        # print('lab_t2的size', lab_t2.size())
        att_l = torch.bmm(lab_new, lab_t2)

        # Fusion layer
        # print('att_size',att.size())
        # print('att_l_size',att_l.size())
        mask_c = torch.cat([att.unsqueeze(1),att_l.unsqueeze(1)],dim=1)
        new_mask = self.fusion(mask_c).squeeze(1)
        a = 0.5 #修改
        # print('这是new_mask',new_mask.size())#[1, 320, 320]
        # print('这是point_node.size：',point_node.size())#[1, 320, 64]
        point_node = torch.bmm(new_mask, point_node)#[1, 320, 64]
        # print('这是point_node2.size：', point_node.size())
        # print('这是lab_new1',lab_new.size())#[1, 320, 640]
        lab_new = torch.mul(torch.bmm(new_mask,lab_new),1-a) +  torch.mul(lab_new,a)
        # print('这是lab_new2', lab_new.size())
        point_node = torch.cat([point_node,lab_new],dim=2)
        # print('这是point_node3：', point_node.size())#[1, 320, 704]

        # for l in range(self.generation):
        #
        #     point_edge, node_similarity_l2 = self._modules['point_sim_generation_{}'.format(l)](point_node, point_edge, self.point_metric)
        #     point_node_out = self._modules['distribution2point_generation_{}'.format(l)](point_edge, point_node)
        #
        #     # Layer memory attention (dense connection)
        #     point_node = torch.cat([point_node, point_node_out], dim=2)
        #
        #     point_similarities.append(point_edge)
        #     node_similarities_l2.append(node_similarity_l2)
        #     point_nodes.append(point_node)
        # return point_similarities, node_similarities_l2,point_nodes,point_edge
        for l in range(self.generation):
            point_edge, node_similarity_l2 = cp.checkpoint(self._modules['point_sim_generation_{}'.format(l)],
                                                           point_node, point_edge, self.point_metric)
            point_node_out = cp.checkpoint(self._modules['distribution2point_generation_{}'.format(l)], point_edge,
                                           point_node)
            point_node = torch.cat([point_node, point_node_out], dim=2)
            point_similarities.append(point_edge)
            node_similarities_l2.append(node_similarity_l2)
            point_nodes.append(point_node)
        return point_similarities, node_similarities_l2, point_nodes, point_edge
