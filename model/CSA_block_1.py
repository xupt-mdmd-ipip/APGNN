from copy import deepcopy
import torch
from torch import nn
from sklearn import preprocessing
import numpy as np
import warnings
import math

warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

ETH_EPS = 1e-8


def normalize_keypoints(kpts, shape_or_size):
    if isinstance(shape_or_size, (tuple, list)):
        # it's a shape
        h, w = shape_or_size[-2:]
        size = kpts.new_tensor([[w, h]])
    else:
        # it's a size
        assert isinstance(shape_or_size, torch.Tensor)
        size = shape_or_size.to(kpts)
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7  # somehow we used 0.7 for SG
    return (kpts - c[:, None, :]) / f[:, None, :]


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class EndPtEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([5] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, endpoints, scores):
        # endpoints should be [B, N, 2, 2]
        # output is [B, feature_dim, N * 2]
        b_size, n_pts, _, _ = endpoints.shape
        assert tuple(endpoints.shape[-2:]) == (2, 2)
        endpt_offset = (endpoints[:, :, 1] - endpoints[:, :, 0]).unsqueeze(2)
        endpt_offset = torch.cat([endpt_offset, -endpt_offset], dim=2)
        endpt_offset = endpt_offset.reshape(b_size, 2 * n_pts, 2).transpose(1, 2)
        inputs = [endpoints.flatten(1, 2).transpose(1, 2),
                  endpt_offset, scores.repeat(1, 2).unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))  # 修改gropus的值为184
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def MLP2(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))  # 修改gropus的值为184
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))  # 添加池化层
    layers.append(nn.MaxPool1d(kernel_size=9, stride=1, padding=0))  #修改：试着分成多个maxpool1d从k=2一直到k=9
    return nn.Sequential(*layers)
# def MLP2(channels: list, do_bn=True):
#     """ Multi-layer perceptron """
#     n = len(channels)
#     layers = []
#     for i in range(1, n):
#         layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
#         if i < (n - 1):
#             if do_bn:
#                 layers.append(nn.BatchNorm1d(channels[i]))
#             layers.append(nn.ReLU())
#     layers.append(nn.Conv1d(channels[-1], channels[-1], kernel_size=1, bias=True))
#     layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))  # 添加池化层
#     layers.append(nn.MaxPool1d(kernel_size=9, stride=1, padding=0))
#     return nn.Sequential(*layers)



def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        '''
        source:源域数据（n * len(x))
        target:目标域数据（m * len(y))
        kernel_mul:核的倍数
        kernel_num:多少个核
        fix_sigma: 不同高斯核的sigma值
        '''
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
        # 将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # 调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        # loss = 0.0
        # print('这是f_of_X',f_of_X.size())
        # delta = f_of_X.mean(0).flatten() - f_of_Y.mean(0).flatten()
        # loss = delta.dot(delta.T)
        # return loss
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        # loss = torch.matmul(delta, delta.t())  # 使用`.t()`方法进行转置计算
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            # 将核矩阵分成4部分
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)  # 因为一般都是n==m，所以L矩阵一般不加入计算
                del XX, YY, XY, YX
            torch.cpu.empty_cache()
            return loss


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads, skip_init=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        if skip_init:
            self.register_parameter('scaling', nn.Parameter(torch.tensor(0.)))
        else:
            self.scaling = 1.

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)) * self.scaling


class GNNLayer(nn.Module):
    def __init__(self, feature_dim, layer_type, skip_init):
        super().__init__()
        assert layer_type in ['cross', 'self']
        self.type = layer_type
        self.update = AttentionalPropagation(feature_dim, 4, skip_init)

    def forward(self, desc0, desc1):
        if self.type == 'cross':
            src0, src1 = desc1, desc0
        elif self.type == 'self':
            src0, src1 = desc0, desc1
        else:
            raise ValueError("Unknown layer type: " + self.type)
        # self.update.attn.prob = []
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class LineLayer(nn.Module):
    def __init__(self, feature_dim, line_attention=False):
        super().__init__()
        self.dim = feature_dim
        self.mlp = MLP2([self.dim, self.dim * 2, self.dim], do_bn=True)  # 修改
        self.line_attention = line_attention
        if line_attention:
            self.proj_node = nn.Conv1d(self.dim, self.dim, kernel_size=1)
            self.proj_neigh = nn.Conv1d(2 * self.dim, self.dim, kernel_size=1)

    def get_endpoint_update(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]    #  1,720,320         1,320,320
        # and lines_junc_idx [bs, n_lines * 2]  # 320 320
        # Create one message per line endpoint
        # print('这是pnodes', point_nodes.size())
        # print('这是point_edge', point_edge.size())
        # point_nodes = point_nodes.cpu().detach().numpy()
        # point_nodes=np.transpose(point_nodes, (2, 0, 1))
        # point_nodes = torch.from_numpy(point_nodes)
        # device = point_nodes.device
        # point_edge = point_edge.to(device)
        # b_size = point_edge.shape[0]
        # point_edge = point_edge.type(torch.long)
        # # point_edge = point_edge.unsqueeze(1)
        # print('这是point_edge2', point_edge.size())
        # line_desc = torch.gather(point_nodes, 2, point_edge[:, None]).to(device)
        # print('这是line_desc', line_desc.size())
        # a=line_desc.reshape(b_size, 1, -1, 320).flip([-1]).flatten(2, 3).clone()
        # print('这是a',a.size())
        # message = torch.cat([
        #     line_desc,
        #     line_desc.reshape(b_size, 1, -1, 320).flip([-1]).flatten(2, 3).clone()], dim=1)
        # print('这是message', message.size())
        # message=message.cuda()
        # return self.mlp(message)  # [b_size, D, n_lines * 2]
        # b_size = lines_junc_idx.shape[0]
        # print('这是b_size', b_size)  # 320
        # print('这是dim', self.dim)  # 320
        # print('ldesc', ldesc.size())  # 1,320,720
        # print('line_enc', line_enc.size())  # 1,320,16
        # print('lines_junc_idx', lines_junc_idx.size())  # 320,320
        device = 'cpu'
        lines_junc_idx = lines_junc_idx.cpu()
        lines_junc_idx = lines_junc_idx.type(torch.int64)
        ldesc = ldesc.type(torch.int64)
        # print('这是ldesc', ldesc.size())
        # print('ldesc', ldesc.size())  # 1,320,720
        # a = lines_junc_idx[:, None].reshape(1, 320, 320)
        # print('a', a.size())  # 1,320,320
        line_desc = torch.gather(ldesc, 2, lines_junc_idx[:, None].reshape(1,320,320)) #修改
        # print('这是line_desc', line_desc.size())  # 1,320,320
        # print('这是line_enc1', line_enc.size())
        # print('line_desc.reshape(b_size, 1, -1, 320).flip([-1]).flatten(2, 3).clone()',
        #       line_desc.reshape(b_size, 1, -1, 320).flip([-1]).flatten(2, 3).clone().size())
        line_desc = line_desc.cpu()
        line_enc = line_enc.cpu()
        print('line_desc',line_desc.size())
        print('line_desc.reshape(1, 320, -1, 2).flip([-1]).flatten(2, 3).clone()', line_desc.reshape(1, 320, -1, 2).flip([-1]).flatten(2, 3).clone().size())
        print('line_enc', line_enc.size())
        message = torch.cat([
            line_desc,
            line_desc.reshape(1, 320, -1, 2).flip([-1]).flatten(2, 3).clone(),
            line_enc], dim=2)
        # print('这是message', message.size())  # 1,320,656
        message = message.cpu()
        # b = self.mlp(message)
        # print('这是mlp(message)的大小', b.size())
        return self.mlp(message)  # [b_size, D, n_lines * 2]   #1，320，656

    def get_endpoint_attention(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        # b_size = point_edge.shape[0]
        # expanded_lines_junc_idx = point_edge[:, None].repeat(1, self.dim, 1)
        #
        # # Query: desc of the current node
        # query = self.proj_node(p_nodes)  # [b_size, D, n_junc]
        # query = torch.gather(query, 2, expanded_lines_junc_idx)
        # # query is [b_size, D, n_lines * 2]
        #
        # # Key: combination of neighboring desc and line encodings
        # line_desc = torch.gather(p_nodes, 2, expanded_lines_junc_idx)
        # key = self.proj_neigh(torch.cat([
        #     line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone()], dim=1))  # [b_size, D, n_lines * 2]
        #
        # # Compute the attention weights with a custom softmax per junction
        # prob = (query * key).sum(dim=1) / self.dim ** .5  # [b_size, n_lines * 2]
        # prob = torch.exp(prob - prob.max())
        # denom = torch.zeros_like(p_nodes[:, 0]).scatter_reduce_(
        #     dim=1, index=point_edge,
        #     src=prob, reduce='sum', include_self=False)  # [b_size, n_junc]
        # denom = torch.gather(denom, 1, point_edge)  # [b_size, n_lines * 2]
        # prob = prob / (denom + ETH_EPS)
        # return prob  # [b_size, n_lines * 2]
        b_size = lines_junc_idx.shape[0]
        expanded_lines_junc_idx = lines_junc_idx[:, None].repeat(1, self.dim, 1)

        # Query: desc of the current node
        query = self.proj_node(ldesc)  # [b_size, D, n_junc]
        query = torch.gather(query, 2, expanded_lines_junc_idx)
        # query is [b_size, D, n_lines * 2]

        # Key: combination of neighboring desc and line encodings
        line_desc = torch.gather(ldesc, 2, expanded_lines_junc_idx)
        key = self.proj_neigh(torch.cat([
            line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(),
            line_enc], dim=1))  # [b_size, D, n_lines * 2]

        # Compute the attention weights with a custom softmax per junction
        prob = (query * key).sum(dim=1) / self.dim ** .5  # [b_size, n_lines * 2]
        prob = torch.exp(prob - prob.max())
        denom = torch.zeros_like(ldesc[:, 0]).scatter_reduce_(
            dim=1, index=lines_junc_idx,
            src=prob, reduce='sum', include_self=False)  # [b_size, n_junc]
        denom = torch.gather(denom, 1, lines_junc_idx)  # [b_size, n_lines * 2]
        prob = prob / (denom + ETH_EPS)
        return prob  # [b_size, n_lines * 2]

    def forward(self, ldesc0, ldesc1, line_enc0, line_enc1, lines_junc_idx0,
                lines_junc_idx1):
        # Gather the endpoint updates
        # lupdate0 = self.get_endpoint_update(point_nodes_src, point_edge_src)
        # lupdate1 = self.get_endpoint_update(point_nodes_tar, point_edge_tar)
        #
        # update0, update1 = torch.zeros_like(point_nodes_src), torch.zeros_like(point_nodes_tar)
        # dim = point_nodes_src.shape[1]
        # if self.line_attention:
        #     # Compute an attention for each neighbor and do a weighted average
        #     prob0 = self.get_endpoint_attention(point_nodes_src, point_edge_src)
        #     lupdate0 = lupdate0 * prob0[:, None]
        #     update0 = update0.scatter_reduce_(
        #         dim=2, index=point_edge_src[:, None].repeat(1, dim, 1),
        #         src=lupdate0, reduce='sum', include_self=False)
        #     prob1 = self.get_endpoint_attention(point_nodes_tar, point_edge_tar)
        #     lupdate1 = lupdate1 * prob1[:, None]
        #     update1 = update1.scatter_reduce_(
        #         dim=2, index=point_edge_tar[:, None].repeat(1, dim, 1),
        #         src=lupdate1, reduce='sum', include_self=False)
        # else:
        #     # Average the updates for each junction (requires torch > 1.12)
        #     b=point_edge_src[:, None]
        #     print('这是b111111',b.size())
        #     # update0 = update0.scatter_reduce_(
        #     #     dim=2, index = point_edge_src[:, None].repeat(1, dim, 1).long(),
        #     #     src=lupdate0, reduce='mean', include_self=False)
        #
        #     update0 = update0.expand(320, 720, 320).clone()
        #     update0 = update0.scatter_reduce_(dim=2,index=point_edge_src[:, None].repeat(1, dim,1).long(),
        #                                                                 src=lupdate0, reduce='mean', include_self=False)
        # update1= update1.expand(320, 720, 320).clone()
        # update1 = update1.scatter_reduce_(dim=2,index=point_edge_tar[:, None].repeat(1, dim,1).long(),
        #                                                             src=lupdate1, reduce='mean', include_self=False)

        # 修改
        # Update
        # point_nodes_src = point_nodes_src + update0
        # point_nodes_tar = point_nodes_tar + update0
        #
        # return point_nodes_src, point_nodes_tar
        lupdate0 = self.get_endpoint_update(ldesc0, line_enc0, lines_junc_idx0)
        lupdate1 = self.get_endpoint_update(ldesc1, line_enc1, lines_junc_idx1)

        update0, update1 = torch.zeros_like(ldesc0), torch.zeros_like(ldesc1)
        # print('update0', update0.size())
        dim = ldesc0.shape[1]
        if self.line_attention:
            # Compute an attention for each neighbor and do a weighted average
            prob0 = self.get_endpoint_attention(ldesc0, line_enc0,
                                                lines_junc_idx0)
            lupdate0 = lupdate0 * prob0[:, None]
            update0 = update0.scatter_reduce_(
                dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1),
                src=lupdate0, reduce='sum', include_self=False)
            prob1 = self.get_endpoint_attention(ldesc1, line_enc1,
                                                lines_junc_idx1)
            lupdate1 = lupdate1 * prob1[:, None]
            update1 = update1.scatter_reduce_(
                dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1),
                src=lupdate1, reduce='sum', include_self=False)
        else:
            # Average the updates for each junction (requires torch > 1.12)
            b = lines_junc_idx0
            # print('这是b111111', b.size())
            lines_junc_idx0 = lines_junc_idx0.unsqueeze(0).repeat(update0.shape[0], 1, 1)
            lines_junc_idx1 = lines_junc_idx1.unsqueeze(0).repeat(update1.shape[0], 1, 1)
            update0 = update0.scatter_reduce_(dim=2, index=lines_junc_idx0.long(), src=lupdate0, reduce='mean',
                                              include_self=False)
            update1 = update1.scatter_reduce_(dim=2, index=lines_junc_idx1.long(), src=lupdate1, reduce='mean',
                                              include_self=False)

        # Update
        ldesc0 = ldesc0 + update0
        ldesc1 = ldesc1 + update1

        return ldesc0, ldesc1


class AttentionalGNN(nn.Module):
    def __init__(self, num_support, feature_dim: int, layer_types: list, checkpointed=True,
                 skip=False, inter_supervision=None, num_line_iterations=1,
                 line_attention=False):
        super().__init__()
        self.checkpointed = checkpointed
        self.inter_supervision = inter_supervision
        self.num_line_iterations = num_line_iterations
        self.inter_layers = {}
        self.layers = nn.ModuleList([
            GNNLayer(feature_dim, layer_type, skip)
            for layer_type in layer_types])
        # print('这是layers', self.layers)
        layers_enum = list(enumerate(self.layers))
        # print('这是len(enumerate(self.layers))::::::::::::::')  # 9
        print(len(layers_enum))
        self.line_layers = nn.ModuleList([
            LineLayer(feature_dim, line_attention)
            for _ in range(len(layer_types) // 2)])
        # print('这是linelayers：', self.line_layers)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])

        self.mmd = MMD_loss(kernel_type='linear')

    # def forward(self, p_nodes_src, p_nodes_tar,):
    #         global AttentionalGNN_return
    #         flag = 0
    #         dev = p_nodes_src[0].device
    #         for i in range(len(p_nodes_src)):
    #             p_nodes_src[i] = torch.tensor(
    #                     preprocessing.scale(p_nodes_src[i].squeeze(0).cpu().detach().numpy())).to(
    #                     dev).unsqueeze(0).transpose(2, 1)
    #             p_nodes_tar[i] = torch.tensor(
    #                     preprocessing.scale(p_nodes_tar[i].squeeze(0).cpu().detach().numpy())).to(
    #                     dev).unsqueeze(0).transpose(2, 1)
    #
    #
    #             for layer, name in zip(self.layers, self.names):
    #                 if name == 'cross':
    #                     p_src0, p_src1 = p_nodes_tar[flag], p_nodes_src[flag]
    #                 delta0, delta1 = layer(p_nodes_src[flag], p_src0, p_src0), layer(p_nodes_tar[flag], p_src1, p_src1)
    #                 p_nodes_src_temp, p_nodes_tar_temp = torch.einsum('bij,bij->bij', delta0, p_src0), torch.einsum(
    #                     'bij,bij->bij', delta1, p_src1)
    #                 delta0, delta1 = layer(p_nodes_src_temp), layer(p_nodes_tar_temp)
    #                 p_nodes_src[0] = p_nodes_src[0] + self.mlp(torch.cat([p_nodes_src[flag], delta0], dim=1))
    #                 p_nodes_tar[0] = p_nodes_tar[0] + self.mlp(torch.cat([p_nodes_tar[flag], delta1], dim=1))
    #
    #                 flag += 1
    #                 AttentionalGNN_return = self.mmd(p_nodes_src[0].squeeze(0).transpose(1, 0),
    #                                                  p_nodes_tar[0].squeeze(0).transpose(1, 0))
    #
    #         return AttentionalGNN_return
    def forward(self, point_nodes_src, point_nodes_tar, distribution_nodes_src, distribution_nodes_tar, point_edge_src,
                point_edge_tar):
        global AttentionalGNN_return
        dev = point_nodes_src[0].cpu()
        flag = 0
        for i in range(len(point_nodes_src)):
            # print('len(point_nodes_src):::::', len(point_nodes_src))
            point_nodes_src[i] = torch.tensor(
                preprocessing.scale(point_nodes_src[i].squeeze(0).cpu().detach().numpy())).cpu().unsqueeze(0)
            point_nodes_tar[i] = torch.tensor(
                preprocessing.scale(point_nodes_tar[i].squeeze(0).cpu().detach().numpy())).cpu().unsqueeze(0)
            distribution_nodes_src[i] = torch.tensor(
                preprocessing.scale(distribution_nodes_src[i].squeeze(0).cpu().detach().numpy())).cpu().unsqueeze(0)
            distribution_nodes_tar[i] = torch.tensor(
                preprocessing.scale(distribution_nodes_tar[i].squeeze(0).cpu().detach().numpy())).cpu().unsqueeze(0)
            point_edge_src[i] = torch.tensor(
                preprocessing.scale(point_edge_src[i].squeeze(0).cpu().detach().numpy())).cpu().unsqueeze(0)
            point_edge_tar[i] = torch.tensor(
                preprocessing.scale(point_edge_tar[i].squeeze(0).cpu().detach().numpy())).cpu().unsqueeze(0)
            # print('point_nodes_src[i]::::::::::::', point_nodes_src[i].size())
            # print('当前i=', i)

        for j, layer in enumerate(self.layers):
            if self.checkpointed:
                point_nodes_src[flag], point_nodes_tar[flag] = torch.utils.checkpoint.checkpoint(
                    layer, point_nodes_src[flag], point_nodes_tar[flag], preserve_rng_state=False)
            else:
                # print('这是point_nodes_src[flag]维度', point_nodes_src[flag].size())
                point_nodes_src[flag], point_nodes_tar[flag] = layer(point_nodes_src[flag], point_nodes_tar[flag])
            if (layer.type == 'self' and point_edge_src[flag].shape[1] > 0
                    and point_edge_tar[flag].shape[1] > 0):
                # Add line self attention layers after every self layer
                for _ in range(self.num_line_iterations):
                    if self.checkpointed:
                        if 0 <= j < 8:
                            point_nodes_src[flag], point_nodes_tar[flag] = torch.utils.checkpoint.checkpoint(
                                self.line_layers[j // 2],
                                point_nodes_src[flag], point_nodes_tar[flag], distribution_nodes_src[flag],
                                distribution_nodes_tar[flag], point_edge_src[flag], point_edge_tar[flag],
                                preserve_rng_state=False)
                    else:
                        if 0 <= j < 8:
                            # print('j=', j)
                            # print('range(self.num_line_iterations)=', range(self.num_line_iterations))
                            # print('flag=', flag)
                            point_nodes_src[flag], point_nodes_tar[flag] = self.line_layers[j // 2](
                                point_nodes_src[flag], point_nodes_tar[flag], distribution_nodes_src[flag],
                                distribution_nodes_tar[flag], point_edge_src[flag], point_edge_tar[flag])

            # Optionally store the line descriptor at intermediate layers
            if (self.inter_supervision is not None
                    and (j // 2) in self.inter_supervision
                    and layer.type == 'cross'):
                self.inter_layers[j // 2] = (point_nodes_src[flag].clone(), point_nodes_tar[flag].clone())

            AttentionalGNN_return = self.mmd(point_nodes_src[0].squeeze(0).transpose(1, 0),
                                             point_nodes_tar[0].squeeze(0).transpose(1, 0))  # 修改：看看能不能加上边

            # AttentionalGNN_return2=self.mmd(point_edge_src[0].squeeze(0).transpose(1, 0),
            #                          point_edge_tar[0].squeeze(0).transpose(1, 0))
            # AttentionalGNN_return=AttentionalGNN_return1+AttentionalGNN_return2

        return AttentionalGNN_return

    def _get_matches(self, scores_mat):
        max0 = scores_mat[:, :-1, :-1].max(2)
        max1 = scores_mat[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores_mat.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf.filter_threshold)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))
        return m0, m1, mscores0, mscores1

    def _get_line_matches(self, ldesc0, ldesc1, lines_junc_idx0,
                          lines_junc_idx1, final_proj):
        mldesc0 = final_proj(ldesc0)
        mldesc1 = final_proj(ldesc1)

        line_scores = torch.einsum('bdn,bdm->bnm', mldesc0, mldesc1)
        line_scores = line_scores / self.conf.descriptor_dim ** .5

        # Get the line representation from the junction descriptors
        n2_lines0 = lines_junc_idx0.shape[1]
        n2_lines1 = lines_junc_idx1.shape[1]
        line_scores = torch.gather(
            line_scores, dim=2,
            index=lines_junc_idx1[:, None, :].repeat(1, line_scores.shape[1], 1))
        line_scores = torch.gather(
            line_scores, dim=1,
            index=lines_junc_idx0[:, :, None].repeat(1, 1, n2_lines1))
        line_scores = line_scores.reshape((-1, n2_lines0 // 2, 2,
                                           n2_lines1 // 2, 2))

        # Match either in one direction or the other
        raw_line_scores = 0.5 * torch.maximum(
            line_scores[:, :, 0, :, 0] + line_scores[:, :, 1, :, 1],
            line_scores[:, :, 0, :, 1] + line_scores[:, :, 1, :, 0])
        line_scores = log_double_softmax(raw_line_scores, self.line_bin_score)
        m0_lines, m1_lines, mscores0_lines, mscores1_lines = self._get_matches(
            line_scores)
        return (line_scores, m0_lines, m1_lines, mscores0_lines,
                mscores1_lines, raw_line_scores)

    def loss(self, pred, data):
        raise NotImplementedError()

    def metrics(self, pred, data):
        raise NotImplementedError()


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, :-1, -1] = scores0[:, :, -1]
    scores[:, -1, :-1] = scores1[:, -1, :]
    return scores


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


if __name__ == '__main__':
    model = AttentionalGNN(num_support=16, feature_dim=64, layer_names=['GNN_layers'])
    model.eval()
    print(model)
    # input = torch.randn(64, 30, 15, 15)
    # y = model(input)
    # print(y.size())
