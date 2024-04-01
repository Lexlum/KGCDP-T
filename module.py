# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from transformers import RobertaTokenizer, RobertaForMaskedLM
import math
import numpy as np
import torch.backends.cudnn as cudnn


class ListModule(nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class GraphEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, gnn_layers, embedding_size, initilized_embedding, hidden_size,
                 dropout_ratio=0.3):
        super(GraphEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.gnn_layers = gnn_layers
        self.embedding_size = embedding_size
        self.dropout_ratio = dropout_ratio
        self.half_hiddenstate = int(hidden_size / 2)
        self.node_embedding = nn.Embedding(num_nodes, embedding_size)
        self.node_embedding.from_pretrained(torch.from_numpy(np.load(initilized_embedding)), freeze=False,
                                            padding_idx=0)

        self.dropout = nn.Dropout(dropout_ratio)

        self.gnn = []
        for layer in range(gnn_layers):
            self.gnn.append(RGCNConv(embedding_size, embedding_size, self.num_relations, self.num_nodes))
            # if rgcn is too slow, you can use gcn
        self.gnn = ListModule(*self.gnn)

        self.fc = nn.Linear(embedding_size, self.half_hiddenstate)

        self.bigru = nn.GRU(self.half_hiddenstate, self.half_hiddenstate, num_layers=2, bidirectional=True,
                            batch_first=True)

    def forward(self, nodes, edges, types):
        """
        :param nodes: tensor, shape [batch_size, num_nodes]
        :param edges: List(List(edge_idx))
        :param types: List(type_idx)
        """
        batch_size = nodes.size(0)
        device = nodes.device

        # (batch_size, num_nodes, output_size)
        node_embeddings = []
        for bid in range(batch_size):
            embed = self.node_embedding(nodes[bid, :])
            edge_index = torch.as_tensor(edges[bid], dtype=torch.long, device=device)
            edge_type = torch.as_tensor(types[bid], dtype=torch.long, device=device)
            for lidx, rgcn in enumerate(self.gnn):
                if lidx == len(self.gnn) - 1:
                    embed = rgcn(embed, edge_index=edge_index, edge_type=edge_type)
                else:
                    embed = self.dropout(F.relu(rgcn(embed, edge_index=edge_index, edge_type=edge_type)))
            node_embeddings.append(embed)
        node_embeddings = torch.stack(node_embeddings, 0)  # [batch_size, num_node, embedding_size]
        nodeseq_embedding, _ = self.bigru(self.fc(node_embeddings))
        return nodeseq_embedding, self.node_embedding


class GraphReconstructor(nn.Module):
    def __init__(self, num_relations, hidden_size):
        super(GraphReconstructor, self).__init__()
        self.num_relations = num_relations
        self.hidden_size = hidden_size

        self.proj_linear = nn.Linear(3 * hidden_size, num_relations)

    def forward(self, pairs, hidden_states):
        """
        :param pairs: tensor [batch_size, num_pairs, 2, 2]
        :param hidden_states: tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, num_pairs = pairs.size(0), pairs.size(1)
        hidden_size = hidden_states.size(-1)

        head, tail = torch.chunk(pairs, chunks=2, dim=2)

        h_start, h_end = torch.chunk(head, chunks=2, dim=3)
        t_start, t_end = torch.chunk(tail, chunks=2, dim=3)

        hs_expand = h_start.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        hs_embed = torch.gather(hidden_states, dim=1, index=hs_expand)

        he_expand = h_end.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        he_embed = torch.gather(hidden_states, dim=1, index=he_expand)

        head_embed = (hs_embed + he_embed) / 2.0

        ts_expand = t_start.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        ts_embed = torch.gather(hidden_states, dim=1, index=ts_expand)

        te_expand = t_end.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        te_embed = torch.gather(hidden_states, dim=1, index=te_expand)

        tail_embed = (ts_embed + te_embed) / 2.0

        logits = self.proj_linear(torch.cat([head_embed, tail_embed, head_embed * tail_embed], dim=-1))

        return logits


class GraphPointer(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(GraphPointer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.proj_linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, embeddings, hidden_states, pointer):
        """
        :param embeddings: tensor [batch_size, seq_len, embedding_size]
        :param hidden_states: tensor [batch_size, seq_len, hidden_size]
        :param pointer: tensor [batch_size, seq_len]
        """
        copy_prob = torch.sigmoid(self.proj_linear(torch.cat([embeddings, hidden_states], dim=-1))).squeeze(-1)
        copy_prob = torch.where(pointer.bool(), 1 - copy_prob, copy_prob)

        return copy_prob


class Memory_Shift(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_relation, initilized_rel_embedding):
        super(Memory_Shift, self).__init__()
        self.relation_embedding = nn.Embedding(num_relation, hidden_size)
        self.relation_embedding.from_pretrained(torch.from_numpy(np.load(initilized_rel_embedding)), freeze=False,
                                                padding_idx=0)
        self.squeeze = nn.Linear(hidden_size * 2, hidden_size)
        self.aerfa_gate = nn.Linear(hidden_size, 1)
        self.relation_attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)

        self.w_d = nn.Linear(hidden_size, hidden_size)
        self.w_u = nn.Linear(hidden_size, 1)
        self.w_g = nn.Linear(hidden_size, hidden_size)

    def forward(self, batched_hidden_states, heads, tails, tri_mask, relations_idx, student_embeddings):
        tri_mask_repeat = tri_mask.unsqueeze(1).repeat(1, batched_hidden_states[0].size(-1), 1).permute(0, 2, 1)
        batchsize = batched_hidden_states[0].size(0)
        total_layer = len(batched_hidden_states)
        batchfirst_hidden_states = torch.stack(batched_hidden_states, dim=0).permute(1, 0, 2, 3)

        batched_qt = []
        for b in range(0, batchsize):
            hidden_states = batchfirst_hidden_states[b]
            splits_heads_embedding = [torch.stack([student_embeddings[b][nodeidx] for nodeidx in node], dim=0) for node
                                      in
                                      heads[b]]
            splits_tails_embedding = [torch.stack([student_embeddings[b][nodeidx] for nodeidx in node], dim=0) for node
                                      in
                                      tails[b]]
            heads_embedding = torch.stack([torch.sum(splits_head_embedding, dim=0) for splits_head_embedding in
                                           splits_heads_embedding], dim=0)
            tails_embedding = torch.stack([torch.sum(splits_tail_embedding, dim=0) for splits_tail_embedding in
                                           splits_tails_embedding], dim=0)
            u = self.squeeze(torch.concat((heads_embedding, tails_embedding), dim=-1))
            u_masked = u.masked_fill(torch.ne(tri_mask_repeat[b], 1), value=torch.tensor(0))

            # pasi related to relations
            relation = torch.stack([self.relation_embedding(rel) for rel in relations_idx[b]], dim=0)
            _, pasi = self.relation_attention(hidden_states, relation.repeat(total_layer, 1, 1),
                                              relation.repeat(total_layer, 1, 1),
                                              attn_mask=torch.ne(tri_mask[b], 1).repeat(hidden_states.size(1), 1))

            # update u according dt
            # for layer in range(0, total_layer):
            hiddenstates_onelayer = hidden_states[-1]
            aerfa = torch.sigmoid(self.aerfa_gate(hiddenstates_onelayer))  # aerfa_gate each timpstep
            wd = self.w_d(hiddenstates_onelayer)
            u_updatecontent = self.w_g(hiddenstates_onelayer)
            wu = []
            beta = []
            u_new = [u_masked]
            for t in range(0, hidden_states.size(1)):
                wu.append(self.w_u(u_new[-1]))
                beta.append(torch.mul(torch.sigmoid(wd[t] + wu[-1]), aerfa.unsqueeze(-1)))
                u_new.append(torch.mul(u_new[-1], 1 - beta[t][-1]) + torch.mul(u_updatecontent[t], beta[t][-1]))

            # qt
            qt = torch.sum(torch.mul(torch.stack(u_new[1:], dim=0), pasi[-1].unsqueeze(-1)), dim=1)
            batched_qt.append(qt)
        return torch.stack(batched_qt, dim=0)


class GenProjection(nn.Module):
    def __init__(self, hidden_size, num_node, num_vocab=50265):
        super(GenProjection, self).__init__()
        self.num_node = num_node
        self.gen_project = nn.Linear(hidden_size * 2, num_vocab)
        self.w_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_x = nn.Linear(hidden_size, hidden_size)
        self.pro_switch = nn.Linear(hidden_size, 1)

        self.copy_project = nn.Linear(hidden_size * 2, num_node)
        self.gsem_softgate = nn.Linear(hidden_size * 2, 1)
        self.memory_softgate = nn.Linear(hidden_size * 2, 1)

    def forward(self, hidden_states, qt, token_embedding, edges, nodeidx2bartidx):
        batchsize = token_embedding.size(0)
        m = [torch.zeros(batchsize, self.num_node).cuda()]
        gsem = []
        # 生成
        gen_pro = self.gen_project(torch.cat((hidden_states, qt), dim=-1))

        # gate for generate or copy
        decoder_feat = self.w_h(hidden_states)
        context_feat = self.w_s(qt)
        input_feat = self.w_x(token_embedding)
        gen_feat = context_feat + decoder_feat + input_feat
        pro_switch = torch.sigmoid(self.pro_switch(gen_feat))

        # copy
        copy_pro = self.copy_project(torch.cat((qt, token_embedding), dim=-1))


        # # GSEM
        Graph = []
        gsem_gate = torch.sigmoid(self.gsem_softgate(torch.cat((hidden_states, qt), dim=-1)))
        for b in range(0, batchsize):
            e = torch.tensor(edges[b])
            v = torch.ones_like(torch.Tensor(edges[b][0]))
            g = torch.sparse_coo_tensor(e, v, (self.num_node, self.num_node),
                                        dtype=torch.float32).to_dense().cuda()
            Graph.append(g)
        Graph = torch.stack(Graph, dim=0)

        memory_softgate = torch.sigmoid(self.memory_softgate(torch.cat((hidden_states, qt), dim=-1)))
        for t in range(0, hidden_states.size(1)):
            m_onestep = torch.mul(copy_pro[:, t, :], memory_softgate[:, t, :]) + \
                        torch.mul(m[-1], (1 - memory_softgate[:, t, :]))
            m.append(m_onestep)
            gsem_onestep = torch.matmul(Graph, m[-1].unsqueeze(-1)).squeeze()
            gsem.append(gsem_onestep)
        gsem = torch.stack(gsem, dim=1)
        pro_gsem_unexpended = torch.mul(copy_pro, gsem_gate) + torch.mul(gsem, (1 - gsem_gate))

        coverage = torch.softmax(torch.mul(pro_gsem_unexpended, pro_switch), dim=-1).sum(dim=1)

        # scatter_add
        index = torch.LongTensor([value for value in nodeidx2bartidx.values()]).cuda()
        gsem_pro = []
        for b in range(0, batchsize):
            pg = torch.zeros_like(gen_pro[b]).scatter_add(1, index.repeat(gen_pro.size(1), 1), pro_gsem_unexpended[b])
            gsem_pro.append(pg)
        gsem_pro = torch.stack(gsem_pro, dim=0)

        total_pro = torch.mul(gen_pro, (1 - pro_switch)) + torch.mul(gsem_pro, pro_switch)
        # total_pro = gen_pro
        return total_pro, coverage, pro_switch.squeeze()


class StaticPlanReconstruction(nn.Module):
    def __init__(self, num_nodes, hidden_size, embedding_size):
        super(StaticPlanReconstruction, self).__init__()
        self.hidden_size = hidden_size
        self.sp_reconstruction = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.hiddenstates_attention = nn.MultiheadAttention(self.hidden_size, num_heads=1, batch_first=True)
        self.sprec_project = nn.Linear(self.hidden_size * 2, num_nodes)
        self.embedding_size = embedding_size
        self.input_tra = nn.Linear(self.embedding_size, self.hidden_size)

    def forward(self, encoder_last_hiddenstates, hidden_states, nodes, gen_masks, vocabnode_embedding):
        batchsize = hidden_states.size(0)
        h = torch.zeros(batchsize, self.hidden_size).cuda()
        c = torch.zeros(batchsize, self.hidden_size).cuda()
        d_ini = encoder_last_hiddenstates
        q, att_weight = self.hiddenstates_attention(query=d_ini.unsqueeze(1), key=hidden_states, value=hidden_states,
                                                    attn_mask=torch.ne(gen_masks[:, 1:].unsqueeze(1), 1))
        input_lstm = hidden_states[:, -1, :]
        node_pro = []
        for i in range(0, nodes.size(1)):
            h, c = self.sp_reconstruction(input_lstm, (h, c))
            npro = self.sprec_project(torch.cat((h, q.squeeze()), dim=-1))
            node_pro.append(npro)
            q, att_weight = self.hiddenstates_attention(query=h.unsqueeze(1), key=hidden_states, value=hidden_states,
                                                        attn_mask=torch.ne(gen_masks[:, 1:].unsqueeze(1), 1))
            input_lstm = self.input_tra(vocabnode_embedding(nodes[:, i]))
        node_pro = torch.stack(node_pro, dim=0)
        return node_pro
