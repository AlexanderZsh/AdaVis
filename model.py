#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import *
import random
import pickle
import math
from icecream import ic
from torch.cuda.amp import autocast
from sklearn.metrics import ndcg_score

ic.configureOutput(includeContext=True)
from tqdm import tqdm


def Identity(x):
    return x


def cal_recall(preds, gt):
    return len(set(gt).intersection(preds)) / len(gt)


def cal_prec(preds, gt):
    return len(set(gt).intersection(preds)) / len(preds)


def cal_F1(recall, prec):
    if prec == 0 and recall == 0:
        return 0
    return (2 * recall * prec) / (prec + recall)


class SetIntersection(nn.Module):
    def __init__(self, mode_dims, expand_dims, agg_func=torch.min):
        super(SetIntersection, self).__init__()
        self.agg_func = agg_func
        self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat", self.pre_mats)
        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat", self.post_mats)
        self.pre_mats_im = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats_im)
        self.register_parameter("premat_im", self.pre_mats_im)
        self.post_mats_im = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats_im)
        self.register_parameter("postmat_im", self.post_mats_im)

    def forward(self, embeds1, embeds2, embeds3=[], name="real"):
        if name == "real":
            temp1 = F.relu(embeds1.mm(self.pre_mats))
            temp2 = F.relu(embeds2.mm(self.pre_mats))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats)

        elif name == "img":
            temp1 = F.relu(embeds1.mm(self.pre_mats_im))
            temp2 = F.relu(embeds2.mm(self.pre_mats_im))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats_im))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats_im)
        return combined


class CenterSet(nn.Module):
    def __init__(
        self, mode_dims, expand_dims, center_use_offset, agg_func=torch.min, bn="no", nat=1, name="Real_center"
    ):
        super(CenterSet, self).__init__()
        assert nat == 1, "vanilla method only support 1 nat now"
        self.center_use_offset = center_use_offset
        self.agg_func = agg_func
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))

        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat_%s" % name, self.pre_mats)
        if bn != "no":
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn2 = nn.BatchNorm1d(mode_dims)
            self.bn3 = nn.BatchNorm1d(mode_dims)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s" % name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        if self.center_use_offset:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1
            temp2 = embeds2
            if len(embeds3) > 0:
                temp3 = embeds3

        if self.bn == "no":
            temp1 = F.relu(temp1.mm(self.pre_mats))
            temp2 = F.relu(temp2.mm(self.pre_mats))
        elif self.bn == "before":
            temp1 = F.relu(self.bn1(temp1.mm(self.pre_mats)))
            temp2 = F.relu(self.bn2(temp2.mm(self.pre_mats)))
        elif self.bn == "after":
            temp1 = self.bn1(F.relu(temp1.mm(self.pre_mats)))
            temp2 = self.bn2(F.relu(temp2.mm(self.pre_mats)))
        if len(embeds3) > 0:
            if self.bn == "no":
                temp3 = F.relu(temp3.mm(self.pre_mats))
            elif self.bn == "before":
                temp3 = F.relu(self.bn3(temp3.mm(self.pre_mats)))
            elif self.bn == "after":
                temp3 = self.bn3(F.relu(temp3.mm(self.pre_mats)))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined


class MeanSet(nn.Module):
    def __init__(self):
        super(MeanSet, self).__init__()

    def forward(self, embeds, embeds_o):

        return torch.mean(torch.stack(embeds, dim=0), dim=0)


class MinSet(nn.Module):
    def __init__(self):
        super(MinSet, self).__init__()

    def forward(self, embeds, embeds_o):
        assert len(embeds_o) == 46
        return torch.min(torch.stack(embeds_o, dim=0), dim=0)[0]


class OffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, agg_func=torch.min, name="Real_offset"):
        super(OffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        if offset_use_center:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s" % name, self.pre_mats)
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s" % name, self.pre_mats)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s" % name, self.post_mats)



    def forward(self, embeds, embeds_o):
        if self.offset_use_center:
            temp_list = []
            for emb, emb_offset in zip(embeds, embeds_o):
                temp = torch.cat([emb, emb_offset], dim=1)
                temp = F.relu(temp.mm(self.pre_mats))
                temp_list.append(temp)
        else:
            temp_list = []
            for emb_offset in embeds_o:
                temp = emb_offset
                temp = F.relu(temp.mm(self.pre_mats))
                temp_list.append(temp)

        combined = torch.stack(temp_list)

        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined


class InductiveOffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, off_reg, agg_func=torch.min, name="Real_offset"):
        super(InductiveOffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.OffsetSet_Module = OffsetSet(mode_dims, expand_dims, offset_use_center, self.agg_func)



    def forward(self, embeds, embeds_o):
        offset_min = torch.mean(torch.stack([emb_o for emb_o in embeds_o]), dim=0)[0]
        offset = offset_min * torch.sigmoid(self.OffsetSet_Module(embeds, embeds_o))

        return offset


class AttentionSet(nn.Module):
    def __init__(
        self,
        mode_dims,
        expand_dims,
        center_use_offset,
        att_reg=0.0,
        att_tem=1.0,
        att_type="whole",
        bn="no",
        nat=1,
        name="Real",
    ):
        super(AttentionSet, self).__init__()
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.Attention_module = Attention(mode_dims, expand_dims, center_use_offset, att_type=att_type, bn=bn, nat=nat)


    def forward(self, embeds, embeds_o):
        """
        input:
            embeds: tuple: ((inter_head_1, batch_size,dim),..)
            embeds_o: tuple: ((inter_head_1, batch_size, dim),..)
        output:
            combined: (agg_head, batch_size, dim)
            center: (batch_size, dim)
        """

        temp_list = []
        # * Box Embedding
        if embeds_o is not None:
            for embed, embed_offset in zip(embeds, embeds_o):
                # * temp is the attention weight of a feature (single value)
                temp = (self.Attention_module(embed, embed_offset) + self.att_reg) / (self.att_tem + 1e-4)
                temp_list.append(temp)
        else:  # * TransE
            for embed in embeds:
                temp = (self.Attention_module(embed, None) + self.att_reg) / (self.att_tem + 1e-4)
                temp_list.append(temp)

        if self.att_type == "whole":
            combined = F.softmax(torch.cat(temp_list, dim=1), dim=1)
            center = 0
            for i in range(len(temp_list)):
                center += embeds[i] * (combined[:, i].view(embeds[i].size(0), 1))

        elif self.att_type == "ele":
            center = 0
            # * softmax dim=0, get the weight for element on the each row, same position
            combined = F.softmax(torch.stack(temp_list), dim=0)
            for i in range(len(temp_list)):

                center += embeds[i] * combined[i]

        return center


class Attention(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_type, bn, nat, name="Real"):
        super(Attention, self).__init__()
        self.center_use_offset = center_use_offset
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims * 2, mode_dims))
        else:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats1)
        self.register_parameter("atten_mats1_%s" % name, self.atten_mats1)
        if self.nat >= 2:
            self.atten_mats1_1 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_1)
            self.register_parameter("atten_mats1_1_%s" % name, self.atten_mats1_1)
        if self.nat >= 3:
            self.atten_mats1_2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_2)
            self.register_parameter("atten_mats1_2_%s" % name, self.atten_mats1_2)
        if bn != "no":
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn1_1 = nn.BatchNorm1d(mode_dims)
            self.bn1_2 = nn.BatchNorm1d(mode_dims)
        if att_type == "whole":
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, 1))
        elif att_type == "ele":
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats2)
        self.register_parameter("atten_mats2_%s" % name, self.atten_mats2)

    def forward(self, center_embed, offset_embed=None):
        if self.center_use_offset:
            temp1 = torch.cat([center_embed, offset_embed], dim=1)
        else:
            temp1 = center_embed
        if self.nat >= 1:
            if self.bn == "no":
                temp2 = F.relu(temp1.mm(self.atten_mats1))
            elif self.bn == "before":
                temp2 = F.relu(self.bn1(temp1.mm(self.atten_mats1)))
            elif self.bn == "after":
                temp2 = self.bn1(F.relu(temp1.mm(self.atten_mats1)))
        if self.nat >= 2:
            if self.bn == "no":
                temp2 = F.relu(temp2.mm(self.atten_mats1_1))
            elif self.bn == "before":
                temp2 = F.relu(self.bn1_1(temp2.mm(self.atten_mats1_1)))
            elif self.bn == "after":
                temp2 = self.bn1_1(F.relu(temp2.mm(self.atten_mats1_1)))
        if self.nat >= 3:
            if self.bn == "no":
                temp2 = F.relu(temp2.mm(self.atten_mats1_2))
            elif self.bn == "before":
                temp2 = F.relu(self.bn1_2(temp2.mm(self.atten_mats1_2)))
            elif self.bn == "after":
                temp2 = self.bn1_2(F.relu(temp2.mm(self.atten_mats1_2)))
        temp3 = temp2.mm(self.atten_mats2)
        return temp3


class Query2box(nn.Module):
    def __init__(
        self,
        model_name,
        nentity,
        nrelation,
        hidden_dim,
        gamma,
        writer=None,
        geo=None,
        cen=None,
        offset_deepsets=None,
        center_deepsets=None,
        offset_use_center=None,
        center_use_offset=None,
        att_reg=0.0,
        off_reg=0.0,
        att_tem=1.0,
        euo=False,
        gamma2=0,
        bn="no",
        nat=1,
        activation="relu",
    ):
        super(Query2box, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.writer = writer
        self.geo = geo
        self.cen = cen
        self.offset_deepsets = offset_deepsets
        self.center_deepsets = center_deepsets
        self.offset_use_center = offset_use_center
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.off_reg = off_reg
        self.att_tem = att_tem
        self.euo = euo
        self.his_step = 0
        self.bn = bn
        self.nat = nat
        if activation == "none":
            self.func = Identity
        elif activation == "relu":
            self.func = F.relu
        elif activation == "softplus":
            self.func = F.softplus

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        if gamma2 == 0:
            gamma2 = gamma

        self.gamma2 = nn.Parameter(torch.Tensor([gamma2]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        if self.geo == "vec":
            if self.center_deepsets == "vanilla":
                self.deepsets = CenterSet(
                    self.relation_dim, self.relation_dim, False, agg_func=torch.mean, bn=bn, nat=nat
                )
            elif self.center_deepsets == "attention":
                self.deepsets = AttentionSet(
                    self.relation_dim,
                    self.relation_dim,
                    False,
                    att_reg=self.att_reg,
                    att_tem=self.att_tem,
                    bn=bn,
                    nat=nat,
                )
            elif self.center_deepsets == "eleattention":
                self.deepsets = AttentionSet(
                    self.relation_dim,
                    self.relation_dim,
                    False,
                    att_reg=self.att_reg,
                    att_type="ele",
                    att_tem=self.att_tem,
                    bn=bn,
                    nat=nat,
                )
            elif self.center_deepsets == "mean":
                self.deepsets = MeanSet()
            else:
                assert False

        if self.geo == "box":
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(tensor=self.offset_embedding, a=0.0, b=self.embedding_range.item())
            if self.euo:
                self.entity_offset_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
                nn.init.uniform_(tensor=self.entity_offset_embedding, a=0.0, b=self.embedding_range.item())

            if self.center_deepsets == "vanilla":
                self.center_sets = CenterSet(
                    self.relation_dim, self.relation_dim, self.center_use_offset, agg_func=torch.mean, bn=bn, nat=nat
                )
            elif self.center_deepsets == "attention":
                self.center_sets = AttentionSet(
                    self.relation_dim,
                    self.relation_dim,
                    self.center_use_offset,
                    att_reg=self.att_reg,
                    att_tem=self.att_tem,
                    bn=bn,
                    nat=nat,
                )
            elif self.center_deepsets == "eleattention":
                self.center_sets = AttentionSet(
                    self.relation_dim,
                    self.relation_dim,
                    self.center_use_offset,
                    att_reg=self.att_reg,
                    att_type="ele",
                    att_tem=self.att_tem,
                    bn=bn,
                    nat=nat,
                )
            elif self.center_deepsets == "mean":
                self.center_sets = MeanSet()
            else:
                assert False

            if self.offset_deepsets == "vanilla":
                self.offset_sets = OffsetSet(
                    self.relation_dim, self.relation_dim, self.offset_use_center, agg_func=torch.mean
                )
            elif self.offset_deepsets == "inductive":
                self.offset_sets = InductiveOffsetSet(
                    self.relation_dim, self.relation_dim, self.offset_use_center, self.off_reg, agg_func=torch.mean
                )
            elif self.offset_deepsets == "min":
                self.offset_sets = MinSet()
            else:
                assert False

        if model_name not in ["TransE", "BoxTransE"]:
            raise ValueError("model %s not supported" % model_name)

    def vec_inference(self, head, relation, rel_len, qtype, tail=None):

        """
        head: [batch_size * head_num, 1, dim]
        relation: [batch_size * head_inter, rel, 1, dim]
        """
        if qtype == "chain-inter":
            raise "not chain-inter yet"

        elif qtype == "inter-chain":

            # * rel_len + 1 for an extra chain projection
            relations = torch.chunk(relation, rel_len + 1, dim=0)
            query_center_list = []
            heads = torch.chunk(head, rel_len, dim=0)

            for i in range(rel_len):
                query_center = heads[i] + relations[i][:, 0, :, :]
                query_center_list.append(query_center.squeeze(1))

            conj_query_center = self.deepsets(query_center_list, None).unsqueeze(1)
            # * rel_len is the last chain project index
            new_query_center = conj_query_center + relations[rel_len][:, 0, :, :]
            # NOTE: calculate the model score

            score = new_query_center - tail

            score = self.gamma.item() - torch.norm(score, p=1, dim=-1)

            return score, None, None, 0.0, []


        else:
            # * n-chain or n-inter
            query_center = head
            for rel in range(rel_len):
                query_center = query_center + relation[:, rel, :, :]

            # * chain
            if "inter" not in qtype and "union" not in qtype:

                result_query_center = query_center

            # * inter
            else:
                rel_len = int(qtype.split("-")[0])
                assert rel_len > 1

                # * queries_center are divied into rel_len parts: (query_1:[batch_size,dim],...,query_rel_len:[batch_size, dim])
                query_center = query_center.squeeze(1)
                queries_center = torch.chunk(query_center, rel_len, dim=0)

                # * get offset for each instance in batch
                if "inter" in qtype:
                    result_query_center = self.deepsets(queries_center, None)

                else:
                    assert False, "qtype not exists: %s" % qtype
        return result_query_center

    def inference(self, head, relation, mode, offset, head_offset, rel_len, qtype, tail=None):

        """
        head: [batch_size * head_num, 1, dim]
        head_offset: [batch_size * head_num, 1, dim]
        relation: [batch_size * head_inter, rel, 1, dim]
        offset:
            inter:[batch_size*rel_len, 1, 1, dim]
            chain:[batch_size, rel_len, 1, dim]
        """
        if qtype == "chain-inter":
            raise "not chain-inter yet"

        elif qtype == "inter-chain":

            # * rel_len + 1 for an extra chain projection
            relations = torch.chunk(relation, rel_len + 1, dim=0)
            offsets = torch.chunk(offset, rel_len + 1, dim=0)
            if head_offset is not None:
                head_offsets = torch.chunk(head_offset, rel_len, dim=0)

            query_center_list = []
            heads = torch.chunk(head, rel_len, dim=0)

            for i in range(rel_len):
                query_center = heads[i] + relations[i][:, 0, :, :]
                query_center_list.append(query_center.squeeze(1))

            query_min_list = []
            query_max_list = []
            if head_offset is not None:
                for i in range(rel_len):
                    query_min = (
                        query_center_list[i]
                        - 0.5 * self.func(offsets[i][:, 0, :, :])
                        - 0.5 * self.func(head_offsets[i])
                    )

                    query_max = (
                        query_center_list[i]
                        + 0.5 * self.func(offsets[i][:, 0, :, :])
                        + 0.5 * self.func(head_offsets[i])
                    )
                    query_min_list.append(query_min)
                    query_max_list.append(query_max)
            else:
                for i in range(rel_len):
                    query_min = query_center_list[i] - 0.5 * self.func(offsets[i][:, 0, :, :])
                    query_max = query_center_list[i] + 0.5 * self.func(offsets[i][:, 0, :, :])
                    query_min_list.append(query_min)
                    query_max_list.append(query_max)
            assert len(query_min_list) == rel_len & len(query_max_list) == rel_len

            offset_list = []
            for i in range(rel_len):
                offset_temp = (query_max_list[i] - query_min_list[i]).squeeze(1)
                offset_list.append(offset_temp)

            conj_query_center = self.center_sets(query_center_list, offset_list).unsqueeze(1)

            new_query_center = conj_query_center + relations[rel_len][:, 0, :, :]
            # NOTE: calculate the model score
            new_offset = self.offset_sets(query_center_list, offset_list).unsqueeze(1)

            new_query_min = (
                new_query_center - 0.5 * self.func(new_offset) - 0.5 * self.func(offsets[rel_len][:, 0, :, :])
            )

            new_query_max = (
                new_query_center + 0.5 * self.func(new_offset) + 0.5 * self.func(offsets[rel_len][:, 0, :, :])
            )
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center
            result_offset = new_query_max - new_query_min
            # # * distance outside score
            score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)

            x_in_box = torch.min(new_query_max, torch.max(new_query_min, tail))
            padding = torch.min(x_in_box - new_query_min, new_query_max - x_in_box)

            score_center_plus = (
                self.gamma.item()
                - torch.norm(score_offset, p=1, dim=-1)
                - torch.norm(score_center, p=1, dim=-1)
                + torch.norm(padding, p=1, dim=-1)
            )
            score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)
            assert score_center_plus.size() == score.size()
            return (
                score,
                score_center,
                torch.mean(torch.norm(result_offset, p=2, dim=2).squeeze(1)),
                score_center_plus,
                None,
            )
        elif qtype == "union-chain":
            raise "not union-chain yet"

        else:
            # * n-chain or n-inter
            query_center = head
            for rel in range(rel_len):
                query_center = query_center + relation[:, rel, :, :]

            query_min = query_center
            query_max = query_center
            if head_offset is not None:
                for rel in range(0, rel_len):
                    query_min = query_min - 0.5 * self.func(offset[:, rel, :, :]) - 0.5 * self.func(head_offset)
                    query_max = query_max + 0.5 * self.func(offset[:, rel, :, :]) + 0.5 * self.func(head_offset)
            else:
                for rel in range(0, rel_len):
                    query_min = query_min - 0.5 * self.func(offset[:, rel, :, :])
                    query_max = query_max + 0.5 * self.func(offset[:, rel, :, :])

            # * chain
            if "inter" not in qtype and "union" not in qtype:

                result_query_offset = query_max - query_min
                result_query_center = query_center

            # * inter
            else:
                rel_len = int(qtype.split("-")[0])
                assert rel_len > 1

                # * queries_center are divied into rel_len parts: (query_1:[batch_size,dim],...,query_rel_len:[batch_size, dim])
                query_center = query_center.squeeze(1)
                queries_center = torch.chunk(query_center, rel_len, dim=0)

                offsets = query_max - query_min
                offsets = offsets.squeeze(1)
                # * get offset for each instance in batch
                offsets = torch.chunk(offsets, rel_len, dim=0)
                if "inter" in qtype:
                    result_query_center = self.center_sets(queries_center, offsets)
                    result_query_offset = self.offset_sets(queries_center, offsets)

                else:
                    assert False, "qtype not exists: %s" % qtype
        return result_query_center, result_query_offset

    def forward(self, sample, rel_len, qtype, mode="single"):
        if qtype == "chain-inter":
            assert mode == "tail-batch"
            raise "not chain-inter task right now"

        elif qtype == "nocf-double-inters-chain":

            assert mode == "tail-batch"
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            col_size = 2
            single_ft_num = 80
            assert len(head_part[0][:-1]) == col_size * single_ft_num * 2 + 3
            col_head_list = []
            col_offset_list = []
            for i in range(col_size):
                head_list = []
                relation_list = []
                offset_list = []
                for j in range(single_ft_num):
                    start_idx = i * single_ft_num * 2
                    head_temp = torch.index_select(
                        self.entity_embedding, dim=0, index=head_part[:, j * 2 + start_idx]
                    ).unsqueeze(1)

                    relation_temp = (
                        torch.index_select(self.relation_embedding, dim=0, index=head_part[:, j * 2 + 1 + start_idx])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    if self.geo == "box":
                        offset_temp = (
                            torch.index_select(self.offset_embedding, dim=0, index=head_part[:, j * 2 + 1 + start_idx])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset_list.append(offset_temp)
                    relation_list.append(relation_temp)
                    head_list.append(head_temp)

                heads = torch.cat(head_list, dim=0)
                if self.geo == "box":
                    offsets = torch.cat(offset_list, dim=0)

                relations = torch.cat(relation_list, dim=0)
                if single_ft_num == 80:
                    if self.geo == "box":
                        col, col_off = self.inference(heads, relations, None, offsets, None, 1, "80-inter")
                    else:
                        col = self.vec_inference(heads, relations, 1, "80-inter")

                elif single_ft_num == 45:
                    if self.geo == "box":
                        col, col_off = self.inference(heads, relations, None, offsets, None, 1, "45-inter")
                    else:
                        col = self.vec_inference(heads, relations, 1, "45-inter")
                else:
                    raise "rel_len in inference is wrong"
                col_head_list.append(col)
                if self.geo == "box":
                    col_offset_list.append(col_off)

            col_heads = torch.cat(col_head_list, dim=0).unsqueeze(1)
            head_list = []
            relation_list = []
            offset_list = []

            # NOTE: cross feature first, and then  column embedding. relation, offset is similar
            all_heads = col_heads

            # * concate column & cross feature & vis-choice relation, offset
            start_idx = single_ft_num * 4
            for i in range(3):
                relation_temp = (
                    torch.index_select(self.relation_embedding, dim=0, index=head_part[:, start_idx + i])
                    .unsqueeze(1)
                    .unsqueeze(1)
                )
                if self.geo == "box":
                    offset_temp = (
                        torch.index_select(self.offset_embedding, dim=0, index=head_part[:, start_idx + i])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    offset_list.append(offset_temp)

                    # * add aggregated column offset to in_dataset offset
                    if i < 2:
                        offset_temp += self.func(col_offset_list[i])
                relation_list.append(relation_temp)

            all_relations = torch.cat(relation_list, dim=0)
            if self.geo == "box":
                all_offsets = torch.cat(offset_list, dim=0)
            rel_len = all_relations.size(0) - 1
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )
            if self.geo == "box":
                score, score_cen, offset_norm, score_cen_plus, _ = self.inference(
                    all_heads, all_relations, None, all_offsets, None, rel_len, "inter-chain", tail
                )
                # return score, score_cen, offset_norm, score_cen_plus, None, None
            else:
                score, score_cen, offset_norm, score_cen_plus, _ = self.vec_inference(
                    all_heads, all_relations, rel_len, "inter-chain", tail
                )

            return score, score_cen, offset_norm, score_cen_plus, None, None

        # # * Evaluation part
        # # *  inter(single column)->chain (single cross column)->inter(dataset)->chain(chart)
        elif qtype == "double-inters-chain":

            assert mode == "tail-batch"
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            col_size = 2
            single_ft_num = 80
            # single_ft_num = 45
            cross_ft_num = 40
            assert len(head_part[0][:-1]) == col_size * single_ft_num * 2 + cross_ft_num * 2 + 3
            col_head_list = []
            col_offset_list = []
            for i in range(col_size):
                head_list = []
                relation_list = []
                offset_list = []
                for j in range(single_ft_num):
                    start_idx = i * single_ft_num * 2
                    head_temp = torch.index_select(
                        self.entity_embedding, dim=0, index=head_part[:, j * 2 + start_idx]
                    ).unsqueeze(1)

                    relation_temp = (
                        torch.index_select(self.relation_embedding, dim=0, index=head_part[:, j * 2 + 1 + start_idx])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    if self.geo == "box":
                        offset_temp = (
                            torch.index_select(self.offset_embedding, dim=0, index=head_part[:, j * 2 + 1 + start_idx])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset_list.append(offset_temp)
                    relation_list.append(relation_temp)
                    head_list.append(head_temp)

                heads = torch.cat(head_list, dim=0)
                if self.geo == "box":
                    offsets = torch.cat(offset_list, dim=0)

                relations = torch.cat(relation_list, dim=0)
                if single_ft_num == 80:
                    if self.geo == "box":
                        col, col_off = self.inference(heads, relations, None, offsets, None, 1, "80-inter")
                    else:
                        col = self.vec_inference(heads, relations, 1, "80-inter")

                elif single_ft_num == 45:
                    if self.geo == "box":
                        col, col_off = self.inference(heads, relations, None, offsets, None, 1, "45-inter")
                    else:
                        col = self.vec_inference(heads, relations, 1, "45-inter")
                else:
                    raise "rel_len in inference is wrong"
                col_head_list.append(col)
                if self.geo == "box":
                    col_offset_list.append(col_off)

            col_heads = torch.cat(col_head_list, dim=0).unsqueeze(1)
            head_list = []
            relation_list = []
            offset_list = []

            start_idx = single_ft_num * 4

            for cf_idx in range(cross_ft_num):
                head_temp = torch.index_select(
                    self.entity_embedding, dim=0, index=head_part[:, cf_idx * 2 + start_idx]
                ).unsqueeze(1)
                relation_temp = (
                    torch.index_select(self.relation_embedding, dim=0, index=head_part[:, cf_idx * 2 + 1 + start_idx])
                    .unsqueeze(1)
                    .unsqueeze(1)
                )
                if self.geo == "box":
                    offset_temp = (
                        torch.index_select(self.offset_embedding, dim=0, index=head_part[:, cf_idx * 2 + 1 + start_idx])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    offset_list.append(offset_temp)

                head_list.append(head_temp)
                relation_list.append(relation_temp)
            cf_heads = torch.cat(head_list, dim=0)
            all_heads = torch.cat((cf_heads, col_heads), dim=0)

            # * concate column & cross feature & vis-choice relation, offset
            start_idx = single_ft_num * 4 + cross_ft_num * 2
            for i in range(3):
                relation_temp = (
                    torch.index_select(self.relation_embedding, dim=0, index=head_part[:, start_idx + i])
                    .unsqueeze(1)
                    .unsqueeze(1)
                )
                if self.geo == "box":
                    offset_temp = (
                        torch.index_select(self.offset_embedding, dim=0, index=head_part[:, start_idx + i])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    offset_list.append(offset_temp)

                    # * add aggregated column offset to in_dataset offset
                    if i < 2:
                        offset_temp += self.func(col_offset_list[i])
                relation_list.append(relation_temp)

            all_relations = torch.cat(relation_list, dim=0)
            if self.geo == "box":
                all_offsets = torch.cat(offset_list, dim=0)
            rel_len = all_relations.size(0) - 1
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )
            if self.geo == "box":
                score, score_cen, offset_norm, score_cen_plus, _ = self.inference(
                    all_heads, all_relations, None, all_offsets, None, rel_len, "inter-chain", tail
                )
            else:
                score, score_cen, offset_norm, score_cen_plus, _ = self.vec_inference(
                    all_heads, all_relations, rel_len, "inter-chain", tail
                )

            return score, score_cen, offset_norm, score_cen_plus, None, None

        elif qtype == "inter-chain":

            assert mode == "tail-batch"
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head_list = []
            for i in range(rel_len):

                head_temp = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, i * 2]).unsqueeze(1)
                head_list.append(head_temp)

            head = torch.cat(head_list, dim=0)

            if self.euo and self.geo == "box":
                # raise "ENTITY USING OFFSET"
                head_offset_list = []
                for i in range(rel_len):
                    head_offset_temp = torch.index_select(
                        self.entity_offset_embedding, dim=0, index=head_part[:, i * 2]
                    ).unsqueeze(1)
                    head_offset_list.append(head_offset_temp)
                head_offset = torch.cat(head_offset_list, dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )

            relation_list = []
            for i in range(rel_len):

                relation_temp = (
                    torch.index_select(self.relation_embedding, dim=0, index=head_part[:, i * 2 + 1])
                    .unsqueeze(1)
                    .unsqueeze(1)
                )
                relation_list.append(relation_temp)
            choice_rel = (
                torch.index_select(self.relation_embedding, dim=0, index=head_part[:, rel_len * 2])
                .unsqueeze(1)
                .unsqueeze(1)
            )
            relation_list.append(choice_rel)
            relation = torch.cat(relation_list, dim=0)

            if self.geo == "box":
                offset_list = []
                for i in range(rel_len):
                    offset_temp = (
                        torch.index_select(self.offset_embedding, dim=0, index=head_part[:, i * 2 + 1])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    offset_list.append(offset_temp)
                offset_list.append(choice_rel)
                offset = torch.cat(offset_list, dim=0)

        elif qtype in ["2-inter", "40-inter", "42-inter", "45-inter", "46-inter", "80-inter"]:
            if mode == "single":
                batch_size, negative_sample_size = sample.size(0), 1

                head_list = []
                for i in range(rel_len):

                    head_temp = torch.index_select(self.entity_embedding, dim=0, index=sample[:, i * 2]).unsqueeze(1)
                    head_list.append(head_temp)
                head = torch.cat(head_list, dim=0)
                if self.euo and self.geo == "box":
                    head_offset_list = []
                    for i in range(rel_len):
                        head_offset_temp = torch.index_select(
                            self.entity_offset_embedding, dim=0, index=sample[:, i * 2]
                        ).unsqueeze(1)
                        head_offset_list.append(head_offset_temp)
                    head_offset = torch.cat(head_offset_list, dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)
                tail = torch.cat([tail for _ in range(rel_len)], dim=0)

                relation_list = []
                for i in range(rel_len):
                    relation_temp = (
                        torch.index_select(self.relation_embedding, dim=0, index=sample[:, i * 2 + 1])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    relation_list.append(relation_temp)
                relation = torch.cat(relation_list, dim=0)

                if self.geo == "box":
                    offset_list = []
                    for i in range(rel_len):
                        offset_temp = (
                            torch.index_select(self.offset_embedding, dim=0, index=sample[:, i * 2 + 1])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset_list.append(offset_temp)
                    offset = torch.cat(offset_list, dim=0)

            elif mode == "tail-batch":
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head_list = []

                for i in range(rel_len):

                    head_temp = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, i * 2]).unsqueeze(1)
                    head_list.append(head_temp)

                head = torch.cat(head_list, dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                    batch_size, negative_sample_size, -1
                )
                if tail.size(1) != self.entity_embedding.size(0):
                    tail = torch.cat([tail for _ in range(rel_len)], dim=0)

                relation_list = []
                for i in range(rel_len):

                    relation_temp = (
                        torch.index_select(self.relation_embedding, dim=0, index=head_part[:, i * 2 + 1])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    relation_list.append(relation_temp)

                relation = torch.cat(relation_list, dim=0)

                if self.geo == "box":
                    offset_list = []
                    for i in range(rel_len):
                        offset_temp = (
                            torch.index_select(self.offset_embedding, dim=0, index=head_part[:, i * 2 + 1])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset_list.append(offset_temp)

                    offset = torch.cat(offset_list, dim=0)

        elif qtype == "1-chain" or qtype == "2-chain" or qtype == "3-chain":

            if mode == "single":
                batch_size, negative_sample_size = sample.size(0), 1

                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                relation = (
                    torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                )
                if self.geo == "box":
                    offset = (
                        torch.index_select(self.offset_embedding, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1)
                    )
                    if self.euo:
                        head_offset = torch.index_select(
                            self.entity_offset_embedding, dim=0, index=sample[:, 0]
                        ).unsqueeze(1)
                if rel_len == 2 or rel_len == 3:
                    relation2 = (
                        torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                    )
                    relation = torch.cat([relation, relation2], 1)
                    if self.geo == "box":
                        offset2 = (
                            torch.index_select(self.offset_embedding, dim=0, index=sample[:, 2])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    relation3 = (
                        torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                    )
                    relation = torch.cat([relation, relation3], 1)
                    if self.geo == "box":
                        offset3 = (
                            torch.index_select(self.offset_embedding, dim=0, index=sample[:, 3])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset = torch.cat([offset, offset3], 1)

                assert relation.size(1) == rel_len
                if self.geo == "box":
                    assert offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, -1]).unsqueeze(1)

            elif mode == "tail-batch":
                head_part, tail_part = sample

                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)

                relation = (
                    torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
                )
                if self.geo == "box":
                    offset = (
                        torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    if self.euo:
                        head_offset = torch.index_select(
                            self.entity_offset_embedding, dim=0, index=head_part[:, 0]
                        ).unsqueeze(1)
                if rel_len == 2 or rel_len == 3:
                    relation2 = (
                        torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    relation = torch.cat([relation, relation2], 1)
                    if self.geo == "box":
                        offset2 = (
                            torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    relation3 = (
                        torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3])
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )
                    relation = torch.cat([relation, relation3], 1)
                    if self.geo == "box":
                        offset3 = (
                            torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3])
                            .unsqueeze(1)
                            .unsqueeze(1)
                        )
                        offset = torch.cat([offset, offset3], 1)

                assert relation.size(1) == rel_len
                if self.geo == "box":
                    assert offset.size(1) == rel_len

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                    batch_size, negative_sample_size, -1
                )

        else:
            raise ValueError("mode %s not supported" % mode)

        model_func = {
            "BoxTransE": self.BoxTransE,
            "TransE": self.TransE,
        }
        if self.geo == "vec":
            offset = None
            head_offset = None
        if self.geo == "box":
            if not self.euo:
                head_offset = None

        if self.model_name in model_func:
            # * Intersection query
            if qtype in ["2-inter", "40-inter", "42-inter", "45-inter", "46-inter", "80-inter"]:
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](
                    head, relation, tail, mode, offset, head_offset, 1, qtype
                )
            # * 1-chain & inter-chain
            elif "chain" in qtype:
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](
                    head, relation, tail, mode, offset, head_offset, rel_len, qtype
                )
            else:
                raise "Wrong qtype"
        else:
            raise ValueError("model %s not supported" % self.model_name)
        return score, score_cen, offset_norm, score_cen_plus, None, None

    def BoxTransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):

        """
        Calculate the query loss

        Parameters
        ----------
        head : [batch_size * head_num, 1, dim]

        relation : [batch_size * head_num, chain_rel, 1, dim]
            NOTE: Inter-Chain: head_num = inter_rel + chain_rel (46i->46+1=47)

        tail : [batch_size * head_num, entity_num, dim]

        mode : str
            Tail or single

        offset : [batch_size * head_num, chain_rel, 1, dim]

        head_offset : [batch_size * head_num, 1, dim]

        rel_len : int
            relation number
            NOTE: inter-related query excludes the chain relation for

        qtype : str
            query type

        Returns
        -------
        score: float (torch)
        score_center: float (torch)
        offset score: float (torch)
        score_center_plus: float (torch)
        """
        if qtype == "chain-inter":
            raise "not chain-inter yet"

        elif qtype == "inter-chain":

            # * rel_len + 1 for an extra chain projection
            relations = torch.chunk(relation, rel_len + 1, dim=0)
            offsets = torch.chunk(offset, rel_len + 1, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, rel_len, dim=0)

            query_center_list = []
            heads = torch.chunk(head, rel_len, dim=0)
            for i in range(rel_len):
                query_center = (heads[i] + relations[i][:, 0, :, :]).squeeze(1)
                query_center_list.append(query_center)

            query_min_list = []
            query_max_list = []
            if self.euo:

                for i in range(rel_len):
                    query_min = (
                        query_center_list[i]
                        - 0.5 * self.func(head_offsets[i])
                        - 0.5 * self.func(offsets[i][:, 0, :, :])
                    )
                    query_max = (
                        query_center_list[i]
                        + 0.5 * self.func(head_offsets[i])
                        + 0.5 * self.func(offsets[i][:, 0, :, :])
                    )
                    query_min_list.append(query_min)
                    query_max_list.append(query_max)

            else:
                for i in range(rel_len):

                    query_min = query_center_list[i] - 0.5 * self.func(offsets[i][:, 0, :, :])
                    query_max = query_center_list[i] + 0.5 * self.func(offsets[i][:, 0, :, :])

                    query_min_list.append(query_min)
                    query_max_list.append(query_max)

            assert len(query_min_list) == rel_len & len(query_max_list) == rel_len

            offset_list = []
            for i in range(rel_len):
                offset_temp = (query_max_list[i] - query_min_list[i]).squeeze(1)
                offset_list.append(offset_temp)

            conj_query_center = self.center_sets(query_center_list, offset_list).unsqueeze(1)

            new_query_center = conj_query_center + relations[rel_len][:, 0, :, :]
            new_offset = self.offset_sets(query_center_list, offset_list).unsqueeze(1)

            new_query_min = (
                new_query_center - 0.5 * self.func(new_offset) - 0.5 * self.func(offsets[rel_len][:, 0, :, :])
            )
            new_query_max = (
                new_query_center + 0.5 * self.func(new_offset) + 0.5 * self.func(offsets[rel_len][:, 0, :, :])
            )

            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center - tail

            x_in_box = torch.min(new_query_max, torch.max(new_query_min, tail))
            padding = torch.min(x_in_box - new_query_min, new_query_max - x_in_box)

        elif qtype == "union-chain":
            raise "not union-chain yet"

        else:
            # * n-chain or n-inter
            query_center = head
            for rel in range(rel_len):

                query_center = query_center + relation[:, rel, :, :]
            if self.euo:
                query_min = query_center - 0.5 * self.func(head_offset)
                query_max = query_center + 0.5 * self.func(head_offset)
            else:
                query_min = query_center
                query_max = query_center
            for rel in range(0, rel_len):
                new_query_min = query_min - 0.5 * self.func(offset[:, rel, :, :])
                new_query_max = query_max + 0.5 * self.func(offset[:, rel, :, :])

            # * chain
            if "inter" not in qtype:
                score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
                score_center = query_center - tail

                x_in_box = torch.min(new_query_max, torch.max(new_query_min, tail))
                padding = torch.min(x_in_box - new_query_min, new_query_max - x_in_box)
            # * inter
            else:
                rel_len = int(qtype.split("-")[0])
                assert rel_len > 1
                queries_min = torch.chunk(query_min, rel_len, dim=0)
                queries_max = torch.chunk(query_max, rel_len, dim=0)

                query_center = query_center.squeeze(1)
                queries_center = torch.chunk(query_center, rel_len, dim=0)
                if tail.size(1) == self.entity_embedding.size(0):
                    tails = tail.unsqueeze(0)
                else:  # * training intersection query
                    tails = torch.chunk(tail, rel_len, dim=0)

                offsets = query_max - query_min
                offsets = offsets.squeeze(1)
                # * get offset for each instance in batch
                offsets = torch.chunk(offsets, rel_len, dim=0)
                if "inter" in qtype:
                    # * new_query_center: [batch_size, dim]
                    # * new_offset: [batch_size, dim]

                    new_query_center = self.center_sets(queries_center, offsets)
                    new_offset = self.offset_sets(queries_center, offsets)

                    new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
                    new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)
                    score_offset = F.relu(new_query_min - tails[0]) + F.relu(tails[0] - new_query_max)
                    score_center = new_query_center.unsqueeze(1) - tails[0]

                    x_in_box = torch.min(new_query_max, torch.max(new_query_min, tails[0]))
                    padding = torch.min(x_in_box - new_query_min, new_query_max - x_in_box)
                else:
                    assert False, "qtype not exists: %s" % qtype
        # * distance outside score
        score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  # *[batch_size]


        result_offset = new_query_max - new_query_min
        # REVIEW: Encourage point closer to point
        score_center_plus = (
            self.gamma.item()
            - torch.norm(score_offset, p=1, dim=-1)
            - torch.norm(score_center, p=1, dim=-1)
            + self.cen * torch.norm(padding, p=1, dim=-1)
        )

        score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)
        assert score.size() == score_center_plus.size()
        return (
            score,
            score_center,
            torch.mean(torch.norm(result_offset, p=2, dim=2).squeeze(1)),
            score_center_plus,
            None,
        )

    def TransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):

        if qtype == "inter-chain":


            relations = torch.chunk(relation, rel_len + 1, dim=0)
            heads = torch.chunk(head, rel_len, dim=0)
            score_list = []
            for i in range(rel_len):
                score = (heads[i] + relations[i][:, 0, :, :]).squeeze(1)
                score_list.append(score)
            conj_score = self.deepsets(score_list, None).unsqueeze(1)
            score = conj_score + relations[rel_len][:, 0, :, :] - tail

        else:
            score = head
            for rel in range(rel_len):
                score = score + relation[:, rel, :, :]

            if "inter" not in qtype and "union" not in qtype:
                score = score - tail
            else:
                rel_len = int(qtype.split("-")[0])
                assert rel_len > 1
                score = score.squeeze(1)
                scores = torch.chunk(score, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                if "inter" in qtype:
                    conj_score = self.deepsets(scores, None)
                    conj_score = conj_score.unsqueeze(1)
                    score = conj_score - tails[0]
                elif "union" in qtype:
                    conj_score = torch.stack(scores, dim=0)
                    score = conj_score - tails[0]
                else:
                    assert False, "qtype not exist: %s" % qtype

        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)

        return score, None, None, 0.0, []

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, scaler, step):
        model.train()
        optimizer.zero_grad()

        # * negative_sample:[batch_size]  false entities
        # * mode: tail-batch or single
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        rel_len = int(train_iterator.qtype.split("-")[0])
        qtype = train_iterator.qtype
        with autocast(enabled=args.use_amp):
            negative_score, negative_score_cen, negative_offset, negative_score_cen_plus, _, _ = model(
                (positive_sample, negative_sample),
                rel_len,
                qtype,
                mode=mode,
            )

            # * model_geo: model has vector or box
            model_geo = model.module.geo if hasattr(model, "module") else model.geo
            if model_geo == "box":
                negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)

            positive_score, positive_score_cen, positive_offset, positive_score_cen_plus, _, _ = model(
                positive_sample,
                rel_len,
                qtype,
            )

            if model_geo == "box":
                positive_score = F.logsigmoid(positive_score_cen_plus).squeeze(dim=1)
            else:
                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            if args.uni_weight:
                positive_sample_loss = -positive_score.mean()
                negative_sample_loss = -negative_score.mean()
            else:

                positive_sample_loss = -(subsampling_weight * positive_score).sum()
                negative_sample_loss = -(subsampling_weight * negative_score).sum()
                positive_sample_loss /= subsampling_weight.sum()
                negative_sample_loss /= subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2

            if args.regularization != 0.0:
                regularization = args.regularization * (
                    model.module.entity_embedding.norm(p=3) ** 3
                    + model.module.relation_embedding.norm(p=3).norm(p=3) ** 3
                )
                loss = loss + regularization
                regularization_log = {"regularization": regularization.item()}
            else:
                regularization_log = {}
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        log = {
            **regularization_log,
            "positive_sample_loss": positive_sample_loss.item(),
            "negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, test_triples, test_ans, args):

        qtype = test_triples[0][-1]
        if qtype == "nocf-double-inters-chain":
            rel_len = len(test_triples[0][:-5])
        elif qtype == "double-inters-chain":
            rel_len = len(test_triples[0][:-5])
        elif qtype == "inter-chain":
            rel_len = len(test_triples[0][:-3])
        else:
            rel_len = int(test_triples[0][-1].split("-")[0])

        model.eval()
        if qtype == "nocf-double-inters-chain":
            test_dataloader_tail = DataLoader(
                TestDoubleInterChainDataset(test_triples, test_ans, args.nentity, args.nrelation, "tail-batch"),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn,
                shuffle=True,
            )
        elif qtype == "double-inters-chain":
            test_dataloader_tail = DataLoader(
                TestDoubleInterChainDataset(test_triples, test_ans, args.nentity, args.nrelation, "tail-batch"),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn,
                shuffle=True,
            )
        elif qtype == "inter-chain":
            test_dataloader_tail = DataLoader(
                TestInterChainDataset(test_triples, test_ans, args.nentity, args.nrelation, "tail-batch"),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn,
                shuffle=True,
            )
        elif "inter" in qtype:
            test_dataloader_tail = DataLoader(
                TestInterDataset(
                    test_triples,
                    test_ans,
                    args.nentity,
                    args.nrelation,
                    "tail-batch",
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn,
                shuffle=True,
            )
        elif "chain" in qtype:
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    test_ans,
                    args.nentity,
                    args.nrelation,
                    "tail-batch",
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TestDataset.collate_fn,
                shuffle=True,
            )
        else:
            raise "test qtype is not found"
        traces = ["bar", "box", "line", "scatter"]
        trace_id = [args.entity2id[i] for i in traces]
        mask = torch.zeros((1, args.nentity))
        mask[:, trace_id] = 1
        mask = mask.cuda()
        METRICS = ["multiple", "one"]
        assert "one" in METRICS or "multiple" in METRICS

        # * model_geo: model has vector or box
        model_geo = model.module.geo if hasattr(model, "module") else model.geo

        test_dataset_list = [test_dataloader_tail]
        # test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        step = 0

        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        logs = []
        one_ans_logs = []
        multi_ans_logs = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode, query in tqdm(test_dataset):

                    multi_ans_log = dict()
                    one_ans_log = dict()
                    if args.local_rank in [0, -1]:
                        if torch.cuda.is_available():

                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                        else:
                            logging.warning("using cpu right now !!!!!!!!!!!!")

                    batch_size = positive_sample.size(0)
                    assert batch_size == 1, batch_size

                    if model_geo == "box":
                        score_box, score_cen, offset_norm, score_cen_plus, _, _ = model(
                            (positive_sample, negative_sample), rel_len, qtype, mode=mode
                        )
                        # * Evaluate multiple recommendation
                        if len(test_ans[query]) == 4:
                            continue
                        if "multiple" in METRICS and len(test_ans[query]) > 1:
                            ans_list = list(test_ans[query])
                            model_gamma = model.module.gamma if hasattr(model, "module") else model.gamma
                            distance = model_gamma.item() - score_box
                            multi_preds = set(
                                (distance == 0).nonzero(as_tuple=False).flatten().cpu().numpy()
                            ).intersection(set(trace_id))
                            if len(multi_preds) != 0:
                                multi_ans_log.update(
                                    {
                                        "recall": cal_recall(multi_preds, test_ans[query]),
                                        "precision": cal_prec(multi_preds, test_ans[query]),
                                        "F1": cal_F1(
                                            cal_recall(multi_preds, test_ans[query]),
                                            cal_prec(multi_preds, test_ans[query]),
                                        ),
                                    }
                                )
                            else:
                                score2 = score_cen_plus
                                score2 -= torch.min(score_cen_plus) - 1
                                filter_chart_score = mask * score2
                                sorted_chart = torch.argsort(filter_chart_score, dim=1, descending=True)
                                pred = sorted_chart[:, 0].cpu().tolist()
                                multi_ans_log.update(
                                    {
                                        "recall": cal_recall(pred, ans_list),
                                        "precision": cal_prec(pred, ans_list),
                                        "F1": cal_F1(cal_recall(pred, ans_list), cal_prec(pred, ans_list)),
                                    }
                                )
                            multi_ans_logs.append(multi_ans_log)
                        if "one" in METRICS:
                            ans_list = [positive_sample[:, -1].cpu().item()]
                            score_one_ans = score_cen
                            score_one_ans -= torch.min(score_one_ans) - 1
                            filter_chart_score = mask * score_one_ans
                            ans_tensor = (
                                torch.LongTensor(ans_list)
                                if not torch.cuda.is_available()
                                else torch.LongTensor(ans_list).cuda()
                            )

                            sorted_chart = torch.argsort(filter_chart_score, dim=1, descending=True)
                            sorted_chart = torch.transpose(torch.transpose(sorted_chart, 0, 1) - ans_tensor, 0, 1)
                            ranking = (sorted_chart == 0).nonzero(as_tuple=False)
                            ranking = ranking[:, 1]
                            ranking = ranking + 1

                            hits1m_newd = torch.mean((ranking <= 1).to(torch.float)).item()
                            hits2m_newd = torch.mean((ranking <= 2).to(torch.float)).item()
                            mrm_newd = torch.mean(ranking.to(torch.float)).item()
                            mrrm_newd = torch.mean(1.0 / ranking.to(torch.float)).item()

                            true_relevance = np.where(np.array(sorted(trace_id)) == ans_list[0], 1, 0)
                            pred_score = np.array([filter_chart_score[mask.bool()].tolist()])
                            nDCG = ndcg_score(true_relevance.reshape(1, -1), pred_score)
                            one_ans_log.update(
                                {
                                    "MRRm_new": mrrm_newd,
                                    "MRm_new": mrm_newd,
                                    "HITS@1m_new": hits1m_newd,
                                    "HITS@2m_new": hits2m_newd,
                                    "nDCG": nDCG,
                                }
                            )

                        one_ans_logs.append(one_ans_log)
                    else:
                        raise "need box"
                    step += 1
                    if step > 5000:
                        logging.info("End the validation/testing")
                        break
                    metrics = {}
                    if len(multi_ans_logs) > 0:
                        for metric in multi_ans_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in multi_ans_logs]) / len(multi_ans_logs)
                    for metric in one_ans_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in one_ans_logs]) / len(one_ans_logs)
        return metrics
