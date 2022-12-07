import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import random

class BehaviorAggregator(nn.Module):
    def __init__(self, opt):
        super(BehaviorAggregator, self).__init__()
        self.opt = opt
        self.aggregator = opt["aggregator"]
        self.lambda_a = opt["lambda_a"]
        embedding_dim = opt["latent_dim"]
        dropout_rate = opt["dropout"]

        self.W_agg = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["user_attention"]:
            self.W_att = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, id_emb, sequence_emb, score):
        out = id_emb
        if self.aggregator == "mean":
            out = self.mean_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self.user_attention_pooling(id_emb, sequence_emb)
        elif self.aggregator == "item_similarity":
            out = self.item_similarity_pooling(sequence_emb, score)
        else:
            print("a wrong aggregater!!")
            exit(0)
        return self.lambda_a * id_emb + (1 - self.lambda_a) * out

    def user_attention_pooling(self, id_emb, sequence_emb):
        key = self.W_att(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, id_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_agg(output)

    def mean_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
        return self.W_agg(mean)

    def item_similarity_pooling(self, sequence_emb, score):
        if len(score.size()) != 2:
            score = score.view(score.size(0), -1)
        score = F.softmax(score, dim = -1)
        score = score.unsqueeze(-1)
        ans = (score * sequence_emb).sum(dim=1)
        return self.W_agg(ans)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-12)


class UniCDR(nn.Module):
    def __init__(self, opt):
        super(UniCDR, self).__init__()
        self.opt=opt

        self.specific_user_emb_list = []
        self.specific_item_emb_list = []
        self.agg_list = []

        self.user_index_list = []
        self.item_index_list = []
        self.dis_list = []

        if "item" in opt["task"]:
            self.share_user_emb_list = []

        for i in range(self.opt["num_domains"]):
            self.agg_list.append(BehaviorAggregator(opt))
            self.dis_list.append(nn.Bilinear(opt["latent_dim"], opt["latent_dim"], 1))

            self.specific_user_emb_list.append(nn.Embedding(opt["user_max"][i], opt["latent_dim"]))
            self.specific_item_emb_list.append(nn.Embedding(opt["item_max"][i] + 1, opt["latent_dim"], padding_idx=0))  # padding

            if "item" in opt["task"]:
                self.share_user_emb_list.append(nn.Embedding(opt["user_max"][i], opt["latent_dim"]))

            self.user_index_list.append(torch.arange(0, opt["user_max"][i], 1))
            self.item_index_list.append(torch.arange(0, opt["item_max"][i] + 1, 1))


        # the last shared module
        self.agg_list.append(BehaviorAggregator(opt))

        self.specific_user_emb_list = nn.ModuleList(self.specific_user_emb_list)
        self.specific_item_emb_list = nn.ModuleList(self.specific_item_emb_list)
        self.agg_list = nn.ModuleList(self.agg_list)
        self.dis_list = nn.ModuleList(self.dis_list)
        if "item" in opt["task"]:
            self.share_user_emb_list = nn.ModuleList(self.share_user_emb_list)


        self.criterion = nn.BCEWithLogitsLoss()

        if opt["task"] == "dual-user-intra":
            self.share_user_embedding = nn.Embedding(max(opt["user_max"]), opt["latent_dim"])
        if opt["task"] == "multi-user-intra":
            self.share_user_embedding = nn.Embedding(max(opt["user_max"]), opt["latent_dim"])
        elif opt["task"] == "multi-item-intra":
            self.share_item_embedding = nn.Embedding(max(opt["item_max"]) + 1, opt["latent_dim"])
        elif opt["task"] == "dual-user-inter":
            self.share_user_embedding = nn.Embedding(opt["shared_user"], opt["latent_dim"])
            self.share_user_embedding_A = nn.Embedding(opt["user_max"][0], opt["latent_dim"])
            self.share_user_embedding_B = nn.Embedding(opt["user_max"][1], opt["latent_dim"])
            self.share_user_index = torch.arange(0, opt["shared_user"], 1)

        self.dropout = opt["dropout"]

        if "multi" in opt["task"]:
            self.warmup = 1
        else:
            self.warmup = 0

        if self.opt["cuda"]:
            self.user_index_list = [l.cuda() for l in self.user_index_list]
            self.item_index_list = [l.cuda() for l in self.item_index_list]
            if opt["task"] == "dual-user-inter":
                self.share_user_index = self.share_user_index.cuda()

            self.criterion.cuda()

        self.con_item_emb_list = [0] * (self.opt["num_domains"] + 1)

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def predict_dot(self, user_embedding, item_embedding):
        if len(item_embedding.size()) == 3:
            user_embedding = user_embedding.unsqueeze(1)
            user_embedding = user_embedding.repeat(1, item_embedding.size(1), 1)
        output = (user_embedding * item_embedding).sum(dim=-1)
        return output

    def item_embedding_select(self):
        for idx in range(self.opt["num_domains"]):
            self.con_item_emb_list[idx] = self.my_index_select_embedding(self.specific_item_emb_list[idx], self.item_index_list[idx])

        if "item" in self.opt["task"]:
            self.con_item_emb_list[-1] = self.my_index_select_embedding(self.share_item_embedding, self.item_index_list[-1])

    def forward_user(self, domain_id, user, context_item, context_score, global_item, global_score):
        user_emb = self.specific_user_emb_list[domain_id](user)
        context_item_emb = self.my_index_select(self.con_item_emb_list[domain_id], context_item)
        specific_user = self.agg_list[domain_id](user_emb, context_item_emb, context_score)
        specific_user = F.dropout(specific_user, self.dropout, training=self.training)

        if self.opt["task"] == "dual-user-inter":
            shared_user_emb = self.share_user_embedding(self.share_user_index)
            local_user_emb_A = self.share_user_embedding_A(self.user_index_list[0])
            local_user_emb_B = self.share_user_embedding_B(self.user_index_list[1])
            if self.training:
                global_user_emb_A = torch.cat((shared_user_emb, local_user_emb_A), dim=0)
                global_user_emb_B = torch.cat((shared_user_emb, local_user_emb_B), dim=0)
            else:
                global_user_emb_A = torch.cat((shared_user_emb, local_user_emb_B), dim=0)
                global_user_emb_B = torch.cat((shared_user_emb, local_user_emb_A), dim=0)
            if domain_id == 0:
                global_user_emb = self.my_index_select(global_user_emb_A, user)
            if domain_id == 1:
                global_user_emb = self.my_index_select(global_user_emb_B, user)
        elif self.opt["task"] == "multi-item-intra":
            global_user_emb = self.share_user_emb_list[domain_id](user)
        else:
            global_user_emb = self.share_user_embedding(user)

        global_item_emb = None
        if self.opt["task"] == "multi-item-intra":
            global_item_emb = self.my_index_select(self.con_item_emb_list[-1], context_item)
        else:
            for i in range(self.opt["num_domains"]):
                res = self.my_index_select(self.con_item_emb_list[i], global_item[:,i,:].contiguous()).contiguous()
                if global_item_emb is None:
                    global_item_emb = res
                else:
                    global_item_emb = torch.cat((global_item_emb, res), dim = -2)

        if self.opt["task"] == "multi-item-intra":
            share_user = self.agg_list[-1](global_user_emb, global_item_emb, context_score)
        else:
            share_user = self.agg_list[-1](global_user_emb, global_item_emb, global_score)
        share_user = F.dropout(share_user, self.dropout, training=self.training)

        if self.training and (self.warmup == 0):
            random_label = (torch.arange(0, share_user.size(0), 1).cuda(share_user.device) + torch.randint(1, share_user.size(0), (1,)).item()) % share_user.size(0)

            pos = self.dis_list[domain_id](specific_user,share_user).view(-1)
            neg = self.dis_list[domain_id](self.my_index_select(specific_user, random_label), share_user).view(-1)

            pos_label, neg_label = torch.ones(pos.size()), torch.zeros(
                neg.size())
            if self.opt["cuda"]:
                pos_label = pos_label.cuda()
                neg_label = neg_label.cuda()

            self.critic_loss = self.criterion(pos, pos_label) + self.criterion(neg, neg_label)
        else:
            self.critic_loss = 0

        if self.warmup: # warm-up for multi-domain scenario
            return specific_user

        # In Eq.13, same weight
        return specific_user + share_user

    def forward_item(self, domain_id, item):
        learn_item = self.my_index_select(self.con_item_emb_list[domain_id], item)
        learn_item = F.dropout(learn_item, self.dropout, training=self.training)
        return learn_item
