import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.UniCDR import UniCDR
from utils import torch_utils
import numpy as np
import pdb
import math

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch=None):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class CrossTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if self.opt["model"] == "UniCDR":
            self.model = UniCDR(opt)
        else :
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'], opt["weight_decay"])
        self.epoch_rec_loss = []

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            item = inputs[1]
            context_item = inputs[2]
            context_score = inputs[3]
            global_item = inputs[4]
            global_score = inputs[5]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            item = inputs[1]
            context_item = inputs[2]
            context_score = inputs[3]
            global_item = inputs[4]
            global_score = inputs[5]
        return user, item, context_item, context_score, global_item, global_score

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            pos_item = inputs[1]
            neg_item = inputs[2]
            context_item = inputs[3]
            context_score = inputs[4]
            global_item = inputs[5]
            global_score = inputs[6]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            pos_item = inputs[1]
            neg_item = inputs[2]
            context_item = inputs[3]
            context_score = inputs[4]
            global_item = inputs[5]
            global_score = inputs[6]
        return user, pos_item, neg_item, context_item, context_score, global_item, global_score

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def reconstruct_graph(self, domain_id, batch):
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)

        user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
        pos_item_feature = self.model.forward_item(domain_id, pos_item)
        neg_item_feature = self.model.forward_item(domain_id, neg_item)

        pos_score = self.model.predict_dot(user_feature, pos_item_feature)
        neg_score = self.model.predict_dot(user_feature, neg_item_feature)

        pos_labels, neg_labels = torch.ones(pos_score.size()), torch.zeros(neg_score.size())

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        loss = self.opt["lambda_loss"] * (self.criterion(pos_score, pos_labels) + self.criterion(neg_score, neg_labels)) + (1 - self.opt["lambda_loss"]) * self.model.critic_loss

        return loss

    def predict(self, domain_id, eval_dataloader):
        MRR = 0.0
        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        valid_entity = 0

        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)

            scores = self.model.predict_dot(user_feature, item_feature)

            scores = scores.data.detach().cpu().numpy()

            for pred in scores:

                rank = (-pred).argsort().argsort()[0].item()

                valid_entity += 1
                MRR += 1 / (rank + 1)
                if rank < 1:
                    NDCG_1 += 1 / np.log2(rank + 2)
                    HT_1 += 1
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HT_10 += 1
                if valid_entity % 100 == 0:
                    print('+', end='')

        print("")
        metrics = {}
        # metrics["MRR"] = MRR / valid_entity
        # metrics["NDCG_5"] = NDCG_5 / valid_entity
        metrics["NDCG_10"] = NDCG_10 / valid_entity
        # metrics["HT_1"] = HT_1 / valid_entity
        # metrics["HT_5"] = HT_5 / valid_entity
        metrics["HT_10"] = HT_10 / valid_entity

        return metrics


    def predict_full_rank(self, domain_id, eval_dataloader, train_map, eval_map):

        def nDCG(ranked_list, ground_truth_length):
            dcg = 0
            idcg = IDCG(ground_truth_length)
            for i in range(len(ranked_list)):
                if ranked_list[i]:
                    rank = i + 1
                    dcg += 1 / math.log(rank + 1, 2)
            return dcg / idcg

        def IDCG(n):
            idcg = 0
            for i in range(n):
                idcg += 1 / math.log(i + 2, 2)
            return idcg

        def precision_and_recall(ranked_list, ground_number):
            hits = sum(ranked_list)
            pre = hits / (1.0 * len(ranked_list))
            rec = hits / (1.0 * ground_number)
            return pre, rec

        ndcg_list = []
        pre_list = []
        rec_list = []

        NDCG_10 = 0.0
        HT_10 = 0

        # pdb.set_trace()
        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)


            scores = self.model.predict_dot(user_feature, item_feature)

            scores = scores.data.detach().cpu().numpy()
            user = user.data.detach().cpu().numpy()
            # pdb.set_trace()
            for idx, pred in enumerate(scores):
                rank = (-pred).argsort()
                score_list = []

                hr=0
                for i in rank:
                    i = i + 1
                    if (i in train_map[user[idx]]) and (i not in eval_map[user[idx]]):
                        continue
                    else:
                        if i in eval_map[user[idx]]:
                            hr = 1
                            # nd += 1 / np.log2(len(score_list) + 2)
                            score_list.append(1)
                        else:
                            score_list.append(0)
                        if len(score_list) == 10:
                            break

                HT_10 += hr

                pre, rec = precision_and_recall(score_list, len(eval_map[user[idx]]))
                pre_list.append(pre)
                rec_list.append(rec)
                ndcg_list.append(nDCG(score_list, len(eval_map[user[idx]])))

                if len(ndcg_list) % 100 == 0:
                    print('+', end='')
        print("")

        metrics = {}
        metrics["HT_10"] = HT_10 / len(ndcg_list)
        metrics["NDCG_10"] = sum(ndcg_list) / len(ndcg_list)

        # metrics["MRR"] = 0
        # precision = sum(pre_list) / len(pre_list)
        # recall = sum(rec_list) / len(rec_list)
        # metrics["F_10"] = 2 * precision * recall / (precision + recall + 0.00000001)

        return metrics
