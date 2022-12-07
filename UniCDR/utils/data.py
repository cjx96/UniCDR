import os
import torch
import random
import resource
# import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import pdb
import pickle
import codecs

random.seed(0)


class MetaDomain_DataLoader(object):
    """Data Loader for domains, samples task and returns the dataloader for that Domain"""

    def __init__(self, task_list, sample_batch_size, task_batch_size=2, shuffle=True, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        def collate_fn_train(data):
            user_list, pos_item_list, neg_item_list, context_item_list, context_socre_list, global_item_list, global_score_list = zip(
                *data)
            user, pos_item, neg_item, context_item, context_socre, global_item, global_score = [], [], [], [], [], [], []
            for id, l in enumerate(user_list):
                user += user_list[id]
                pos_item += pos_item_list[id]
                neg_item += neg_item_list[id]
                context_item += context_item_list[id]
                context_socre += context_socre_list[id]
                global_item += global_item_list[id]
                global_score += global_score_list[id]
            return (torch.LongTensor(user), torch.LongTensor(pos_item), torch.LongTensor(neg_item), torch.LongTensor(context_item), torch.FloatTensor(context_socre), torch.LongTensor(global_item), torch.FloatTensor(global_score))

        self.num_domains = len(task_list)
        self.task_list = task_list
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_batch_size = sample_batch_size
        self.task_list_loaders = {
            idx: DataLoader(task_list[idx], batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=False, collate_fn=collate_fn_train) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx: iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        self.task_batch_size = min(task_batch_size, self.num_domains)


    def get_iterator(self, index):
        return self.task_list_iters[index]

    def sample_task(self):
        sampled_task_idx = random.randint(0, self.num_domains - 1)
        return self.task_list_loaders[sampled_task_idx]

    def __len__(self):
        return self.num_domains

    def __getitem__(self, index):
        return self.task_list_loaders[index]


class MetaDomain_Dataset(object):
    """
    Wrapper around domain data (task)
    ratings: {
      0: market_gen,
      1: market_gen,
      ...
    }
    """

    def __init__(self, task_gen_dict, num_negatives=4, meta_split='train'):
        self.num_domains = len(task_gen_dict)
        if meta_split == 'train':
            self.task_gen_dict = {idx: cur_task.instance_a_train_task(num_negatives) for idx, cur_task in
                                  task_gen_dict.items()}
        else:
            self.task_gen_dict = {idx: cur_task.instance_a_valid_task(idx, split=meta_split) for idx, cur_task in
                                  task_gen_dict.items()}

    def __len__(self):
        return self.num_domains

    def __getitem__(self, index):
        return self.task_gen_dict[index]


class DomainTask(Dataset):
    """
    Individual Market data that is going to be wrapped into a metadataset  i.e. MetaMarketDataset

    Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset
    """
    def __init__(self, opt, user_list, item_list, positive_set=None, positive_list=None, item_set=None, all_domain_list=None, idx = None,  number_neg=None, test_mode = None):

        self.opt = opt
        self.user = user_list
        self.item = item_list
        self.item_set = item_set
        self.positive_set = positive_set
        self.positive_list = positive_list
        self.number_neg = number_neg
        self.cache = []
        self.test_mode = test_mode
        self.all_domain_list = all_domain_list
        self.idx = idx

    def __len__(self):
        return len(self.user)

    def __getitem__(self, index):
        if self.test_mode is 1:
            if self.opt["task"] == "multi-user-intra":
                test_max_len = 100
            else:
                test_max_len = 50
            if self.opt["task"] == "dual-user-inter": # cold-start user just have padding in test domain
                cur_context = [[0, -100]] * 2 # -100 is the weight for softmax
            else:
                cur_context = random.sample(self.positive_list[self.user[index]],
                                            min(test_max_len, len(self.positive_list[self.user[index]])))
                if len(cur_context) < test_max_len:
                    cur_context += [[0, -100]] * (test_max_len - len(cur_context))

            res = []
            for id in range(self.opt["num_domains"]):
                if id == self.idx: # cold-start user just have padding in its domain
                    if self.opt["task"] == "dual-user-inter":
                        res.append([[0, -100]] * test_max_len)
                        continue

                if id != self.idx and self.opt["task"] == "multi-item-intra":
                    res.append([[0, -100]] * test_max_len)
                    continue

                if self.user[index] in self.all_domain_list[id]:
                    ress = random.sample(self.all_domain_list[id][self.user[index]],
                                         min(test_max_len, len(self.all_domain_list[id][self.user[index]])))
                else:
                    ress = [[0, -100]] * test_max_len

                if len(ress) < test_max_len:
                    ress += [[0, -100]] * (test_max_len - len(ress))
                res.append(ress)
            global_context = res

            context_item = [item for item, score in cur_context]
            context_score = [score for item, score in cur_context]

            global_item, global_score = [], []

            for cur in global_context:
                global_item.append([item for item, score in cur])
                global_score.append([score for item, score in cur])

            return ([self.user[index]], [self.item[index]], [context_item], [context_score], [global_item], [global_score])

        if self.opt["static_sample"] and len(self.cache) > index:
            return self.cache[index]

        user = []
        pos_item = []
        neg_item = []

        global_item = []
        # import pdb
        # pdb.set_trace()
        u = self.user[index]
        user.append(u)
        pos_item.append(self.item[index])

        # mask some interactions
        cur_context = random.sample(self.positive_list[u],
                                    min(self.opt["maxlen"], int(len(self.positive_list[u]) * (1 - self.opt["mask_rate"]))))

        try:
            cur_context.remove(self.item[index])
        except:
            pass

        if len(cur_context) < self.opt["maxlen"]:
            cur_context += [[0, -100]] * (self.opt["maxlen"] - len(cur_context))

        res = []
        for id in range(self.opt["num_domains"]):
            # domain mask
            if id != self.idx and self.opt["task"] != "multi-item-intra":
                if self.opt["task"] == "dual-user-inter":
                    if u >= self.opt["shared_user"]: # attention here!!
                        res.append([[0, -100]] * self.opt["maxlen"])
                        continue

                if u in self.all_domain_list[id]:
                    if "inter" in self.opt["task"]:
                        ress = random.sample(self.all_domain_list[id][u], min(self.opt["maxlen"], len(self.all_domain_list[id][u])))
                    else:
                        ress = random.sample(self.all_domain_list[id][u],min(self.opt["maxlen"], int(len(self.all_domain_list[id][u]) * (1 - self.opt["mask_rate"]))))
                else:
                    ress = [[0, -100]] * self.opt["maxlen"]

                if len(ress) < self.opt["maxlen"]:
                    ress += [[0, -100]] * (self.opt["maxlen"] - len(ress))
                res.append(ress)
            else:
                res.append([[0, -100]] * self.opt["maxlen"])
        global_item = res

        neg_res = []
        for i in range(self.number_neg):
            while True:
                r = random.sample(self.item_set, 1)[0]
                if r not in self.positive_set[u]:
                    neg_res.append(r)
                    break
        neg_item.append(neg_res)

        context_item = [item for item, score in cur_context]
        context_score = [score for item, score in cur_context]

        global_item_item, global_item_score = [], []
        for cur_domain in global_item:
            global_item_item.append([item for item, score in cur_domain])
            global_item_score.append([score for item, score in cur_domain])

        self.cache.append((user, pos_item, neg_item, [context_item], [context_score], [global_item_item], [global_item_score]))
        return (user, pos_item, neg_item, [context_item], [context_score], [global_item_item], [global_item_score])


class TaskGenerator(object):
    """Construct dataset"""

    def __init__(self, train_file, opt, all_domain_list, all_domain_set, idx, total_graphs):
        self.opt = opt
        self.all_domain_list = all_domain_list
        self.all_domain_set = all_domain_set
        self.idx = idx
        self.positive_set, self.positive_list, self.train_data, self.user_set, self.item_set = self.read_train_data(train_file, total_graphs.ease[idx])
        self.eval_set = {}

    def read_train_data(self, train_file, ease):
        # pdb.set_trace()
        if self.opt["aggregator"] == "item_similarity":
            ease_dense = ease.to_dense()
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user_set = set()
            item_set = set()
            positive_set = {}
            positive_list = {}
            for line in infile:
                line=line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                train_data.append([user, item])
                if user not in positive_set.keys():
                    positive_set[user] = set()
                    positive_list[user] = []
                if item not in positive_set[user]:
                    positive_set[user].add(item)
                    if self.opt["aggregator"] == "item_similarity":
                        # add the prior knowledge
                        positive_list[user].append([item, ease_dense[user][item]])
                    else:
                        positive_list[user].append([item, 1])
                user_set.add(user)
                item_set.add(item)
        print("the min/max user/item number of ", train_file)
        print("user:", min(user_set), max(user_set))
        print("item:", min(item_set), max(item_set))



        if self.opt["aggregator"] == "item_similarity": # extend items for 1-hop sampled graph_sage
            def takeSecond(elem):
                return -elem[1]

            user_index = ease.coalesce().indices()[0].numpy()
            item_index = ease.coalesce().indices()[1].numpy()

            for id, user in enumerate(user_index):
                item = item_index[id]
                if item not in positive_set[user]:
                    positive_list[user].append([item, ease_dense[user][item]])

            for user in positive_list:
                positive_list[user].sort(key=takeSecond)
                # To accelerate computation, filter some smaller elements
                positive_list[user] = positive_list[user][:int(1.5 * len(positive_set[user]))]
                for item, score in positive_list[user]:
                    if item not in positive_set[user]:
                        positive_set[user].add(item)

        return positive_set, positive_list, train_data, user_set, item_set

    def instance_a_train_task(self, num_negatives):
        """instance train task's torch Dataset"""
        users, items = [], []
        train_data = self.train_data

        for interaction in train_data:
            users.append(interaction[0])
            items.append(interaction[1])

        dataset = DomainTask(self.opt, user_list=users,
                             item_list=items,
                             positive_set=self.positive_set,
                             positive_list=self.positive_list,
                             item_set=self.item_set,
                             number_neg=num_negatives,
                             all_domain_list=self.all_domain_list,
                             idx = self.idx
                             )
        return dataset

    def load_domain_valid_run(self, valid_run_file):
        users, items = [], []
        item_list = list(self.item_set)

        if self.opt["task"] == "multi-user-intra": # full rank prediction
            ret = []
            for i in range(1, max(self.item_set) + 1):
                ret.append(i)

            with open(valid_run_file, 'r') as f:
                for line in f:
                    linetoks = line.split('\t')
                    user_id = int(linetoks[0])
                    item_id = int(linetoks[1]) + 1
                    if user_id not in self.eval_set:
                        users.append(user_id)
                        items.append(ret)
                        self.eval_set[user_id] = set()
                    self.eval_set[user_id].add(item_id)

            print(valid_run_file, "valid user: ", len(users))

            dataset = DomainTask(self.opt, user_list=users,
                                 item_list=items,
                                 positive_set=self.positive_set,
                                 positive_list=self.positive_list,
                                 item_set=self.item_set,
                                 test_mode=1,
                                 all_domain_list=self.all_domain_list,
                                 idx=self.idx
                                 )
        else:
            negative_number = 999
            with open(valid_run_file, 'r') as f:
                for line in f:
                    linetoks = line.split('\t')
                    user_id = int(linetoks[0])
                    item_id = int(linetoks[1]) + 1
                    if item_id in self.item_set:
                        ret = [item_id]
                    else:
                        continue

                    if self.opt["task"] == "multi-item-intra":
                        for i in linetoks[2:]:
                            ret.append(int(i) + 1)
                    else:
                        for i in range(negative_number):
                            while True:
                                rand = random.choice(item_list)
                                if user_id in self.all_domain_set[self.idx].keys():
                                    if rand in self.all_domain_set[self.idx][user_id]:
                                        continue
                                    if rand in ret:
                                        continue
                                ret.append(rand)
                                break
                    users.append(user_id)
                    items.append(ret)

            if self.opt["task"] == "dual-user-inter":
                self.opt["shared_user"]= min(self.opt["shared_user"], min(users))

            dataset = DomainTask(self.opt, user_list=users,
                                 item_list=items,
                                 positive_set=self.positive_set,
                                 positive_list=self.positive_list,
                                 item_set=self.item_set,
                                 test_mode = 1,
                                 all_domain_list=self.all_domain_list,
                                 idx=self.idx
                                 )
        return dataset

    def instance_a_valid_dataloader(self, valid_run_file, sample_batch_size, shuffle=False, num_workers=0):
        """instance domain's validation data torch Dataloader"""

        def collate_fn_valid(data):
            user_list, item_list, context_item_list, context_score_list, global_item_list, global_score_list = zip(*data)

            user, item, context_item, context_score, global_item, global_score = [], [], [], [], [], []
            for id in range(len(user_list)):
                user += user_list[id]
                item += item_list[id]
                context_item += context_item_list[id]
                context_score += context_score_list[id]
                global_item += global_item_list[id]
                global_score += global_score_list[id]

            return (torch.LongTensor(user), torch.LongTensor(item), torch.LongTensor(context_item), torch.FloatTensor(context_score), torch.LongTensor(global_item), torch.FloatTensor(global_score))

        print("the evaluation data: ", valid_run_file)
        dataset = self.load_domain_valid_run(valid_run_file)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=collate_fn_valid)