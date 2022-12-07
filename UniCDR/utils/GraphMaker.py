import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy
import pdb
import os

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker(object):
    def __init__(self, opt, all_domain_list):
        print("begin graphmaker................")
        self.opt = opt
        self.UV = []
        self.VU = []
        self.ease = []

        self.UV_edges = []
        self.VU_edges = []

        for idx in range(len(all_domain_list)):
            UV, VU, ease = self.preprocess(all_domain_list[idx], opt, idx)
            self.UV.append(UV)
            self.VU.append(VU)
            self.ease.append(ease)

        print("graphmaker done.........")

    def preprocess(self,data, opt, idx):
        UV_edges = []
        VU_edges = []

        user_ind = 0
        item_ind = 0
        if "item" in opt["task"]:
            user_ind = sum(opt["user_max"][:idx])
        else:
            item_ind = sum(opt["item_max"][:idx]) - idx

        print("The alignment id", user_ind, item_ind)
        for user in data:
            for item in data[user]:
                UV_edges.append([user, item])
                VU_edges.append([item, user])

                self.UV_edges.append([user+user_ind, item+item_ind]) # remain for global pre-process, not used in this work
                self.VU_edges.append([item+item_ind, user+user_ind])

        UV_adj, VU_adj, EASE_pred = self.matrix_norm(UV_edges, VU_edges, opt["user_max"][idx], opt["item_max"][idx], self.opt["domains"][idx])
        return UV_adj, VU_adj, EASE_pred

    def matrix_norm(self, UV_edges, VU_edges, user_number, item_number, cur_domain):
        if self.opt["task"] == "multi-item-intra":
            cur_src_ease_dir = os.path.join("../datasets/" + str(self.opt["task"]) + "/dataset/", cur_domain + "/ease")
        else:
            cur_src_ease_dir = os.path.join("../datasets/" + str(self.opt["task"]) + "/dataset/", cur_domain + "/ease")

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        print(user_number, item_number)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(user_number, item_number),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(item_number, user_number),
                               dtype=np.float32)

        def cal_num(X):
            sans = sp.csr_matrix(X)
            print(sans.nnz)

        if self.opt["aggregator"] == "item_similarity":
            print("Start the EASE")
            X = copy.deepcopy(UV_adj)

            if os.path.exists(cur_src_ease_dir + ".npy"):
                print("load")
                B = np.load(cur_src_ease_dir + ".npy")
            else:
                print("calcul")
                G = X.T.dot(X).toarray()
                diagIndices = np.diag_indices(G.shape[0])

                G[diagIndices] += self.opt["lambda"]
                P = np.linalg.inv(G)
                B = P / (-np.diag(P))

                B[diagIndices] = 0
                np.save(cur_src_ease_dir, B)

            EASE_pred = X.dot(B)

            # To accelerate computation, filter some smaller elements
            cal_num(EASE_pred)
            EASE_pred[EASE_pred < 0.1] = 0
            cal_num(EASE_pred)

            EASE_pred = sp.csr_matrix(EASE_pred)
            EASE_pred = normalize(EASE_pred)
            EASE_pred = sparse_mx_to_torch_sparse_tensor(EASE_pred)
            print("EASE End")
        else:
            EASE_pred = torch.tensor([0])

        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)

        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)

        return UV_adj, VU_adj, EASE_pred