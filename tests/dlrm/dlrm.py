# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import argparse
import sys
import math
import time

# numpy
import numpy as np

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler

### define dlrm in PyTorch ###
from typing import List, Optional

from data import make_criteo_data_and_loaders, make_random_data_and_loader

from quiver import EmbeddingBag
from quiver import SynchronousOptimizer


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count - 1) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = (self.num_decay_steps - decayed_steps) / self.num_decay_steps
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, len(ln)):
            n = ln[i]

            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def create_quiver_emb(self, m, ln, rank, device_list, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, len(ln)):
            n = ln[i]

            W = torch.tensor(np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ), dtype=torch.float32)

            print("Creating Quiver EmbeddingBag {} * {}".format(n, m))
            EE = EmbeddingBag(n, m, "sum", rank, device_list, W)

            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
            self,
            args: argparse.Namespace,
            rank: int,
            device_list: list,
            m_spa: int,
            ln_emb: List[int],
            ln_bot: List[int],
            ln_top: List[int],
            arch_interaction_op: str,
            arch_interaction_itself: bool,
            sigmoid_bot: int = -1,
            sigmoid_top: int = -1,
            sync_dense_params: bool = True,
            loss_threshold: float = 0.0,
            ndevices: int = -1,
            weighted_pooling: Optional[str] = None,
            loss_function: str = "bce",
            use_quiver: bool = False
    ):
        """
        :param args: arguments
        :param m_spa: dimension of sparse features
        :param ln_emb: sizes of embedding tables
        :param ln_bot: sizes of bottom mlp layers
        :param ln_top: sizes of top mlp layers
        :param arch_interaction_op: interation operations
        :param arch_interaction_itself: if interact itself or not
        :param sigmoid_bot: position to use sigmoid in bottom layers
        :param sigmoid_top: position to use sigmoid in top layers
        :param sync_dense_params: if sync dense parameters
        :param loss_threshold: threshold of output
        :param ndevices: number of devices
        :param weighted_pooling: weighted pooling method
        :param loss_function: loss function
        """

        super(DLRM_Net, self).__init__()

        # save arguments
        self.ndevices = ndevices
        self.output_d = 0
        self.parallel_model_batch_size = -1
        self.parallel_model_is_not_prepared = True
        self.arch_interaction_op = arch_interaction_op
        self.arch_interaction_itself = arch_interaction_itself
        self.sync_dense_params = sync_dense_params
        self.loss_threshold = loss_threshold
        self.loss_function = loss_function
        if weighted_pooling is not None and weighted_pooling != "fixed":
            self.weighted_pooling = "learned"
        else:
            self.weighted_pooling = weighted_pooling

        # create operators
        if ndevices <= 1:
            if use_quiver:
                self.emb_l, w_list = self.create_quiver_emb(m_spa, ln_emb, rank, device_list, weighted_pooling)
            else:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
            if self.weighted_pooling == "learned":
                self.v_W_l = nn.ParameterList()
                for w in w_list:
                    self.v_W_l.append(Parameter(w))
            else:
                self.v_W_l = w_list
        self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
        self.top_l = self.create_mlp(ln_top, sigmoid_top)

        # specify the loss function
        if self.loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif self.loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        elif self.loss_function == "wbce":
            self.loss_ws = torch.tensor(
                np.fromstring(args.loss_weights, dtype=float, sep="-")
            )
            self.loss_fn = torch.nn.BCELoss(reduction="none")
        else:
            sys.exit(
                "ERROR: --loss-function=" + self.loss_function + " is not supported"
            )

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        print(len(lS_i))
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=per_sample_weights,
            )

            print(V.size())

            ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        if self.ndevices <= 1:
            # single device run
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            # single-node multi-device run
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be <= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            w_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                t_list.append(emb.to(d))
                if self.weighted_pooling == "learned":
                    w_list.append(Parameter(self.v_W_l[k].to(d)))
                elif self.weighted_pooling == "fixed":
                    w_list.append(self.v_W_l[k].to(d))
                else:
                    w_list.append(None)
            self.emb_l = nn.ModuleList(t_list)
            if self.weighted_pooling == "learned":
                self.v_W_l = nn.ParameterList(w_list)
            else:
                self.v_W_l = w_list
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dash_separated_ints(value):
    vals = value.split("-")
    processed_vals = []
    for val in vals:
        try:
            processed_vals.append(int(val))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return processed_vals


def dash_separated_floats(value):
    vals = value.split("-")
    processed_vals = []
    for val in vals:
        try:
            processed_vals.append(float(val))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return processed_vals


def dlrm_wrap(dlrm, X, lS_o, lS_i, use_gpu, device, ndevices=1):
    if use_gpu:  # .cuda()
        # lS_i can be either a list of tensors or a stacked tensor.
        # Handle each case below:
        if ndevices == 1:
            lS_i = (
                [S_i.to(device) for S_i in lS_i]
                if isinstance(lS_i, list)
                else lS_i.to(device)
            )
            lS_o = (
                [S_o.to(device) for S_o in lS_o]
                if isinstance(lS_o, list)
                else lS_o.to(device)
            )
    return dlrm(X.to(device), lS_o, lS_i)


def loss_fn_wrap(dlrm, loss_function, Z, T, device):
    if loss_function == "mse" or loss_function == "bce":
        return dlrm.loss_fn(Z, T.to(device))
    elif loss_function == "wbce":
        loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
        loss_fn_ = dlrm.loss_fn(Z, T.to(device))
        loss_sc_ = loss_ws_ * loss_fn_
        return loss_sc_.mean()


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


def inference(
        dlrm,
        test_ld,
        device,
        use_gpu,
        ndevices
):
    test_accu = 0
    test_samp = 0

    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        # if nbatches > 0 and i >= nbatches:
        #     break

        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )

        # forward pass
        Z_test = dlrm_wrap(
            dlrm,
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        ### gather the distributed results on each rank ###
        # For some reason it requires explicit sync before all_gather call if
        # tensor is on GPU memory
        if Z_test.is_cuda:
            torch.cuda.synchronize()

        # compute loss and accuracy
        S_test = Z_test.detach().cpu().numpy()  # numpy array
        T_test = T_test.detach().cpu().numpy()  # numpy array

        mbs_test = T_test.shape[0]  # = mini_batch_size except last
        A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

        test_accu += A_test
        test_samp += mbs_test

    acc_test = test_accu / test_samp
    return acc_test


def get_args():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-dimension", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=True)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    # Quiver Embedding
    parser.add_argument("--use-quiver", action="store_true", default=False)

    return parser.parse_args()


def run():
    rank = 0
    device_list = [0, 1]
    args = get_args()

    ### some basic setup ###
    np.random.seed(args.rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        ngpus = torch.cuda.device_count()
        ndevices = min(ngpus, args.mini_batch_size)
        device = torch.device("cuda", 0)
        print("Using {}/{} GPU(s)...".format(ndevices, ngpus))
    else:
        ngpus = -1
        ndevices = -1
        device = "cpu"

    m_spa = args.arch_sparse_feature_dimension
    ln_emb = args.arch_embedding_size
    ln_bot = args.arch_mlp_bot
    ln_top = args.arch_mlp_top
    arch_interaction_op = args.arch_interaction_op
    arch_interaction_itself = args.arch_interaction_itself

    assert m_spa == ln_bot[-1]

    # dataset
    if args.data_generation == "dataset":
        train_data, train_ld, val_data, val_ld = make_criteo_data_and_loaders(args)
        table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_val = len(val_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = list(
                map(
                    lambda x: x if x < args.max_ind_range else args.max_ind_range,
                    ln_emb,
                )
            )
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        m_den = ln_bot[0]
        train_data, train_ld, val_data, val_ld = make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_val = len(val_ld)

    if arch_interaction_op == "dot":
        diagonal = 1 if arch_interaction_itself else -1
        num_features = (len(ln_emb) + 1)
        ln_top = [m_spa + num_features * (num_features + diagonal) // 2] + ln_top
    elif arch_interaction_op == "cat":
        num_features = (len(ln_emb) + 1)
        ln_top = [m_spa * num_features] + ln_top
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + arch_interaction_op
            + " is not supported"
        )

    print(ndevices, m_spa, ln_emb, ln_bot, ln_top, arch_interaction_op, arch_interaction_itself)

    model = DLRM_Net(args,
                     rank,
                     device_list,
                     m_spa,
                     ln_emb,
                     ln_bot,
                     ln_top,
                     arch_interaction_op,
                     arch_interaction_itself,
                     sigmoid_top=len(ln_top) - 2,
                     sync_dense_params=args.sync_dense_params,
                     loss_threshold=args.loss_threshold,
                     ndevices=ndevices,
                     weighted_pooling=args.weighted_pooling,
                     loss_function=args.loss_function,
                     use_quiver=args.use_quiver
                     )

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        model = model.to(device)  # .cuda()
        if model.ndevices > 1:
            model.emb_l, model.v_W_l = model.create_emb(
                m_spa, ln_emb, args.weighted_pooling
            )
        else:
            if model.weighted_pooling == "fixed":
                for k, w in enumerate(model.v_W_l):
                    model.v_W_l[k] = w.cuda()

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = model.parameters()
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        if args.use_quiver:
            optimizer = SynchronousOptimizer(model.parameters(), optimizer)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )

        # training or inference
        best_acc_test = 0
        best_auc_test = 0
        total_time = 0
        total_loss = 0
        total_iter = 0
        total_samp = 0

        # if not args.inference_only:
        for i in range(args.nepochs):
            t1 = time_wrap(use_gpu)
            for j, inputBatch in enumerate(train_ld):
                print(j)
                model.train()
                X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

                # early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    break

                mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

                # forward pass
                Z = dlrm_wrap(
                    model,
                    X,
                    lS_o,
                    lS_i,
                    use_gpu,
                    device,
                    ndevices=ndevices,
                )

                # print(T.size(), Z.size())

                # loss
                E = loss_fn_wrap(model, args.loss_function, Z, T, device)

                # compute loss and accuracy
                L = E.detach().cpu().numpy()  # numpy array
                # training accuracy is not disabled
                # S = Z.detach().cpu().numpy()  # numpy array
                # T = T.detach().cpu().numpy()  # numpy array

                # # print("res: ", S)

                # # print("j, train: BCE ", j, L)

                # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                # A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                # scaled error gradient propagation
                # (where we do not accumulate gradients across mini-batches)
                optimizer.zero_grad()
                # backward pass
                E.backward()

                # optimizer
                optimizer.step()
                lr_scheduler.step()

                t2 = time_wrap(use_gpu)
                total_time += t2 - t1

                total_loss += L * mbs
                total_iter += 1
                total_samp += mbs

                should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                )
                should_test = (
                        (args.test_freq > 0)
                        and (args.data_generation in ["dataset", "random"])
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )

                # print time, loss and accuracy
                if should_print or should_test:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    train_loss = total_loss / total_samp
                    total_loss = 0

                    str_run_type = (
                        "inference" if args.inference_only else "training"
                    )

                    wall_time = ""
                    if args.print_wall_time:
                        wall_time = " ({})".format(time.strftime("%H:%M"))

                    print(
                        "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                            str_run_type, j + 1, nbatches, i, gT
                        )
                        + " loss {:.6f}".format(train_loss)
                        + wall_time,
                        flush=True,
                    )

                    total_iter = 0
                    total_samp = 0

                # testing
                if should_test:
                    print("Testing at - {}/{} of epoch {},".format(j + 1, nbatches, i))
                    acc_test = inference(
                        model,
                        val_ld,
                        device,
                        use_gpu,
                        ndevices
                    )

                    model_metrics_dict = {
                        "nepochs": args.nepochs,
                        "nbatches": nbatches,
                        "nbatches_test": nbatches_val,
                        "state_dict": model.state_dict(),
                        "test_acc": acc_test,
                    }

                    is_best = acc_test > best_acc_test
                    if is_best:
                        best_acc_test = acc_test
                    print(
                        "accuracy {:3.3f} %, best {:3.3f} %".format(
                            acc_test * 100, best_acc_test * 100
                        ),
                        flush=True,
                    )

                    if (
                            is_best
                            and not (args.save_model == "")
                            and not args.inference_only
                    ):
                        model_metrics_dict["epoch"] = i
                        model_metrics_dict["iter"] = j + 1
                        model_metrics_dict["train_loss"] = train_loss
                        model_metrics_dict["total_loss"] = total_loss
                        model_metrics_dict["opt_state_dict"] = optimizer.state_dict()
                        print("Saving model to {}".format(args.save_model))
                        torch.save(model_metrics_dict, args.save_model)

                    # Uncomment the line below to print out the total time with overhead
                    # print("Total test time for this group: {}" \
                    # .format(time_wrap(use_gpu) - accum_test_time_begin))
                t1 = time_wrap(use_gpu)
    else:
        print("Testing for inference only")
        acc_test = inference(
            model,
            val_ld,
            device,
            use_gpu,
            ndevices
        )
        print("accuracy {:3.3f} %".format(acc_test * 100), flush=True)


if __name__ == '__main__':
    print("Hi~")
    run()
