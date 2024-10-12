#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
from posix import environ
import random

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler
from torch.utils.data import DataLoader

from model import Query2box
from dataloader import *
from weighted_sampler import DistributedSamplerWrapper

from tensorboardX import SummaryWriter
import time
import pickle
import collections
from torch.utils.data.distributed import DistributedSampler
from math import sqrt
from torch.cuda.amp import GradScaler

from icecream import ic
from torch.optim import lr_scheduler

ic.configureOutput(includeContext=True)
from tqdm import tqdm
from collections import Counter


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training and Testing Knowledge Graph Embedding Models", usage="train.py [<args>] [-h | --help]"
    )

    parser.add_argument("--cuda", action="store_true", help="use GPU")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--evaluate_train", action="store_true", help="Evaluate on training data")

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model", default="TransE", type=str)

    parser.add_argument("-n", "--negative_sample_size", default=128, type=int)
    parser.add_argument("-d", "--hidden_dim", default=500, type=int)
    parser.add_argument("-g", "--gamma", default=12.0, type=float)
    parser.add_argument("-adv", "--negative_adversarial_sampling", action="store_true")
    parser.add_argument("-a", "--adversarial_temperature", default=1.0, type=float)
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument("-r", "--regularization", default=0.0, type=float)
    parser.add_argument("--test_batch_size", default=4, type=int, help="valid/test batch size")
    parser.add_argument(
        "--uni_weight", action="store_true", help="Otherwise use subsampling weighting like in word2vec"
    )

    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float)
    parser.add_argument("-cpu", "--cpu_num", default=10, type=int)
    parser.add_argument("-init", "--init_checkpoint", default=None, type=str)
    parser.add_argument("-save", "--save_path", default=None, type=str)
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--warm_up_steps", default=None, type=int)

    parser.add_argument("--save_checkpoint_steps", default=10000, type=int)
    parser.add_argument("--valid_steps", default=10000, type=int)
    parser.add_argument("--log_steps", default=100, type=int, help="train log every xx steps")
    parser.add_argument("--test_log_steps", default=1000, type=int, help="valid/test log every xx steps")

    parser.add_argument("--nentity", type=int, default=0, help="DO NOT MANUALLY SET")
    parser.add_argument("--nrelation", type=int, default=0, help="DO NOT MANUALLY SET")

    parser.add_argument("--geo", default="vec", type=str, help="vec or box")
    parser.add_argument("--print_on_screen", action="store_true")

    parser.add_argument("--task", default="1c.2c.3c.2i.3i", type=str)
    parser.add_argument("--stepsforpath", type=int, default=0)

    parser.add_argument("--offset_deepsets", default="vanilla", type=str, help="inductive or vanilla or min")
    parser.add_argument("--offset_use_center", action="store_true")
    parser.add_argument("--center_deepsets", default="vanilla", type=str, help="vanilla or attention or mean")
    parser.add_argument("--center_use_offset", action="store_true")
    parser.add_argument("--entity_use_offset", action="store_true")
    parser.add_argument("--att_reg", default=0.0, type=float)
    parser.add_argument("--off_reg", default=0.0, type=float)
    parser.add_argument("--att_tem", default=1.0, type=float)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gamma2", default=0, type=float)
    parser.add_argument("--train_onehop_only", action="store_true")
    parser.add_argument("--center_reg", default=0.0, type=float, help="alpha in the paper")
    parser.add_argument("--bn", default="no", type=str, help="no or before or after")
    parser.add_argument("--n_att", type=int, default=1)
    parser.add_argument("--activation", default="relu", type=str, help="relu or none or softplus")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--scheduler", default=None, type=str)

    return parser.parse_args(args)


def override_config(args):  #! may update here
    """
    Override model and data configuration
    """

    with open(os.path.join(args.init_checkpoint, "config.json"), "r") as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict["data_path"]
    args.model = argparse_dict["model"]
    args.hidden_dim = argparse_dict["hidden_dim"]
    args.test_batch_size = argparse_dict["test_batch_size"]


def save_model(model, optimizer, save_variable_list, args, before_finetune=False):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """

    model_to_save = model.module if hasattr(model, "module") else model
    if args.local_rank in [-1, 0]:
        argparse_dict = vars(args)
        with open(
            os.path.join(args.save_path, "config.json" if not before_finetune else "config_before.json"), "w"
        ) as fjson:
            json.dump(argparse_dict, fjson)
        step = save_variable_list["step"]
        torch.save(
            {
                **save_variable_list,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(args.save_path, ("checkpoint" if not before_finetune else "checkpoint_before") + f"_{step}"),
        )

        entity_embedding = model_to_save.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, "entity_embedding" if not before_finetune else "entity_embedding_before"),
            entity_embedding,
        )

        relation_embedding = model_to_save.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, "relation_embedding" if not before_finetune else "relation_embedding_before"),
            relation_embedding,
        )

        print("model save")


def set_logger(args):
    """
    Write logs to checkpoint and console
    """

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, "train.log")
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, "test.log")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs
    """
    for metric in metrics:
        logging.info("%s %s at step %d: %f" % (mode, metric, step, metrics[metric]))


def main(args):
    eval_result = []

    args.local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))

    # * Not distributed training
    if args.local_rank == -1:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logging.warning("using cpu right now !!!!!!!!!!!!")
            device = torch.device("cpu")
    # * distributed training
    else:
        # * distributed: allocate each process gpu
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        logging.warning("Initializing process group == local rank %s ==" % args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    print(f"[init]==local rank: {args.local_rank}==")

    set_global_seed(args.seed)
    args.test_batch_size = 1
    assert args.bn in ["no", "before", "after"]
    assert args.n_att >= 1 and args.n_att <= 3
    assert args.max_steps == args.stepsforpath
    if args.geo == "box":
        assert "Box" in args.model
    elif args.geo == "vec":
        assert "Box" not in args.model

    if args.train_onehop_only:
        assert "1c" in args.task
        args.center_deepsets = "mean"
        if args.geo == "box":
            args.offset_deepsets = "min"

    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError("one of train/val/test mode must be chosen.")

    # * reuse previous saved checkpoint
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError("one of init_checkpoint/data_path must be chosen.")

    if args.task == "1c":
        args.stepsforpath = 0
    else:
        assert args.stepsforpath <= args.max_steps

    # * create file with timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.save_path = args.save_path + f"/{timestr}"
    if args.debug_mode:
        args.save_path = "./logs/KG4VIS/playground"
    elif args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    else:
        print("Not debug mode and ", args.save_path, " exists")

    print("save path ", args.save_path)
    writer = SummaryWriter(args.save_path)
    # * write logger to console
    set_logger(args)

    with open("%s/stats.txt" % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(" ")[-1])
        nrelation = int(entrel[1].split(" ")[-1])
        
    with open('%s/entities.dict' % args.data_path) as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity
    with open('%s/relations.dict' % args.data_path)as fin:
        relation2id = dict()
        id2relation = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation
    
    args.entity2id = entity2id
    args.nentity = nentity
    args.nrelation = nrelation


    logging.info("Geo: %s" % args.geo)
    logging.info("Model: %s" % args.model)
    logging.info("Data Path: %s" % args.data_path)
    logging.info("#entity: %d" % nentity)
    logging.info("#relation: %d" % nrelation)
    logging.info("#max steps: %d" % args.max_steps)
    logging.info("#stepsforpath: %d" % args.stepsforpath)

    tasks = args.task.split(".")
    train_ans = dict()
    valid_ans = dict()
    test_ans = dict()
    # test_ans = None

    if "1c" in tasks:

        with open("%s/train_triples_1c.pkl" % args.data_path, "rb") as handle:
            train_triples = pickle.load(handle)
        with open("%s/train_ans_1c.pkl" % args.data_path, "rb") as handle:
            train_ans_1 = pickle.load(handle)

        with open("%s/valid_triples_1c.pkl" % args.data_path, "rb") as handle:
            valid_triples = pickle.load(handle)
        with open("%s/valid_ans_1c.pkl" % args.data_path, "rb") as handle:
            valid_ans_1 = pickle.load(handle)

        train_ans.update(train_ans_1)
        valid_ans.update(valid_ans_1)

    # * 2i: columns -> cross column feature; columns->dataset
    if "2i" in tasks:
        with open("%s/train_triples_2i.pkl" % args.data_path, "rb") as handle:
            train_triples_2i = pickle.load(handle)
        with open("%s/train_ans_2i.pkl" % args.data_path, "rb") as handle:
            train_ans_2i = pickle.load(handle)


        train_ans.update(train_ans_2i)

    # * 40i: corss features -> dataset
    if "40i" in tasks:
        with open("%s/train_triples_40i.pkl" % args.data_path, "rb") as handle:
            train_triples_40i = pickle.load(handle)

        with open("%s/train_ans_40i.pkl" % args.data_path, "rb") as handle:
            train_ans_40i = pickle.load(handle)

        train_ans.update(train_ans_40i)
    # * 80i: single column features -> column
    if "80i" in tasks:
        with open("%s/train_triples_80i.pkl" % args.data_path, "rb") as handle:
            train_triples_80i = pickle.load(handle)
        with open("%s/train_ans_80i.pkl" % args.data_path, "rb") as handle:
            train_ans_80i = pickle.load(handle)

        with open("%s/valid_triples_80i.pkl" % args.data_path, "rb") as handle:
            valid_triples_80i = pickle.load(handle)
        with open("%s/valid_ans_80i.pkl" % args.data_path, "rb") as handle:
            valid_ans_80i = pickle.load(handle)
        train_ans.update(train_ans_80i)
        valid_ans.update(valid_ans_80i)

    if "45i" in tasks:
        with open("%s/train_triples_45i.pkl" % args.data_path, "rb") as handle:
            train_triples_45i = pickle.load(handle)
        with open("%s/train_ans_45i.pkl" % args.data_path, "rb") as handle:
            train_ans_45i = pickle.load(handle)

        with open("%s/valid_triples_45i.pkl" % args.data_path, "rb") as handle:
            valid_triples_45i = pickle.load(handle)
        with open("%s/valid_ans_45i.pkl" % args.data_path, "rb") as handle:
            valid_ans_45i = pickle.load(handle)
        train_ans.update(train_ans_45i)
        valid_ans.update(valid_ans_45i)

    if "42i" in tasks:
        with open("%s/train_triples_42i.pkl" % args.data_path, "rb") as handle:
            train_triples_42i = pickle.load(handle)
        with open("%s/train_ans_42i.pkl" % args.data_path, "rb") as handle:
            train_ans_42i = pickle.load(handle)

        with open("%s/valid_triples_42i.pkl" % args.data_path, "rb") as handle:
            valid_triples_42i = pickle.load(handle)
        with open("%s/valid_ans_42i.pkl" % args.data_path, "rb") as handle:
            valid_ans_42i = pickle.load(handle)
        train_ans.update(train_ans_42i)
        valid_ans.update(valid_ans_42i)

    if "ic" in tasks:
        with open("%s/test_axis_ic.pkl" % args.data_path, "rb") as handle:
            test_axis_ic = pickle.load(handle)
        with open("%s/axis_ans_ic.pkl" % args.data_path, "rb") as handle:
            axis_ans_ic = pickle.load(handle)
        # test_ans.update(axis_ans_ic)

    if "2ic" in tasks:

        with open("%s/test_triples_2ic.pkl" % args.data_path, "rb") as handle:
            test_triples_2ic = pickle.load(handle)



        for sample in test_triples_2ic:
            ans = sample[-2]
            triple  = sample[:-2]
            if triple in test_ans:
                test_ans[triple].add(ans)
            else:
                test_ans[triple] = set()
                test_ans[triple].add(ans)

    if "1c" in tasks:
        logging.info("#train_1c: %d" % len(train_triples))
        logging.info("#valid_1c: %d" % len(valid_triples))

    if "2i" in tasks:
        logging.info("#train_2i: %d" % len(train_triples_2i))

    if "40i" in tasks:
        logging.info("#train_40i: %d" % len(train_triples_40i))
    if "80i" in tasks:
        logging.info("#train_80i: %d" % len(train_triples_80i))
        logging.info("#valid_80i: %d" % len(valid_triples_80i))
    if "45i" in tasks:
        logging.info("#train_45i: %d" % len(train_triples_45i))
        logging.info("#valid_45i: %d" % len(valid_triples_45i))
    if "42i" in tasks:
        logging.info("#train_42i: %d" % len(train_triples_42i))
        logging.info("#valid_42i: %d" % len(valid_triples_42i))

    if "ic" in tasks:
        logging.info("#test_axis_ic: %d" % len(test_axis_ic))
    if "2ic" in tasks:
        logging.info("#test_2ic: %d" % len(test_triples_2ic))

    query2box = Query2box(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        writer=writer,
        geo=args.geo,
        cen=args.center_reg,
        offset_deepsets=args.offset_deepsets,
        center_deepsets=args.center_deepsets,
        offset_use_center=args.offset_use_center,
        center_use_offset=args.center_use_offset,
        att_reg=args.att_reg,
        off_reg=args.off_reg,
        att_tem=args.att_tem,
        euo=args.entity_use_offset,
        gamma2=args.gamma2,
        bn=args.bn,
        nat=args.n_att,
        activation=args.activation,
    )

    logging.info("Model Parameter Configuration:")
    num_params = 0
    for name, param in query2box.named_parameters():
        logging.info("Parameter %s: %s, require_grad = %s" % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info("Parameter Number: %d" % num_params)

    # * distributed: model to device
    query2box.to(device)

    if args.local_rank != -1:
        logging.info("let's use ", args.local_rank, "GPUs!")
        query2box = torch.nn.parallel.DistributedDataParallel(
            query2box, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    if args.do_train:
        # Set training dataloader iterator
        if "1c" in tasks:
            """
            The tail in the train_triples is not true. The true answer in the train_ans
            """


            # * create weighted sampler for distribution
            counter = Counter([i[0][1] for i in train_triples])
            weight_per_relation = {i: 1 / (counter[i] + 5) for i in counter}
            array_weight = np.array([weight_per_relation[t[0][1]] for t in train_triples])
            # NOTE: using weighted sampler
            # weighted_sampler_1c = WeightedRandomSampler(array_weight, len(array_weight))
            weighted_sampler_1c = RandomSampler(train_triples)
            if args.local_rank != -1:
                weighted_sampler_1c = DistributedSamplerWrapper(weighted_sampler_1c)

            train_dataset_1c = TrainDataset(
                train_triples, nentity, nrelation, args.negative_sample_size, train_ans, "tail-batch"
            )

            train_dataloader_tail = DataLoader(
                train_dataset_1c,
                batch_size=args.batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn,
                sampler=weighted_sampler_1c,
            )
            train_iterator = SingledirectionalOneShotIterator(train_dataloader_tail, train_triples[0][-1])

        if "2i" in tasks:
            train_dataset_2i = TrainInterDataset(
                train_triples_2i, nentity, nrelation, args.negative_sample_size, train_ans, "tail-batch"
            )

            if args.local_rank != -1:
                sampler_2i = DistributedSampler(train_dataset_2i)
            else:
                sampler_2i = RandomSampler(train_triples_2i)

            train_dataloader_2i_tail = DataLoader(
                train_dataset_2i,
                batch_size=args.batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                sampler=sampler_2i,
            )
            train_iterator_2i = SingledirectionalOneShotIterator(train_dataloader_2i_tail, train_triples_2i[0][-1])

        if "80i" in tasks:
            train_dataset_80i = TrainInterDataset(
                train_triples_80i, nentity, nrelation, args.negative_sample_size, train_ans, "tail-batch"
            )
            if args.local_rank != -1:
                sampler_80i = DistributedSampler(train_dataset_80i)
            else:
                sampler_80i = RandomSampler(train_triples_80i)

            train_dataloader_80i_tail = DataLoader(
                train_dataset_80i,
                batch_size=args.batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                sampler=sampler_80i,
            )
            train_iterator_80i = SingledirectionalOneShotIterator(train_dataloader_80i_tail, train_triples_80i[0][-1])
        if "45i" in tasks:
            train_dataset_45i = TrainInterDataset(
                train_triples_45i, nentity, nrelation, args.negative_sample_size, train_ans, "tail-batch"
            )
            if args.local_rank != -1:
                sampler_45i = DistributedSampler(train_dataset_45i)
            else:
                sampler_45i = RandomSampler(train_triples_45i)

            train_dataloader_45i_tail = DataLoader(
                train_dataset_45i,
                batch_size=args.batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                sampler=sampler_45i,
            )
            train_iterator_45i = SingledirectionalOneShotIterator(train_dataloader_45i_tail, train_triples_45i[0][-1])

        if "40i" in tasks:
            train_dataset_40i = TrainInterDataset(
                train_triples_40i, nentity, nrelation, args.negative_sample_size, train_ans, "tail-batch"
            )
            if args.local_rank != -1:
                sampler_40i = DistributedSampler(train_triples_40i)
            else:
                sampler_40i = RandomSampler(train_triples_40i)
            train_dataloader_40i_tail = DataLoader(
                train_dataset_40i,
                batch_size=args.batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                sampler=sampler_40i,
            )
            train_iterator_40i = SingledirectionalOneShotIterator(train_dataloader_40i_tail, train_triples_40i[0][-1])

        if "42i" in tasks:
            train_dataset_42i = TrainInterDataset(
                train_triples_42i, nentity, nrelation, args.negative_sample_size, train_ans, "tail-batch"
            )
            if args.local_rank != -1:
                sampler_42i = DistributedSampler(train_triples_42i)
            else:
                sampler_42i = RandomSampler(train_triples_42i)
            train_dataloader_42i_tail = DataLoader(
                train_dataset_42i,
                batch_size=args.batch_size,
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn,
                sampler=sampler_42i,
            )
            train_iterator_42i = SingledirectionalOneShotIterator(train_dataloader_42i_tail, train_triples_42i[0][-1])

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, query2box.parameters()), lr=current_learning_rate
        )

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        elif args.scheduler:
            warm_up_steps = None
            if args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "max", factor=0.8, patience=10, verbose=True, min_lr=0.00001
                )
        else:
            raise "must choose an adaptive lr method"



    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info("Loading checkpoint %s..." % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, "checkpoint"))
        init_step = checkpoint["step"]
        query2box.load_state_dict(checkpoint["model_state_dict"])
        if args.do_train:
            current_learning_rate = checkpoint["current_learning_rate"]
            warm_up_steps = checkpoint["warm_up_steps"]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        logging.info("Ramdomly Initializing %s Model..." % args.model)
        init_step = 0

    step = init_step

    logging.info("task = %s" % args.task)
    logging.info("init_step = %d" % init_step)
    if args.do_train:
        logging.info("Start Training...")
        logging.info("learning_rate = %d" % current_learning_rate)
    logging.info("batch_size = %d" % args.batch_size)
    logging.info("negative_adversarial_sampling = %d" % args.negative_adversarial_sampling)
    logging.info("hidden_dim = %d" % args.hidden_dim)
    logging.info("gamma = %f" % args.gamma)
    logging.info("negative_adversarial_sampling = %s" % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info("adversarial_temperature = %f" % args.adversarial_temperature)

    # # Set valid dataloader as it would be evaluated during training

    def evaluate_test(scheduler):



        assert not ("ic" in tasks and "2ic" in tasks)
        if "ic" in tasks:
            metrics = Query2box.test_step(query2box, test_axis_ic, test_ans, args)
            for metric in metrics:
                writer.add_scalar("Axis_ic_" + metric, metrics[metric], step)
                logging.info(f"{metric}:{metrics[metric]}")
            scheduler.step(metrics["HITS@1m_new"])

        if "2ic" in tasks:
            metrics = Query2box.test_step(query2box, test_triples_2ic, test_ans, args)
            for metric in metrics:
                writer.add_scalar("Test_2ic_" + metric, metrics[metric], step)
                logging.info(f"{metric}:{metrics[metric]}")
            scheduler.step(metrics["HITS@2m_new"])



    def evaluate_train():

        if "80i" in tasks:
            metrics, _, _ = Query2box.test_step(query2box, valid_triples_80i, valid_ans, args)
            log_metrics("valid 80i", step, metrics)
            for metric in metrics:
                writer.add_scalar("valid_80i_" + metric, metrics[metric], step)

        if "45i" in tasks:
            metrics, _, _ = Query2box.test_step(query2box, valid_triples_45i, valid_ans, args)
            log_metrics("valid 45i", step, metrics)
            for metric in metrics:
                writer.add_scalar("valid_45i_" + metric, metrics[metric], step)

        if "42i" in tasks:
            metrics, _, _ = Query2box.test_step(query2box, valid_triples_42i, valid_ans, args)
            log_metrics("valid 42i", step, metrics)
            for metric in metrics:
                writer.add_scalar("valid_42i_" + metric, metrics[metric], step)

        if "1c" in tasks:

            metrics, _, _ = Query2box.test_step(query2box, valid_triples, valid_ans, args)
            log_metrics("valid 1c", step, metrics)
            for metric in metrics:
                if ("1000" not in metric) and ("5000" not in metric):
                    print(metric)
                    writer.add_scalar("valid_1c_" + metric, metrics[metric], step)

    if args.do_train:
        training_logs = []

        if args.task == "1c":
            begin_pq_step = args.max_steps
        else:
            # * begin_pq_step is 0
            begin_pq_step = args.max_steps - args.stepsforpath
        print("using amp: ", args.use_amp)
        if args.use_amp:
            scaler = GradScaler()
        else:
            scaler = None
        # Training Loop
        for step in tqdm(range(init_step, args.max_steps)):

            if step >= begin_pq_step and not args.train_onehop_only:
                if "80i" in tasks:
                    log = Query2box.train_step(query2box, optimizer, train_iterator_80i, args, scaler, step)
                    if args.local_rank in [-1, 0]:
                        for metric in log:
                            writer.add_scalar("80i_" + metric, log[metric], step)
                    training_logs.append(log)

                if "2i" in tasks:
                    log = Query2box.train_step(query2box, optimizer, train_iterator_2i, args, scaler, step)
                    if args.local_rank in [-1, 0]:
                        for metric in log:
                            writer.add_scalar("2i_" + metric, log[metric], step)
                    training_logs.append(log)

                if "45i" in tasks:
                    log = Query2box.train_step(query2box, optimizer, train_iterator_45i, args, scaler, step)
                    if args.local_rank in [-1, 0]:
                        for metric in log:
                            writer.add_scalar("45i_" + metric, log[metric], step)
                    training_logs.append(log)

                if "40i" in tasks:
                    log = Query2box.train_step(query2box, optimizer, train_iterator_40i, args, scaler, step)
                    if args.local_rank in [-1, 0]:
                        for metric in log:
                            writer.add_scalar("40i_" + metric, log[metric], step)
                    training_logs.append(log)

                if "42i" in tasks:
                    log = Query2box.train_step(query2box, optimizer, train_iterator_42i, args, scaler, step)
                    if args.local_rank in [-1, 0]:
                        for metric in log:
                            writer.add_scalar("42i_" + metric, log[metric], step)
                    training_logs.append(log)

            if "1c" in tasks:
                log = Query2box.train_step(query2box, optimizer, train_iterator, args, scaler, step)
                if args.local_rank in [-1, 0]:
                    for metric in log:
                        writer.add_scalar("1c_" + metric, log[metric], step)
                training_logs.append(log)

            if training_logs == []:
                raise Exception("No tasks are trained!!")

            # NOTE: warm up is optional can be replace by other scheduler method
            if args.warm_up_steps and step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info("Change learning_rate to %f at step %d" % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, query2box.parameters()), lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 4

            if step % args.save_checkpoint_steps == 0:

                save_variable_list = {
                    "step": step,
                    "current_learning_rate": current_learning_rate,
                    "warm_up_steps": warm_up_steps,
                }
                save_model(query2box, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    if metric == "inter_loss":
                        continue
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                inter_loss_sum = 0.0
                inter_loss_num = 0.0
                for log in training_logs:
                    if "inter_loss" in log:
                        inter_loss_sum += log["inter_loss"]
                        inter_loss_num += 1
                if inter_loss_num != 0:
                    metrics["inter_loss"] = inter_loss_sum / inter_loss_num
                log_metrics("Training average", step, metrics)
                training_logs = []

            if args.do_test and step % args.valid_steps == 0 and args.local_rank in [-1, 0]:

                logging.info("Evaluating on Test Dataset...")
                evaluate_test(scheduler)

            if args.evaluate_train and step % args.valid_steps == 0 and args.local_rank in [-1, 0]:
                logging.info("Evaluating on Training Dataset...")
                evaluate_train()


        save_variable_list = {
            "step": step,
            "current_learning_rate": current_learning_rate,
            "warm_up_steps": warm_up_steps,
        }
        save_model(query2box, optimizer, save_variable_list, args)

    try:
        print(step)
    except:
        step = 0


    print("Training finished!!")
    logging.info("training finished!!")


if __name__ == "__main__":
    main(parse_args())
