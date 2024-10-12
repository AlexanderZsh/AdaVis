#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import Query2box
from dataloader import *
from tensorboardX import SummaryWriter
import time
import pickle
import collections
from icecream import ic
ic.configureOutput(includeContext=True)
from tqdm import tqdm

def main(args):
    set_global_seed(args.seed)
    args.test_batch_size = 1
    assert args.bn in ['no', 'before', 'after']
    assert args.n_att >= 1 and args.n_att <= 3
    assert args.max_steps == args.stepsforpath
    if args.geo == 'box':
        assert 'Box' in args.model
    elif args.geo == 'vec':
        assert 'Box' not in args.model
        
    if args.train_onehop_only:
        assert '1c' in args.task
        args.center_deepsets = 'mean'
        if args.geo == 'box':
            args.offset_deepsets = 'min'

    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    cur_time = parse_time()
    print ("overide save string.")
    if args.task == '1c':
        args.stepsforpath = 0
    else:
        assert args.stepsforpath <= args.max_steps
    # logs_newfb237_inter
    
    args.save_path = 'logs/%s/%s/'%(args.data_path.split('/')[-1], args.geo)
    writer = SummaryWriter(args.save_path)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    set_logger(args)

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Geo: %s' % args.geo)
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('#stepsforpath: %d' % args.stepsforpath)

    tasks = args.task.split('.')
    train_ans = dict()
    valid_ans = dict()
    valid_ans_hard = dict()
    test_ans = dict()
    test_ans_hard = dict()

    if '1c' in tasks:
        with open('%s/train_triples_1c.pkl'%args.data_path, 'rb') as handle:
            train_triples = pickle.load(handle)
        with open('%s/train_ans_1c.pkl'%args.data_path, 'rb') as handle:
            train_ans_1 = pickle.load(handle)

        train_ans.update(train_ans_1)


    if '2c' in tasks:
        with open('%s/train_triples_2c.pkl'%args.data_path, 'rb') as handle:
            train_triples_2 = pickle.load(handle)
        with open('%s/train_ans_2c.pkl'%args.data_path, 'rb') as handle:
            train_ans_2 = pickle.load(handle)

        train_ans.update(train_ans_2)


    if '46i' in tasks:
        with open('%s/train_triples_46i.pkl'%args.data_path, 'rb') as handle:
            train_triples_46i = pickle.load(handle)
        with open('%s/train_ans_46i.pkl'%args.data_path, 'rb') as handle:
            train_ans_46i = pickle.load(handle)

        train_ans.update(train_ans_46i)



    if 'ic' in tasks:

        with open('%s/test_triples_ic.pkl'%args.data_path, 'rb') as handle:
            test_triples_ic = pickle.load(handle)
        with open('%s/test_ans_ic.pkl'%args.data_path, 'rb') as handle:
            test_ans_ic = pickle.load(handle)

        test_ans.update(test_ans_ic)


    if '1c' in tasks:
        logging.info('#train: %d' % len(train_triples))

    
    if '2c' in tasks:
        logging.info('#train_2c: %d' % len(train_triples_2))


    
    if '46i' in tasks:
        logging.info('#train_46i: %d' % len(train_triples_46i))

    

    
    if 'ic' in tasks:
        logging.info('#test_ic: %d' % len(test_triples_ic))



    query2box = Query2box(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        writer=writer,
        geo=args.geo,
        cen=args.center_reg,
        offset_deepsets = args.offset_deepsets,
        center_deepsets = args.center_deepsets,
        offset_use_center = args.offset_use_center,
        center_use_offset = args.center_use_offset,
        att_reg = args.att_reg,
        off_reg = args.off_reg,
        att_tem = args.att_tem,
        euo = args.entity_use_offset,
        gamma2 = args.gamma2,
        bn = args.bn,
        nat = args.n_att,
        activation = args.activation
    )
    
    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in query2box.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        query2box = query2box.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        if '1c' in tasks:
            """
            The tail in the train_triples is not true. The true answer in the train_ans
            """
            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn
            )
            train_iterator = SingledirectionalOneShotIterator(train_dataloader_tail, train_triples[0][-1])

        if '2c' in tasks:
            train_dataloader_2_tail = DataLoader(
                TrainDataset(train_triples_2, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn
            )
            train_iterator_2 = SingledirectionalOneShotIterator(train_dataloader_2_tail, train_triples_2[0][-1])

        if '46i' in tasks:

            train_dataloader_46i_tail = DataLoader(
                TrainInterDataset(train_triples_46i, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn
            )
            train_iterator_46i = SingledirectionalOneShotIterator(train_dataloader_46i_tail, train_triples_46i[0][-1])

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, query2box.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        query2box.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step 

    logging.info('task = %s' % args.task)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # # Set valid dataloader as it would be evaluated during training
    
    def evaluate_test():

        if 'ic' in tasks:
            metrics = query2box.test_step(query2box, test_triples_ic, test_ans, args)


    if args.do_train:
        training_logs = []
        if args.task == '1c':
            begin_pq_step = args.max_steps
        else:
            begin_pq_step = args.max_steps - args.stepsforpath
        #Training Loop
        for step in tqdm(range(init_step, args.max_steps)):
            # print ("begining training step", step)
            # if step == 100:
            #     exit(-1)
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            if step >= begin_pq_step and not args.train_onehop_only:
                if '46i' in tasks:
                    log = query2box.train_step(query2box, optimizer, train_iterator_46i, args, step)
                    for metric in log:
                        writer.add_scalar('2i_'+metric, log[metric], step)
                    training_logs.append(log)
                

                if '2c' in tasks:
                    log = query2box.train_step(query2box, optimizer, train_iterator_2, args, step)
                    for metric in log:
                        writer.add_scalar('2c_'+metric, log[metric], step)
                    training_logs.append(log)
                


            if '1c' in tasks:
                log = query2box.train_step(query2box, optimizer, train_iterator, args, step)
                for metric in log:
                    writer.add_scalar('1c_'+metric, log[metric], step)
                training_logs.append(log)

            if training_logs == []:
                raise Exception("No tasks are trained!!")

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, query2box.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(query2box, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    if metric == 'inter_loss':
                        continue
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                inter_loss_sum = 0.
                inter_loss_num = 0.
                for log in training_logs:
                    if 'inter_loss' in log:
                        inter_loss_sum += log['inter_loss']
                        inter_loss_num += 1
                if inter_loss_num != 0:
                    metrics['inter_loss'] = inter_loss_sum / inter_loss_num
                log_metrics('Training average', step, metrics)
                training_logs = []
            


        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(query2box, optimizer, save_variable_list, args)
        
    try:
        print (step)
    except:
        step = 0


    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        evaluate_test()


    print ('Training finished!!')
    logging.info("training finished!!")
