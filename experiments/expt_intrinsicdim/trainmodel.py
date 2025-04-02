#load packages
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import math, time
import sys
sys.path.append('../..')
import os
from functions.train_test import train, test
from functions.get_data import data_n_loaders
from models import SAE
import wandb
import argparse
from datetime import date
from functions.utils import load_args_from_file, softplus_inverse, resample_deadlatents
from torchsummary import summary
import csv
import argparse
import time
from functions.utils import read_hyperparameters

if __name__=='__main__':
    #read sae width from hyperparameters file in array job
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task_id', type=int, required=True, 
                            help='SLURM task ID')
    argsX = parser.parse_args()
    hyperparameters = read_hyperparameters(argsX.task_id - 1, './hyperparameters2.csv')
    if hyperparameters is not None:
        sae_type = hyperparameters['sae_type']
        kval_topk = int(hyperparameters['kval_topk'])
        gamma_reg = float(hyperparameters['gamma_reg'])
        # scale = hyperparameters['scale']
    else:
        raise ValueError(f"No sae width found for task {argsX.task_id}")
    
    #create necessary subfolders for status files, models, figures
    ARGS_FROM_FILE = True
    RESULTS_PATH = './'
    FIGURES_PATH = RESULTS_PATH+'figs/'
    LAB_DIR = os.environ['USERDIR']
    DATA_PATH = LAB_DIR+'/data'
    STATUS_PATH = RESULTS_PATH + 'status_files/'
    SAVE_MODELS_PATH = RESULTS_PATH+'saved_models/'

    #create directories if they don't exist
    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)
    if not os.path.exists(STATUS_PATH):
        os.makedirs(STATUS_PATH)
    if not os.path.exists(SAVE_MODELS_PATH):
        os.makedirs(SAVE_MODELS_PATH)

    args = load_args_from_file('./settings.txt')

    #from hyperparameters file
    args.sae_type = sae_type 
    args.kval_topk = kval_topk
    args.gamma_reg = gamma_reg
    
    sae_width = args.sae_width
    device = args.device
    if device=='cuda':
        torch.cuda.empty_cache()
    # args.regularizer = args.regularizer if args.regularizer!='None' else None
    #set regularizer based on nonlinearity
    if args.regularizer == 'default':
        if args.sae_type=='relu':
            args.regularizer = 'l1'
        elif args.sae_type=='topk':
            args.regularizer = None
        elif args.sae_type=='topk_relu':
            # args.regularizer = None
            args.regularizer = 'auxloss'
        elif args.sae_type=='jumprelu':
            args.regularizer = 'l0'
        elif args.sae_type=='sparsemax_dist':
            args.regularizer = 'dist_weighted_l1'
    else:
        args.regularizer = args.regularizer if args.regularizer!='None' else None

    if args.gamma_reg=='default':
        if args.regularizer in ['l1', 'dist_weighted_l1', None]:
            args.gamma_reg = 0.1
        elif args.regularizer=='l0':
            args.gamma_reg = 0.01 #l0 loss observed to be larger; smaller gamma to compensate
        elif args.regularizer=='auxloss':
            args.gamma_reg = 1.0

    if args.normalize_decoder=='default':
        if args.sae_type=='relu':
            args.normalize_decoder = True
        elif args.sae_type=='topk':
            args.normalize_decoder = True
        elif args.sae_type=='topk_relu':
            args.normalize_decoder = True
        elif args.sae_type=='jumprelu':
            args.normalize_decoder = True
        elif args.sae_type=='sparsemax_dist':
            args.normalize_decoder = False

    #training params
    # LEARNING_RATE = 1e-2
    MOMENTUM = 0.9
    # WEIGHT_DECAY = 1e-4
    
    #update- to see if weight decay causes 0 weights
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 0.0
    #experiment name using random words
    from wonderwords import RandomWord

    import random
    seedx = random.randint(0, 1000)
    random.seed(seedx)
    r = RandomWord()
    word = r.word(word_min_length=2, word_max_length=5)
    date = date.today().strftime("%m%d%y") if args.experiment_date=='today' else args.experiment_date

    # if sae_width<args.kval_topk:
    #     args.kval_topk = sae_width
    kvalue_str = "k"+str(args.kval_topk)+"_" if 'topk' in args.sae_type else ''
    gamreg_str = "gamreg"+str(args.gamma_reg)+"_" if args.sae_type!='topk_relu' else ''
    saename = args.sae_type if args.sae_type!='sparsemax_dist' else 'spade'
    EXPT_NAME = word+str(seedx)+"_"+saename +"_"+ \
        kvalue_str + gamreg_str + date

    #status file
    STATUS_FILE = STATUS_PATH+'status_'+EXPT_NAME+'.txt'
    def update_status(text, option="a+"):
        with open(STATUS_FILE, option) as f:
            f.write('\n'+text)

    update_status(f"Using {device} device",option='w') #mention device

    for arg, value in vars(args).items(): #write experiment settings into status file
        update_status(f"{arg}: {value}")

    #load data (preprocessed)
    train_dataloader, test_dataloader,\
        train_data, test_data = data_n_loaders(args.dataset, args.batch_size, \
                                                return_data=True, data_path=DATA_PATH,\
                                                    standardise_data = True)

    #define model 
    model = SAE(dimin=args.data_dim, width=sae_width, sae_type=args.sae_type, \
        kval_topk=args.kval_topk, normalize_decoder=args.normalize_decoder)
    model = model.to(device)

    #data init of weights
    if args.weight_init=='data': #data init
        torch.manual_seed(args.seed_id)
        num_train_ex = len(train_data)
        indices_examples = torch.randperm(num_train_ex)
        indices_examples = indices_examples[:sae_width] #choose sae_width indices
        enc_init = torch.zeros((sae_width, args.data_dim)).to(device)
        with torch.no_grad():
            for k in range(sae_width):
                enc_init[k,:] = train_data[indices_examples[k]][0].squeeze().to(device)
            model.Ae.copy_(enc_init)
            model.Ad.copy_(enc_init.T)

    if args.optimizer=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    elif args.optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, \
                                    weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError("optimizer must be one of 'sgd', 'adam'")
        

    #rewrite wandb config
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.wandbprojectname,
        name=EXPT_NAME,
        # track hyperparameters and run metadata
        config=vars(args),
        entity='harvard01'
    )

    EPOCHSnum = math.ceil(len(train_data)/args.batch_size) if args.online_training else EPOCHS
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHSnum, eta_min=1e-4) 

    #train the model
    train_loss_epoch = [None for i in range(EPOCHSnum)]
    test_loss_epoch = [None for i in range(EPOCHSnum)]
    lambda_vals = [None for i in range(EPOCHSnum)]

    if args.save_checkpoints:
        checkpoint_loc = SAVE_MODELS_PATH+'checkpoints_'+EXPT_NAME+'/'
        if not os.path.exists(checkpoint_loc):
            os.makedirs(checkpoint_loc)
        #first save a copy of the args settings file for this experiment
        import shutil
        shutil.copy('./settings.txt', checkpoint_loc+'settings.txt')

        #save checkpoint at init
        savecount = 0
        fname_chk = lambda id: checkpoint_loc+ 'model_'+str(id)+'epochs.pt'
        torch.save({'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'lr_scheduler':scheduler.state_dict()},fname_chk(savecount))
        savecount+=1
        if args.resample_deadneurons:
            track_epochs = [2**i-1 for i in range(math.floor(math.log2(EPOCHSnum))+1)] + \
                [EPOCHSnum-1] + [int(EPOCHS/2)+i for i in range(-1, 6)]
            track_epochs = sorted(list(set(track_epochs)))
        else:
            track_epochs = [2**i-1 for i in range(math.floor(math.log2(EPOCHSnum))+1)] + [EPOCHSnum-1]
    torch.manual_seed(0)
    tic = time.perf_counter()

    if not args.online_training:
        for t in range(EPOCHS):
            update_status(f"Epoch {t+1}\n-------------------------------")

            train_loss_epoch[t] = train(train_dataloader, model, optimizer, \
                                            update_status_fn=update_status, regularizer=args.regularizer, \
                                                encoder_reg=args.encoder_reg, gamma_reg=args.gamma_reg, \
                                                    return_concept_loss=args.return_concept_loss, \
                                                        num_concepts=args.num_concepts, clip_grad=args.clip_grad)
                
            test_loss_epoch[t] = test(test_dataloader, model, update_status_fn=update_status,\
                                        regularizer=args.regularizer, encoder_reg=args.encoder_reg, gamma_reg=args.gamma_reg, \
                                            return_concept_loss=args.return_concept_loss, num_concepts=args.num_concepts)  
            lambda_vals[t] = model.lambda_val.data
            if args.save_checkpoints:
                if t in track_epochs:
                    torch.save({'model':model.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'lr_scheduler':scheduler.state_dict()},fname_chk(t+1))
                    savecount+=1
            if args.resample_deadneurons:
                if t==int(EPOCHS/2)-1:
                    with torch.no_grad():
                        resample_deadlatents(model, train_dataloader, num_batches=15)
                    if args.save_checkpoints:
                            torch.save({'model':model.state_dict(),
                                    'optimizer':optimizer.state_dict(),
                                    'lr_scheduler':scheduler.state_dict()}, checkpoint_loc+ 'model_'+str(t+1.5)+'epochs.pt')
                            savecount+=1
            scheduler.step()
            logdata = {"loss_train_mse":train_loss_epoch[t][0], "loss_train_reg":train_loss_epoch[t][1],\
                        "loss_test_mse":test_loss_epoch[t][0], "loss_test_reg":test_loss_epoch[t][1], \
                        "loss_train": train_loss_epoch[t][0]+train_loss_epoch[t][1],\
                        "loss_test": test_loss_epoch[t][0]+test_loss_epoch[t][1], \
                            "lambda": lambda_vals[t]}
            
            if args.return_concept_loss:
                log_concept_loss = {f"c{i}" + "_loss_train_mse":train_loss_epoch[t][-1][i] for i in range(len(train_loss_epoch[t][-1]))}
                logdata.update(log_concept_loss)
                log_concept_test_loss = {f"c{i}" + "_loss_test_mse":test_loss_epoch[t][-1][i] for i in range(len(test_loss_epoch[t][-1]))}
                logdata.update(log_concept_test_loss)
            wandb.log(logdata)

    else: #online training, go through entire data only once 
        from torch.utils.data import DataLoader, TensorDataset

        g_tr = torch.Generator()
        g_tr.manual_seed(0)
        for t, (batchdata, batchlabels) in enumerate(train_dataloader):
            if t<EPOCHSnum:
                update_status(f"Epoch {t+1}\n-------------------------------")
                batchdataloader = DataLoader(TensorDataset(batchdata, batchlabels), batch_size=args.batch_size, generator=g_tr)
                train_loss_epoch[t] = train(batchdataloader, model, optimizer, \
                                                update_status_fn=update_status, \
                                                    regularizer=args.regularizer, encoder_reg=args.encoder_reg, \
                                                        gamma_reg=args.gamma_reg, \
                                                            return_concept_loss=args.return_concept_loss, \
                                                                num_concepts=args.num_concepts, clip_grad=args.clip_grad)
                lambda_vals[t] = model.lambda_val.data
                scheduler.step()
                logdata = {"loss_train_mse":train_loss_epoch[t][0], "loss_train_reg":train_loss_epoch[t][1],\
                                "loss_train": train_loss_epoch[t][0]+train_loss_epoch[t][1],\
                                    "lambda": lambda_vals[t]}
                if args.return_concept_loss:
                    log_concept_loss = {f"c{i}" + "_loss_train_mse":train_loss_epoch[t][-1][i] for i in range(len(train_loss_epoch[t][-1]))}
                    logdata.update(log_concept_loss)
                wandb.log(logdata)
                if args.save_checkpoints:
                    if t in track_epochs:
                        torch.save({'model':model.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'lr_scheduler':scheduler.state_dict()},fname_chk(t+1))
                        savecount+=1
                if args.resample_deadneurons:
                    numbatches = len(train_dataloader)
                    if t%(numbatches//3) == (numbatches//3)-1:
                        with torch.no_grad():
                            resample_deadlatents(model, train_dataloader, num_batches=15)
                        if args.save_checkpoints:
                                torch.save({'model':model.state_dict(),
                                        'optimizer':optimizer.state_dict(),
                                        'lr_scheduler':scheduler.state_dict()}, checkpoint_loc+ 'model_'+str(t+1.5)+'epochs.pt')
                                savecount+=1
            else:
                break
    toc = time.perf_counter()

    update_status("Done!")
    update_status(f"Time to train {EPOCHS} epochs = {round(toc-tic,2)}s ({round((toc-tic)/EPOCHS,2)}s per epoch)")

    #save losses
    if args.online_training:
        torch.save({'train_loss':train_loss_epoch, 'lambda':lambda_vals}, checkpoint_loc+'losses.pt')
    else:
        torch.save({'train_loss':train_loss_epoch, 'test_loss':test_loss_epoch, 'lambda':lambda_vals}, checkpoint_loc+'losses.pt')

    wandb.finish()