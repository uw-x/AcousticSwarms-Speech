"""
The main training script for training on synthetic data
"""

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import speechbrain.nnet.schedulers as schedulers

import argparse
import importlib
import json
import os, glob
import re
import multiprocessing
import time

import sep.helpers.utils as utils
import numpy as np


VAL_SEED = 0
CURRENT_EPOCH = 0

def seed_from_epoch(seed):
    global CURRENT_EPOCH

    utils.seed_all(seed + CURRENT_EPOCH)

def print_metrics(metrics: list):
    input_sisdr = np.array([x['input_si_sdr'] for x in metrics])
    sisdr = np.array([x['si_sdr'] for x in metrics])

    print("Average Input SI-SDR: {:03f}, Average Output SI-SDR: {:03f}, Average SI-SDRi: {:03f}".format(np.mean(input_sisdr), np.mean(sisdr), np.mean(sisdr - input_sisdr)))


def train(args: argparse.Namespace):
    """
    Resolve the network to be trained
    """
    # Fix random seeds
    utils.seed_all(args.seed)

    # Turn on deterministic algorithms if specified (Note: slower training).
    if torch.cuda.is_available() and args.use_cuda:
        if args.use_nondeterministic_cudnn:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True

    # Load experiment description
    args.experiment_description = os.path.join(args.experiment_dir, 'description.json')
    with open(args.experiment_description, 'rb') as f:
        experiment_params = json.load(f)
    
    model_type = experiment_params['model_name']

    # Import training files for the given model type
    train = importlib.import_module(f'sep.training.{model_type}.train')
    dataset = importlib.import_module(f'sep.training.{model_type}.dataset')
    network = importlib.import_module(f'sep.training.{model_type}.network')
    
    # Get experiment parameters
    model_params = experiment_params['model_params']
    training_params = experiment_params['training_params']
    train_set_params = experiment_params['train_set_params']
    test_set_params = experiment_params['test_set_params']
    lr_sched_params = experiment_params['lr_sched_params']
   
    # Propagate experiment sampling rate to dataset params
    sr = experiment_params['sr']
    train_set_params['sr'] = sr
    test_set_params['sr'] = sr

    experiment_name = os.path.basename(args.experiment_dir.rstrip('/'))
    checkpoints_dir = os.path.join(args.experiment_dir, 'checkpoints')

    # Initialize datasets
    data_train = dataset.Dataset(dataset_type='train', **train_set_params)
    data_test = dataset.Dataset(dataset_type='test', **test_set_params)

    # Set up the device and workers
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using device {}".format('cuda' if use_cuda else 'cpu'))

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'worker_init_fn': lambda x: seed_from_epoch(args.seed),
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=training_params['batch_size'],
                                               shuffle=True,
                                               **kwargs)
   
    kwargs['worker_init_fn'] = lambda x: utils.seed_all(VAL_SEED)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=training_params['batch_size'],
                                              **kwargs)

    # Initialize train.test_epoch
    model = network.Network(**model_params)
    model.print_model_info()
    
    # Set up checkpoints
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # List all pretrained checkpoint for this experiment
    model_pretrain_name = os.path.join(checkpoints_dir, experiment_name + "_*.pt")
    paths = list(glob.glob(model_pretrain_name))
    model_nums = [int(re.search(\
        os.path.join(checkpoints_dir, experiment_name + "_([0-9]+).pt"),\
        path).group(1)) for path in paths]
    
    # If there is at least one checkpoint, start from the latest one
    if len(model_nums) > 0:
        start_epoch = max(model_nums) + 1
        pretrained_path = os.path.join(checkpoints_dir, \
                                       experiment_name + "_{}.pt".format(start_epoch - 1))
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
    # Otherwise, check if experiment starts from a pretrained model
    elif 'pretrain_path' in training_params:
        state_dict = torch.load(training_params['pretrain_path'])
        model.load_state_dict(state_dict)
        start_epoch = 0
    # Otherwise, start from the beginning
    else:
        start_epoch = 0

    # Set model loss
    model.set_loss(training_params['loss'])

    # Create DataParallel model
    model = nn.DataParallel(model)
    model.to(device)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=training_params['lr'])

    # Load current optimizer, scheduler and losses
    state_path = os.path.join(checkpoints_dir, 'state.pt')
    if os.path.exists(state_path):
        state = torch.load(state_path)
        train_losses = state['train_losses']
        val_losses = state['val_losses']

        optimizer.load_state_dict(state['optimizer'])
        scheduler = state['lr_sched']
    else:
        train_losses = []
        val_losses = []
        
        # Initialize scheduler
        scheduler = schedulers.ReduceLROnPlateau(lr_sched_params['lr_min'],
                                                 lr_sched_params['factor'],
                                                 lr_sched_params['patience'],
                                                 lr_sched_params['dont_halve_until_epoch'],
                                                 )    
    # Training loop
    try:        
        # Go over remaining epochs
        for epoch in range(start_epoch, training_params['epochs']):
            global CURRENT_EPOCH, VAL_SEED
            CURRENT_EPOCH = epoch
            seed_from_epoch(args.seed)

            print()
            print("=" * 25, "STARTING EPOCH", epoch, "=" * 25)
            print()

            print("[TRAINING]")
            
            # Run testing step
            
            t1 = time.time()
            train_loss = train.train_epoch(model, device, optimizer, train_loader, epoch, training_params, args.print_interval)
            t2 = time.time()
            print(f"Train epoch time: {t2 - t1:02f}s")

            print("\nTrain set: Average Loss: {:.4f}\n".format(train_loss))

            print()

            # Fix seed for all validation passes 
            # (needed since localization models will choose some random points, so we fix those)
            utils.seed_all(VAL_SEED)

            # Run testing step

            print("[TESTING]")
    
            test_loss, test_metrics = train.test_epoch(model, device, test_loader, sr, args.print_interval)
            
            print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))
            if test_metrics is not None:
                print_metrics(test_metrics)

            current_lr, next_lr = scheduler([optimizer], epoch, test_loss)
            schedulers.update_learning_rate(optimizer, next_lr)

            # if current_lr != next_lr:
            print("NEXT learning rate to: {:0.08f}".format(next_lr))
                    
            # Add and save losses as pkl file
            train_losses.append(train_loss)
            val_losses.append(test_loss)
                    
            # Save model params, optimizer, scheduler, losses
            torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, experiment_name + "_{}.pt".format(epoch)))
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'lr_sched': scheduler,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            torch.save(state, state_path)

            print()
            print("=" * 25, "FINISHED EPOCH", epoch, "=" * 25)
            print()

        return train_losses, val_losses
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experiment Params
    parser.add_argument('experiment_dir', type=str,
                        help='Path to experiment directory')
    
    # Operation Params
    parser.add_argument('--n_workers', type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")

    # Randomization Params
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_nondeterministic_cudnn',
                        action='store_true',
                        help="If using cuda, chooses whether or not to use \
                                non-deterministic cudDNN algorithms. Training will be\
                                faster, but the final results may differ slighty.")
    
    # Logging Params
    parser.add_argument('--print_interval', type=int, default=20,
                        help="Logging interval")
    train(parser.parse_args())
