"""Main entrance for train/eval with/without KD"""

import argparse
import logging
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm
import utils
import dataloader as data_loader
import resnet as R
from evaluate import evaluate, evaluate_kd
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/base_resnet18',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'last'


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=True), \
                    labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            # if i % params.save_summary_steps == 0:
            #     # extract data from torch Variable, move to cpu, convert to numpy arrays
            #     output_batch = output_batch.data.cpu().numpy()
            #     labels_batch = labels_batch.data.cpu().numpy()

            #     # compute all metrics on this batch
            #     summary_batch = {metric: metrics[metric](output_batch, labels_batch)
            #                      for metric in metrics}
            #     summary_batch['loss'] = loss.data.cpu().numpy()
            #     summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    # metrics_mean = {metric: np.mean([x[metric]
    #                                 for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
    #                             for k, v in metrics_mean.items())
    # logging.info("- Train metrics: " + metrics_string)

    # return metrics_mean
    return None


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers
    scheduler = StepLR(optimizer, step_size=params.scheduler_step_size, gamma=0.1)

    # Tensorboard logger
    # tb_path = (os.path.join(model_dir, 'tb_logs'))
    # writer = SummaryWriter(tb_path)
    # writer.add_text('hyperparameters', params.get_content())

    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("\nEpoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        epoch_metrics = train(model, optimizer, loss_fn,
                              train_dataloader, metrics, params)
        scheduler.step()

        # Evaluate for one epoch on validation set
        logging.info("Evaluating on validation set")
        # val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        # val_acc = val_metrics['accuracy']
        # is_best = val_acc >= best_val_acc

        # # Save weights
        # utils.save_checkpoint({'epoch': epoch + 1,
        #                        'state_dict': model.state_dict(),
        #                        'optim_dict': optimizer.state_dict()},
        #                       is_best=is_best,
        #                       checkpoint=model_dir)

        # If best_eval, best_save_path
        # if is_best:
        #     logging.info("- Found new best accuracy")
        #     best_val_acc = val_acc

        #     # Save best val metrics in a json file in the model directory
        #     best_json_path = os.path.join(
        #         model_dir, "metrics_val_best_weights.json")
        #     utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(
        #     model_dir, "metrics_val_last_weights.json")
        # utils.save_dict_to_json(val_metrics, last_json_path)

        # append to epoch_metrics
        # epoch_metrics['val_accuracy'] = val_metrics['accuracy']
        # epoch_metrics['val_loss'] = val_metrics['loss']

        # write to tensorboard
        # for tag, value in epoch_metrics.items():
        #     writer.add_scalar(tag, value, epoch+1)
    writer.flush()
    writer.close()


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    total_loss_avg = utils.RunningAverage()
    reg_loss_avg = utils.RunningAverage()
    kd_loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=True), \
                    labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output, fetch teacher output, and compute KD loss
            output_batch = model(train_batch)

            # get one batch output from teacher_outputs list
            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
            if params.cuda:
                output_teacher_batch = output_teacher_batch.cuda(
                    non_blocking=True)
            
            kd_loss, reg_loss, total_loss = loss_fn_kd(train_batch ,output_batch, labels_batch,
                              output_teacher_batch, params, teacher_model, model)()

            # clear previous gradients, 
            optimizer.zero_grad()
            
            # compute gradients of all variables wrt loss
            total_loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['kd_loss'] = kd_loss.data.cpu().numpy()
                summary_batch['reg_loss'] = reg_loss.data.cpu().numpy()
                summary_batch['loss'] = total_loss.data.cpu().numpy()
                summ.append(summary_batch)

            # update the average loss
            total_loss_avg.update(total_loss.data)
            reg_loss_avg.update(reg_loss.data)
            kd_loss_avg.update(kd_loss.data)

            t.set_postfix(total_loss='{:05.3f}'.format(total_loss_avg()), 
                          reg_loss='{:05.3f}'.format(reg_loss_avg()),
                          kd_loss='{:05.3f}'.format(kd_loss_avg())
                         )
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                    for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    del kd_loss, reg_loss, total_loss, output_batch, labels_batch, train_batch
    torch.cuda.empty_cache()
    return metrics_mean


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          loss_fn_kd, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers for different models:
    scheduler = StepLR(optimizer, step_size=params.scheduler_step_size, gamma=0.1)

    # Tensorboard logger
    tb_path = (os.path.join(model_dir, 'tb_logs'))
    writer = SummaryWriter(tb_path)
    writer.add_text('hyperparameters', params.get_content())

    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("\nEpoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        epoch_metrics = train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader,
                                 metrics, params)
        scheduler.step()

        # Evaluate for one epoch on validation set
        logging.info("Evaluating on validation set")
        val_metrics = evaluate_kd(
            model, teacher_model, loss_fn_kd, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # append to epoch_metrics
        epoch_metrics['val_accuracy'] = val_metrics['accuracy']

        # Not calculating val_loss to save time
        # epoch_metrics['val_kd_loss'] = val_metrics['kd_loss']
        # epoch_metrics['val_reg_loss'] = val_metrics['reg_loss']
        # epoch_metrics['val_total_loss'] = val_metrics['total_loss']

        # write to tensorboard
        for tag, value in epoch_metrics.items():
            writer.add_scalar(tag, value, epoch+1)

    writer.flush()
    writer.close()

def parse_params():
    # Load the parameters from json file
    args = parser.parse_args()
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()

    return params, args

def load_train_objs(params, rank, world_size):
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not params.cuda:
        logging.info("GPU not available. Proceeding with CPU")

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dl, dev_dl = data_loader.get_train_valid_loader(
        data_dir=params.data_dir,
        batch_size=params.batch_size,
        augment=params.augment,
        random_seed=42,
        rank=rank,
        world_size=world_size,
        valid_size=params.valid_size,
        shuffle=True,
        show_sample=False,
        num_workers=params.num_workers,
        pin_memory=params.cuda)

    logging.info("Dataloading done.")

    # Get Model
    model = R.Resnet(model_name=params.model_version,
                          pretrained=params.pretrained)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                          momentum=0.9, weight_decay=5e-4)

    return train_dl, dev_dl, model, optimizer

def run(params, device, model, train_dl, dev_dl, optimizer, metrics, args):
    if params.distillation:
        logging.info("Using Distillation")

        # Specify the pre-trained teacher models for knowledge distillation
        if params.teacher != "none":
            teacher_model = R.Resnet(model_name=params.teacher,
                                          pretrained=False)
            teacher_model = teacher_model.to(device)                             
            teacher_checkpoint = params.teacher_checkpoint
            utils.load_checkpoint(teacher_checkpoint, teacher_model)
        else:
            raise AssertionError("Teacher model not found in params")

        # fetch loss function and metrics definition in model files
        #loss_fn_kd = R.loss_fn_kd
        loss_fn_kd = R.KD_loss

        # Train the model with KD
        logging.info(
            "Experiment - model version: {}".format(params.model_version))
        logging.info(
            "Starting KD training for {} epoch(s)".format(params.num_epochs))
        logging.info(
            "First, loading the teacher model and computing its outputs...")
        
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                              metrics, params, args.model_dir, args.restore_file)

    # non-KD mode: regular training of the baseline ResNet-18/50 models
    else:
        logging.info(
            "Using non-KD mode: regular training of the baseline ResNet-18/50 models")

        # fetch loss function and metrics
        loss_fn = R.loss_fn

        # Train the model
        logging.info(
            "Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model, train_dl, dev_dl, optimizer,
                           loss_fn, metrics, params,
                           args.model_dir, args.restore_file)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int):
    logging.info("Setting up ddp")
    ddp_setup(rank, world_size)
    params, args = parse_params()

    # Set random seed
    random.seed(230)
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    logging.info("Loading objects")
    train_dl, dev_dl, model, optimizer = load_train_objs(params, rank, world_size)

    metrics = R.metrics 
    run(params, rank, model, train_dl, dev_dl, optimizer, metrics, args)

    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    
