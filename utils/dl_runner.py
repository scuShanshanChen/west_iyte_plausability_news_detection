import logging
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger('utils/dl_runner.py')

stats_columns = '{0:>5}|{1:>5}|{2:>5}|{3:>5}|{4:>5}|{5:>5}|{6:>5}|{7:>5}|{8:>5}|{9:>5}|{10:>5}'

stats_template = 'Epoch {epoch_idx}\n' \
                 '{mode} Accuracy: {acc}\n' \
                 '{mode} F1: {f1}\n' \
                 '{mode} Recall: {recall}\n' \
                 '{mode} Precision: {prec}\n' \
                 '{mode} Loss: {loss}\n'


def train(num_epochs, model, train_data, dev_data, optimizer, criterion, device, checkpoint_dir, clip):
    best_dev_f1 = 0
    best_eval_loss = np.inf
    n_total_steps = len(train_data)
    total_iter = n_total_steps * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_iter)
    for i in range(num_epochs):
        best_dev_f1, best_eval_loss = train_epoch(i, model, train_data, dev_data, optimizer, criterion, device,
                                                  best_dev_f1,
                                                  checkpoint_dir, clip, scheduler, best_eval_loss)


def train_epoch(epoch_idx, model, train_data, dev_data, optimizer, criterion, device, best_dev_f1, model_name, clip,
                scheduler, best_eval_loss):
    model.to(device)
    model.train()

    logger.info('Training epoch: {}'.format(epoch_idx))
    train_loss, output, target = train_batch(criterion, model, optimizer, train_data, device, clip, scheduler)
    train_acc, train_f1, train_recall, train_prec = calculate_metrics(target, output)

    print(stats_template
          .format(mode='train', epoch_idx=epoch_idx, acc=train_acc, f1=train_f1, recall=train_recall,
                  prec=train_prec, loss=train_loss))

    eval_loss, output, target = eval_batch(dev_data, model, criterion, device)
    dev_acc, dev_f1, dev_recall, dev_prec = calculate_metrics(target, output)

    print(stats_template
          .format(mode='dev', epoch_idx=epoch_idx, acc=dev_acc, f1=dev_f1, recall=dev_recall,
                  prec=dev_prec, loss=eval_loss))

    if best_dev_f1 < dev_f1:
        logging.debug('New dev f1 {dev_f1} is larger than best dev f1 {best_dev_f1}'.format(dev_f1=dev_f1,
                                                                                            best_dev_f1=best_dev_f1))
        best_dev_f1 = dev_f1
        best_eval_loss = eval_loss
        save_model(model, optimizer, model_name)
    elif best_dev_f1 == dev_f1:
        if eval_loss <= best_eval_loss:
            logging.debug(
                'New dev f1 {dev_f1} is equal to best dev f1 {best_dev_f1} but its loss: {eval_loss} smaller than previous {best_eval_loss}'.format(
                    dev_f1=dev_f1,
                    best_dev_f1=best_dev_f1, eval_loss=eval_loss, best_eval_loss=best_eval_loss))
            best_dev_f1 = dev_f1
            best_eval_loss = eval_loss
            save_model(model, optimizer, model_name)

    return best_dev_f1, best_eval_loss


def train_batch(criterion, model, optimizer, train_data, device, clip, scheduler):
    train_loss = 0
    n_total_steps = len(train_data)
    labels = []
    predictions = []
    for batch in tqdm(train_data):
        optimizer.zero_grad()
        text = Variable(batch['text']).to(device)
        text_len = Variable(batch['text_len']).to(device)
        output = model(text, text_len)
        target = Variable(batch['label']).to(device)
        loss = criterion(output, target)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        output = torch.round(torch.sigmoid(output)).cpu().data.numpy()
        predictions.append(output)
        labels.append(target.cpu().data.numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    train_loss = train_loss / n_total_steps
    return train_loss, labels, predictions


def calculate_metrics(label, pred):
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average='binary')
    recall = recall_score(label, pred)
    prec = precision_score(label, pred)
    return acc, f1, recall, prec


def eval_batch(dev_data, model, criterion, device):
    eval_loss = 0
    n_total_steps = len(dev_data)

    model.eval()

    labels = []
    predictions = []
    for batch in tqdm(dev_data):
        with torch.no_grad():
            text = Variable(batch['text']).to(device)
            text_len = Variable(batch['text_len']).to(device)
            output = model(text, text_len)

        target = Variable(batch['label']).to(device)
        loss = criterion(output, target)
        eval_loss += loss.item()
        output = torch.round(torch.sigmoid(output)).cpu().data.numpy()
        predictions.append(output)
        labels.append(target.cpu().data.numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    eval_loss = eval_loss / n_total_steps
    return eval_loss, labels, predictions


def inference(test_data, model, device):
    model.eval()
    labels = []
    predictions = []
    for batch in tqdm(test_data):
        with torch.no_grad():
            text = Variable(batch['text']).to(device)
            text_len = Variable(batch['text_len']).to(device)
            output = model(text, text_len)
            output = torch.round(torch.sigmoid(output)).cpu().data.numpy()
            predictions.append(output)

        target = Variable(batch['label']).to(device)
        labels.append(target.cpu().data.numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    acc, f1, recall, prec = calculate_metrics(labels, predictions)
    print(stats_template
          .format(mode='test', epoch_idx='__', acc=acc, f1=f1, recall=recall,
                  prec=prec, loss='__'))


def save_model(model, optimizer, checkpoint_dir):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, checkpoint_dir)

    logging.info('Best model is saved to {save_path}'.format(save_path=checkpoint_dir))
    return checkpoint_dir


def load_model(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info('Loaded checkpoint from path {}'
                 .format(checkpoint_path))


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
