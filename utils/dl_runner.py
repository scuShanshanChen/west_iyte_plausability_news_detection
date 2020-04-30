import sys
import os
from tqdm import tqdm
import torch
import logging
import torch.autograd as autograd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

logger = logging.getLogger('utils/dl_runner.py')


def train(train_iter, dev_iter, model, optimizer, loss_criterion, writer, args):
    best_dev_acc = -1

    n_total_steps = len(train_iter)

    for epoch in range(args.epochs):

        model.train()
        train_iter.init_epoch()

        train_loss = 0
        preds = []
        trues = []

        for batch_idx, batch in tqdm(enumerate(train_iter)):
            texts = batch.text
            labels = batch.label

            texts.to(args.device)
            labels.to(args.device)

            optimizer.zero_grad()

            # forward pass
            predictions = model(texts)

            # calculate loss of the network output with respect to training labels
            loss = loss_criterion(predictions, labels)

            # record preds, trues
            _pred = torch.round(torch.sigmoid(predictions)).cpu().data.numpy()
            preds.append(_pred)

            _label = labels.cpu().data.numpy()
            trues.append(_label)

            # backpropagate and update optimizer learning rate
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        logging.debug('Train Acc {acc} in epoch {epoch}'.format(acc=accuracy_score(_label, _pred), epoch=epoch))
        logging.debug('Train Loss {loss} in epoch {epoch}'.format(loss=train_loss / n_total_steps, epoch=epoch))

        writer.add_scalar('Train Acc', accuracy_score(_label, _pred), epoch)
        writer.add_scalar('Train Loss', train_loss / n_total_steps, epoch)
        # add writer.flush()


def eval(data_iter, model, criterion, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.labels
        target.data.sub_(1)  # batch first, index align

        x = feature.data.numpy()
        x = x.T
        feature = autograd.Variable(torch.from_numpy(x))

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = criterion(logit, target)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0] / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
