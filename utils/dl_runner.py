import sys
import os
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, optimizer, loss_criterion, args):
    best_dev_acc = -1

    for epoch in range(args.epochs):

        model.train()
        train_iter.init_epoch()

        for batch_idx, batch in enumerate(train_iter):
            texts = batch.text
            labels = batch.label

            texts.to(args.device)
            labels.to(args.device)

            optimizer.zero_grad()

            # forward pass
            predictions = model(texts)

            # calculate loss of the network output with respect to training labels
            loss = loss_criterion(predictions, labels)

            # backpropagate and update optimizer learning rate
            loss.backward()
            optimizer.step()


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
