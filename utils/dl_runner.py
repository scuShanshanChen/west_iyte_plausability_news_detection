import torch
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

logger = logging.getLogger('utils/dl_runner.py')

stats_columns = '{0:>10} | {1:>10} | {2:>10} | {3:>10} |{4:>10} | {5:>10} | {6:>10} | {7:>10} | {8:>10} | {9:>10} | {10:>10}'


def train(train_iter, dev_iter, model, optimizer, loss_criterion, args):
    best_dev_acc = -1

    n_total_steps = len(train_iter)

    logging.info(
        stats_columns.format(
            'Epoch', 'T-Acc', 'T-F1', 'T-Recall', 'T-Prec', 'T-Loss'
            , 'D-Acc', 'D-F1', 'D-Recall', 'D-Prec', 'D-Loss'))

    for epoch in range(args.epochs):

        model.train()
        train_iter.init_epoch()

        train_loss = 0
        preds = []
        trues = []

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

            # record preds, trues
            _pred = torch.round(torch.sigmoid(predictions)).cpu().data.numpy()
            preds.append(_pred)

            _label = labels.cpu().data.numpy()
            trues.append(_label)

            # backpropagate and update optimizer learning rate
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / n_total_steps

        train_acc, train_f1, train_recall, train_prec = calculate_metrics(_label, _pred)

        _dev_label, _dev_pred, dev_loss = eval(dev_iter, model, loss_criterion, args)

        dev_acc, dev_f1, dev_recall, dev_prec = calculate_metrics(_dev_label, _dev_pred)

        logging.info(
            stats_columns.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss, dev_acc, dev_f1,
                                 dev_recall, dev_prec, dev_loss))

        if best_dev_acc < dev_acc:
            logging.debug('New dev acc {dev_acc} is larger than best dev acc {best_dev_acc}'.format(dev_acc=dev_acc,
                                                                                                    best_dev_acc=best_dev_acc))
            best_dev_acc = dev_acc


def calculate_metrics(label, pred):
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average='binary')
    recall = recall_score(label, pred)
    prec = precision_score(label, pred)
    return acc, f1, recall, prec


def eval(dev_iter, model, loss_criterion, args):
    n_total_steps = len(dev_iter)
    model.eval()
    dev_loss = 0
    preds = []
    trues = []
    for batch_idx, batch in enumerate(dev_iter):
        texts = batch.text
        labels = batch.label

        texts.to(args.device)
        labels.to(args.device)

        # forward pass
        predictions = model(texts)

        # calculate loss of the network output with respect to training labels
        loss = loss_criterion(predictions, labels)
        dev_loss += loss.item()

        # record preds, trues
        _pred = torch.round(torch.sigmoid(predictions)).cpu().data.numpy()
        preds.append(_pred)

        _label = labels.cpu().data.numpy()
        trues.append(_label)

    dev_loss = dev_loss / n_total_steps
    return _label, _pred, dev_loss
