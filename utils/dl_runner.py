import torch
from torchtext import data
import os
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

logger = logging.getLogger('utils/dl_runner.py')

stats_columns = '{0:>5}|{1:>5}|{2:>5}|{3:>5}|{4:>5}|{5:>5}|{6:>5}|{7:>5}|{8:>5}|{9:>5}|{10:>5}'


def train(train_iter, dev_iter, model, optimizer, loss_criterion, args):
    best_dev_f1 = -1

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

            train_loss += loss.item()

            # backpropagate and update optimizer learning rate
            loss.backward()
            optimizer.step()

        train_loss = train_loss / n_total_steps

        train_acc, train_f1, train_recall, train_prec = calculate_metrics(_label, _pred)

        _dev_label, _dev_pred, dev_loss = eval(dev_iter, model, loss_criterion, args)

        dev_acc, dev_f1, dev_recall, dev_prec = calculate_metrics(_dev_label, _dev_pred)

        logging.info(
            stats_columns.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss, dev_acc, dev_f1,
                                 dev_recall, dev_prec, dev_loss))

        if best_dev_f1 < dev_f1:
            logging.debug('New dev acc {dev_acc} is larger than best dev acc {best_dev_acc}'.format(dev_acc=dev_f1,
                                                                                                    best_dev_acc=best_dev_f1))
            best_dev_f1 = dev_f1

            training_mode = args.training_mode
            model_name = '{training_mode}_epoch_{epoch}_dev_f1_{dev_f1:03}.pth.tar'.format(training_mode=training_mode,
                                                                                           epoch=epoch,
                                                                                           dev_f1=dev_f1)
            save_model(model, optimizer, epoch, model_name, training_mode, args.checkpoint_dir)


def calculate_metrics(label, pred):
    logging.debug('Expected: \n{}'.format(label))
    logging.debug('Predicted: \n{}'.format(pred))
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
        with torch.no_grad():
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


def train_kfold(model, train_data, text_field, label_field, optimizer, loss_criterion, args):
    logging.info('Mode: kfold training')
    kfold_range = train_data.kfold(args.kfold)
    accuracys = []
    avg_losses = []

    logging.info('Number of train samples {}'.format(len(train_data)))

    for kfold_i, (train_range, test_range) in enumerate(kfold_range):
        torch.cuda.empty_cache()

        logging.info('Training {} th fold'.format(kfold_i + 1))

        train_data_k, dev_data_k = train_data.get_fold(fields=[('text', text_field), ('label', label_field)],
                                                       train_indexs=train_range,
                                                       test_indexs=test_range)

        train_iter, dev_iter = data.BucketIterator.splits((train_data_k, dev_data_k), device=args.device,
                                                          batch_sizes=(args.batch_size, args.batch_size),
                                                          sort_key=lambda x: len(x.text), shuffle=True)

        logging.info(
            "Number of training samples {train}, number of dev samples {dev} in fold {kfold}".format(
                train=len(train_iter),
                dev=len(dev_iter),
                kfold=kfold_i + 1))

        model.to(args.device)
        train(train_iter, dev_iter, model, optimizer, loss_criterion, args)


def train_split(model, train_data, text_field, label_field, optimizer, loss_criterion, args):
    logging.info('Mode: random split training')
    train_data, valid_data = train_data.splits(fields=[('text', text_field), ('label', label_field)])
    train_iter, dev_iter = data.BucketIterator.splits((train_data, valid_data), device=args.device,
                                                      batch_sizes=(args.batch_size, args.batch_size),
                                                      sort_key=lambda x: len(x.text), shuffle=True)
    torch.cuda.empty_cache()

    logging.info(
        "Number of training samples {train}, number of dev samples {dev} in random split".format(
            train=len(train_iter),
            dev=len(dev_iter)))

    model.to(args.device)
    train(train_iter, dev_iter, model, optimizer, loss_criterion, args)


def save_model(model, optimizer, epoch, model_name, training_mode, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_path = os.path.join(checkpoint_dir, model_name)

    torch.save({
        'epoch': epoch,
        'training_mode': training_mode,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path)

    logging.info('Best model in {training_mode} is saved to {save_path}'.format(training_mode=training_mode,
                                                                                save_path=save_path))
    return save_path


def load_model(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info('Loaded checkpoint from path "{}" (at epoch {}) in training mode {}'
                 .format(checkpoint_path, checkpoint['epoch']), checkpoint['training_mode'])
