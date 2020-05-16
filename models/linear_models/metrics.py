from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def calculate_metrics(label, pred):
    acc = round(accuracy_score(label, pred), 2)
    f1 = round(f1_score(label, pred, average='binary'), 2)
    recall = round(recall_score(label, pred), 2)
    prec = round(precision_score(label, pred), 2)
    return acc, f1, recall, prec
