import datetime
import numpy as np
import torch
import random
import seaborn
import os
import dgl
from torch import nn
from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score, r2_score, \
    mean_squared_error
import matplotlib.pyplot as plt


def collate(samples):
    smiles, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch([dgl.add_self_loop(g) for g in graphs])
    return smiles, batched_graph, labels


def get_loss(args):
    if args.task_type == 'classification':
        criterion = clf_loss(args.alpha)
        # criterion = nn.BCEWithLogitsLoss()
        args.metric = ['AUC', 'AUPR']
    elif args.task_type == 'regression':
        criterion = reg_loss(args.alpha)
        # criterion = nn.MSELoss()
        args.metric = ['MSE', 'R2']
    return criterion


class reg_loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.bag_loss = nn.MSELoss()
        self.ins_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, bag, ins, label):
        bag_loss = self.bag_loss(bag, label)
        ins_label = label.unsqueeze(dim=-1).expand((len(label), ins.shape[1]))
        ins_loss = self.ins_loss(ins, ins_label)
        return (1 - self.alpha) * bag_loss + self.alpha * ins_loss


class clf_loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.bag_loss = nn.BCEWithLogitsLoss()
        self.ins_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, bag, ins, label):
        bag_loss = self.bag_loss(bag, label)
        ins_label = label.unsqueeze(dim=-1).expand((len(label), ins.shape[1]))
        ins_loss = self.ins_loss(ins, ins_label)
        return (1 - self.alpha) * bag_loss + self.alpha * ins_loss


def get_metrics(y_label, y_pred, metric):
    if 'AUC' in metric:
        AUC = roc_auc_score(y_label, y_pred)
        AUPR = average_precision_score(y_label, y_pred)
        return AUC, AUPR
    else:
        MSE = mean_squared_error(y_label, y_pred)
        R2 = r2_score(y_label, y_pred)
        return MSE, R2


def get_metrics_cls(real_score, predict_score):
    """Calculate the performance metrics.
    Resource code is acquired from:
    Yu Z, Huang F, Zhao X et al.
     Predicting drug-disease associations through layer attention graph convolutional network,
     Brief Bioinform 2021;22.
    Parameters
    ----------
    real_score: true labels
    predict_score: model predictions
    Return
    ---------
    AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity
    """
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc[0, 0], aupr[0, 0], accuracy, f1_score, precision, recall, specificity


def set_seed(seed=0):
    print('seed = {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    # dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def checkpoint(args, model, check_list, metric, fold=None):
    saved = False
    loss_list = check_list['loss']
    if args.k == 0:
        metric_1_list = check_list['val_metric_1']
        metric_2_list = check_list['val_metric_2']
    else:
        metric_1_list = check_list['metric_1']
        metric_2_list = check_list['metric_2']
    if args.check_metric == 'loss':
        if loss_list[-1] == min(loss_list):
            saved = True
    elif args.check_metric == 'AUC':
        if metric_1_list[-1] >= max(metric_1_list):
            saved = True
    elif args.check_metric == 'AUPR' or args.check_metric == 'R2':
        if metric_2_list[-1] >= max(metric_2_list):
            saved = True
    elif args.check_metric == 'MSE':
        if metric_1_list[-1] <= min(metric_1_list):
            saved = True
    elif args.check_metric == 'Comprehensive':
        if args.task == 'regression':
            change_1 = (min(metric_1_list[-1]) - metric_1_list[-1]) / min(metric_1_list[-1])
        change_2 = (metric_1_list[-1] - max(metric_1_list[-1])) / max(metric_1_list[-1])
        if change_1 + change_2 >= 0:
            saved = True
    if saved:
        print('save best model')
        if fold:
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, 'model_{}.pkl'.format(fold)))
        else:
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, 'model.pkl'))
        return model


class EarlyStopping(object):
    def __init__(self, patience=10, saved_path='.'):
        dt = datetime.datetime.now()
        self.filename = os.path.join(saved_path, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second))
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))


def plot_result_auc(args, label, predict, auc):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    auc: calculated AUROC score
    """
    seaborn.set_style()
    fpr, tpr, threshold = roc_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    if args.fold:
        plt.savefig(os.path.join(args.save_path, 'result_auc_fold{}.png'.format(args.fold)))
    else:
        plt.savefig(os.path.join(args.save_path, 'result_auc.png'))
    plt.clf()


def plot_result_aupr(args, label, predict, aupr):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    aupr: calculated AUPR score
    """
    seaborn.set_style()
    precision, recall, thresholds = precision_recall_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(precision, recall, color='darkorange',
             lw=lw, label='AUPR Score (area = %0.4f)' % aupr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('RPrecision/Recall Curve')
    plt.legend(loc='lower right')
    if args.fold:
        plt.savefig(os.path.join(args.save_path, 'result_aupr_fold{}.png'.format(args.fold)))
    else:
        plt.savefig(os.path.join(args.save_path, 'result_aupr.png'))
    plt.clf()


def plot_training_curve(args, score):
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(score['epoch'], score['loss'], color='darkorange',
             lw=lw, label='Minimum Loss = {:.3f}'.format(np.min(score['loss'])))
    min_idx = np.where(np.array(score['loss']) == np.min(score['loss']))[0][0]
    plt.plot([score['epoch'][min_idx], score['epoch'][min_idx]], [0, 1])
    # plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss values in the training process')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.save_path, 'loss.png'))
    plt.clf()

    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    if args.metric[0] == 'AUC':
        plt.plot(score['epoch'], score['metric_1'], color='darkorange',
                 lw=lw, label='Maximum Training AUROC = {:.3f}'.format(np.max(score['metric_1'])))
        max_idx = np.where(np.array(score['metric_1']) == np.max(score['metric_1']))[0][0]
        plt.plot([score['epoch'][max_idx], score['epoch'][max_idx]], [0, 1], color='darkorange')

        if score['val_metric_1'] != []:
            plt.plot(score['epoch'], score['val_metric_1'], color='navy',
                     lw=lw, label='Maximum Validating AUROC = {:.3f}'.format(np.max(score['val_metric_1'])))
            max_idx = np.where(np.array(score['val_metric_1']) == np.max(score['val_metric_1']))[0][0]
            plt.plot([score['epoch'][max_idx], score['epoch'][max_idx]], [0, 1], color='navy')

        plt.ylim([0.0, 1.0])
        plt.ylabel('AUROC')
        plt.legend(loc='lower right')
        plt.title('AUROC values in the training process')
        if args.k != 0:
            plt.savefig(os.path.join(args.save_path, 'auc_curve_fold{}.png'.format(args.fold)))
        else:
            plt.savefig(os.path.join(args.save_path, 'auc_curve.png'))

    elif args.metric[0] == 'MSE':
        plt.plot(score['epoch'], score['metric_1'], color='darkorange',
                 lw=lw, label='Minimum Training MSE = {:.3f}'.format(np.min(score['metric_1'])))
        min_idx = np.where(np.array(score['metric_1']) == np.min(score['metric_1']))[0][0]
        plt.plot([score['epoch'][min_idx], score['epoch'][min_idx]], [0, 10], color='darkorange')

        if score['val_metric_1'] != []:
            plt.plot(score['epoch'], score['val_metric_1'], color='navy',
                     lw=lw, label='Minimum Validating MSE = {:.3f}'.format(np.min(score['val_metric_1'])))
            min_idx = np.where(np.array(score['val_metric_1']) == np.min(score['val_metric_1']))[0][0]
            plt.plot([score['epoch'][min_idx], score['epoch'][min_idx]], [0, 10], color='navy')

        plt.ylim([0.0, 10.0])
        plt.ylabel('MSE')
        plt.legend(loc='lower right')
        plt.title('MSE values in the training process')
        if args.k != 0:
            plt.savefig(os.path.join(args.save_path, 'mse_curve_fold{}.png'.format(args.fold)))
        else:
            plt.savefig(os.path.join(args.save_path, 'mse_curve.png'))
    # plt.xlim([0.0, 1.0])
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.clf()

    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    if args.metric[0] == 'AUC':
        plt.plot(score['epoch'], score['metric_2'], color='darkorange',
                 lw=lw, label='Maximum Training AUPR = {:.3f}'.format(np.max(score['metric_2'])))
        max_idx = np.where(np.array(score['metric_2']) == np.max(score['metric_2']))[0][0]
        plt.plot([score['epoch'][max_idx], score['epoch'][max_idx]], [0, 1], color='darkorange')

        if score['val_metric_2'] != []:
            plt.plot(score['epoch'], score['val_metric_2'], color='navy',
                     lw=lw, label='Maximum Validating AUPR = {:.3f}'.format(np.max(score['val_metric_2'])))
            max_idx = np.where(np.array(score['val_metric_2']) == np.max(score['val_metric_2']))[0][0]
            plt.plot([score['epoch'][max_idx], score['epoch'][max_idx]], [0, 1], color='navy')

        plt.ylim([0.0, 1.0])
        plt.ylabel('AUPR')
        plt.legend(loc='lower right')
        plt.title('AUPR values in the training process')
        if args.k != 0:
            plt.savefig(os.path.join(args.save_path, 'aupr_curve_fold{}.png'.format(args.fold)))
        else:
            plt.savefig(os.path.join(args.save_path, 'aupr_curve.png'))

    elif args.metric[0] == 'MSE':
        plt.plot(score['epoch'], score['metric_2'], color='darkorange',
                 lw=lw, label='Maximum Training R2 = {:.3f}'.format(np.max(score['metric_2'])))
        max_idx = np.where(np.array(score['metric_2']) == np.max(score['metric_2']))[0][0]
        plt.plot([score['epoch'][max_idx], score['epoch'][max_idx]], [0, 1], color='darkorange')

        if score['val_metric_2'] != []:
            plt.plot(score['epoch'], score['val_metric_2'], color='navy',
                     lw=lw, label='Maximum Validating R2 = {:.3f}'.format(np.max(score['val_metric_2'])))
            max_idx = np.where(np.array(score['val_metric_2']) == np.max(score['val_metric_2']))[0][0]
            plt.plot([score['epoch'][max_idx], score['epoch'][max_idx]], [0, 1], color='navy')

        plt.ylim([0.0, 1.0])
        plt.ylabel('R2')
        plt.legend(loc='lower right')
        plt.title('R2 values in the training process')
        if args.k != 0:
            plt.savefig(os.path.join(args.save_path, 'r2_curve_fold{}.png'.format(args.fold)))
        else:
            plt.savefig(os.path.join(args.save_path, 'r2_curve.png'))
    # plt.xlim([0.0, 1.0])
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.clf()
