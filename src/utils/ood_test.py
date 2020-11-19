import torch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def test_ood(exp, idd, odd):
    idd_dataloader = idd()
    idd_dataloader.prepare_data()
    idd_dataloader.setup()

    odd_dataloader = odd()
    odd_dataloader.prepare_data()
    odd_dataloader.setup()

    idd_test = idd_dataloader.test_dataloader()
    odd_test = odd_dataloader.test_dataloader()

    idd_result = exp.trainer.test(exp.model, test_dataloaders=[
                                  idd_test], verbose=False)
    odd_result = exp.trainer.test(exp.model, test_dataloaders=[
                                  odd_test], verbose=False)

    idd_results = torch.Tensor(idd_result[1]['loss']).numpy()
    odd_results = torch.Tensor(odd_result[1]['loss']).numpy()

    targets = np.concatenate(
        (np.zeros(len(idd_results)), np.ones(len(odd_results))))
    results = np.concatenate((idd_results, odd_results))

    return (targets, results)


def plot_roc_auc(targets, probs):
    auc = roc_auc_score(targets, probs)
    print(' ROC AUC=%.5f' % (auc))

    fpr, tpr, _ = roc_curve(targets, probs)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
