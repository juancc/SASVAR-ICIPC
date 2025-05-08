"""
Model evaluation functions

JCA
"""
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def show_results(id_labels, trues, preds, r=3):
    # Calc F1
    f1 = f1_score(trues, preds, average='macro', zero_division=0.0)
    print(f'Model score: {round(f1, 2)}')

    # SHOW RESULTS
    conf =  confusion_matrix(trues, preds, labels=list(id_labels.keys()), normalize='true')
    conf = np.round(conf, r)

    conf_matrix_fig = plt.figure(figsize = (13,10))
    sn.set(font_scale=1.2)
    sn.heatmap(conf, annot=True,  xticklabels=id_labels.values(), yticklabels=id_labels.values())

    plt.xlabel('Pred', fontsize=18)
    plt.ylabel('True', fontsize=18)
    # plt.show()

    target_names = list(id_labels.values())
    labels = list(id_labels.keys())
    report = classification_report(trues, preds, target_names=target_names, output_dict=False, labels=labels, zero_division=0.0)
    print(report)

    return conf_matrix_fig, report, f1