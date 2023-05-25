"""Constants and utility functions for NSP

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import typing

from sklearn import metrics


CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"


def load_data(path: str):
    sentences = [["Já jsem Martin a včera jsem šel do obchodu.", "Venku bylo hezké počasí.", 0],
                 ["Umělá inteligence se stále zlepšuje.", "Nikdy však nebude dost chytrá.", 0],
                 ["Strážci galaxie byl skvělý film.", "Není ale lepší než Pán prstenů.", 0],
                 ["Brno je nejlepší město v České republice", "Na cestě stojí auto.", 1]]

    return sentences


def evaluate(ground_truth: list, predictions: list) -> None:
    """Compare predictions with ground truth and print metrics.

        Parameters
        ----------
        ground_truth : list
            Ground truth values
        predictions : list
            Model predictions
    """

    print(metrics.classification_report(ground_truth, predictions, target_names=["IsNextStc", "IsNotNextStc"], digits=4))
    print(f"AUC = {metrics.roc_auc_score(ground_truth, predictions):.4f}\n")

    fpr, tpr, threshold = metrics.roc_curve(ground_truth, predictions)
    
    plt.clf()
    plt.title(f"ROC")
    plt.plot(fpr, tpr, "b")
    plt.plot([0, 1], [0, 1],"r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig("ROC.pdf")
