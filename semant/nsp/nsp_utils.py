"""Constants and utility functions for NSP

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import typing

from sklearn import metrics
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt
import numpy as np


CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"

JOKER = chr(65533) # Special character used in PERO OCR as a JOKER

# These characters can be generated by PERO OCR but are unknown to Huggingface tokenizer
# We either replace them with similliar character or with JOKER
UNKNOWN_CHARS = {
        "—": "-",
        "ϵ": JOKER,
        "℥": JOKER,
        "‘": JOKER,
        "’": JOKER,
        "`": JOKER,
        "“": '"',
        "☞": JOKER,
        "☜": JOKER,
        "˛": ".",
        "⁂": JOKER,
        "ꝛ": JOKER,
        "Ꙃ": "z",
        "Ꙁ": "z",
        "Ꙋ": JOKER,
        "Ѡ": JOKER,
        "Ꙗ": JOKER,
        "Ѥ": JOKER,
        "Ѭ": JOKER,
        "Ѩ": JOKER,
        "Ѯ": JOKER,
        "Ѱ": JOKER,
        "Ѵ": "v",
        "Ҁ": "c",
        "ꙃ": "z",
        "ꙁ": "z",
        "ꙋ": JOKER,
        "ѡ": "w",
        "ꙗ": JOKER,
        "ѥ": JOKER,
        "ѭ": JOKER,
        "ѩ": JOKER,
        "ѯ": JOKER,
        "ѱ": JOKER,
        "ѵ": "v",
        "ҁ": "c",
        "Ӕ": JOKER,
        "ӕ": JOKER,
        "Ϲ": "c",
        "ϲ": "c",
        "ϳ": "j",
        "ϝ": "f",
        "Ⱥ": "a",
        "ⱥ": "a",
        "Ɇ": "e",
        "ɇ": "e",
        "ᵱ": "p",
        "ꝓ": "p",
        "ꝑ": "p",
        "ꝙ": "q",
        "ꝗ": "q",
        "ꝟ": "v",
}

# These characters cause problems in Huggingface tokenizer and text parsing
# They are removed from OCR texts
# Note that this changes the length of the text
TRUNCATED_CHARS = ["ͤ", "̄", "̾", "̃", "̊"]


def remove_accents(text: str) -> str:
    """Remove accents from text.

        Parameters
        ----------
        text : str
            Text from which accents will be removed. Certain special characters
            are replaced by similliar standard characters or replaced with PERO OCR JOKER.

        Returns
        -------
        text_ : str
            Text with removed accents
    """

    # Replace unknown chars so huggingface tokenizer does not generate [UNK] tokens
    for to_replace, replace_with in UNKNOWN_CHARS.items():
        text_ = text.replace(to_replace, replace_with)

    # Remove special accent characters that get discarded by the tokenizer
    for to_remove in TRUNCATED_CHARS:
        text_ = text_.replace(to_remove, "")

    return text_


def load_data(path: str, raw: bool = True):
    data = []

    with open(path, "r") as f:
        if raw:
            return f.readlines()
        
        for line in f:
            sen1, sen2, label = line.split("\t")

            data.append((sen1.strip(), sen2.strip(), int(label)))

    return data


def evaluate(
    ground_truth: list,
    predictions: list,
    train_loss: list = [],
    val_loss: list = [],
    train_accuracy: list = [],
    val_accuracy: list = [],
    view_step: int = 0,
    val_step: int = 0,
    full: bool = False
    ) -> None:
    """Compare predictions with ground truth and print metrics.

        Parameters
        ----------
        ground_truth : list
            Ground truth values
        predictions : list
            Model predictions
        train_loss : list
            List of training loss over time
        val_loss : list
            List of validation loss over time
        accuracy : list
            List of accuracy over time
        view_step : int
            How often train loss was measured
        val_step : int
            How often validation was performed
        full : bool
            If True, output is also an evaluation.pdf file with plots
    """
    print(metrics.classification_report(ground_truth, predictions, target_names=["IsNextStc", "IsNotNextStc"], digits=4))
    auc = metrics.roc_auc_score(ground_truth, predictions)
    print(f"         AUC     {auc:.4f}\n")

    if not full:
        return

    fig, axs = plt.subplots(1, 4, figsize=(13, 3))
    
    # ROC
    fpr, tpr, threshold = metrics.roc_curve(ground_truth, predictions)
    axs[0].plot(fpr, tpr, "b")
    axs[0].plot([0, 1], [0, 1], "r--")
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    axs[0].set_ylabel("TPR")
    axs[0].set_xlabel("FPR")
    axs[0].set_title(f"ROC (AUC = {auc:.2f})")

    # CM
    cm_display = metrics.ConfusionMatrixDisplay.from_predictions(
        ground_truth,
        predictions,
        colorbar=False,
        ax=axs[1],
        display_labels=["IsNextStc", "IsNotNextStc"],
        )

    # Loss
    x_trn = [val for val in range(view_step, (len(train_loss) + 1) * view_step, view_step)]
    x_val = [val for val in range(val_step, (len(val_loss) + 1) * val_step, val_step)]
    axs[2].plot(x_trn, train_loss, "r", label="train")
    axs[2].plot(x_val, val_loss, "g", label="validation")
    axs[2].legend(fontsize="x-small")
    axs[2].set_title("Loss")
    axs[2].set_xlabel("Steps")
    axs[2].set_ylabel("Value")
    axs[2].grid()

    # Accuracy
    axs[3].plot(x_trn, train_accuracy, "r", label="train")
    axs[3].plot(x_val, val_accuracy, "g", label="validation")
    axs[3].legend(fontsize="x-small")
    axs[3].set_title("Accuracy")
    axs[3].set_xlabel("Steps")
    axs[3].set_ylabel("Value")
    axs[3].set_ylim(0, 1)
    axs[3].grid()

    plt.tight_layout()
    plt.savefig("evaluation.pdf")


def accuracy(ground_truth: list, predictions: list) -> float:
    assert len(predictions) == len(ground_truth)
    return np.sum(np.array(ground_truth) == np.array(predictions)) / len(predictions)


def n_params(model, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def model_size(model) -> float:
    param_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()


    size_all = param_size + buffer_size
    return size_all


def build_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    tokenizer.add_special_tokens({"additional_special_tokens": [JOKER]})

    return tokenizer


if __name__ == "__main__":
    gt = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    predictions = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    train_loss = [0.9, 0.8, 0.5, 0.2]
    val_loss = [1.5, 1.2]
    acc = [0.1, 0.3, 0.4, 0.8]
    view_step = 1000
    val_step = 2000
    evaluate(gt, predictions, train_loss, val_loss, acc, view_step, val_step, True)
