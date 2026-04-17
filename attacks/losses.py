import torch.nn.functional as F


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def detection_loss(logits, is_adv_labels):
    return F.cross_entropy(logits, is_adv_labels)


# goal is to maximize classification loss to fool the classifier D, and 
# minimize detection loss (look clean) to fool the classifier C
def combined_loss(logits_C, logits_D, y_class, adv_label, alpha=0.5):
    return (
        classification_loss(logits_D, y_class)
        - alpha * detection_loss(logits_C, adv_label)
    )