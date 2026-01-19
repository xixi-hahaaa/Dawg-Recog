import torch
import torch.nn.functional as F

def is_dog(outputs, threshold=0.4):
    """
    outputs: raw logits from model (1, num_classes)
    """
    probs = F.softmax(outputs, dim=1)
    max_prob, pred = torch.max(probs, dim=1)

    if max_prob.item() < threshold:
        return False, max_prob.item(), pred.item()
    return True, max_prob.item(), pred.item()
