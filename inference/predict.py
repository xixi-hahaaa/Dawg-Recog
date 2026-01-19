import torch
import torch.nn.functional as F

def predict_topk(outputs, class_names, k=3):
    probs = F.softmax(outputs, dim=1)
    top_probs, top_idxs = torch.topk(probs, k)

    return [
        (class_names[idx], prob.item())
        for prob, idx in zip(top_probs[0], top_idxs[0])
    ]

