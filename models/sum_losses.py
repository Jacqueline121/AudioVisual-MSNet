import numpy as np
import torch
import torch.nn.functional as F
    
def cross_entropy_loss(output, label, batch_average=False, reduce=True):

    batch_size = output.size(0)
    output = output.view(batch_size, -1)
    label = label.view(batch_size, -1)

    final_loss = F.binary_cross_entropy_with_logits(output, label, reduce=False).sum(1)

    if reduce:
        final_loss = torch.sum(final_loss) / batch_size

    return final_loss
