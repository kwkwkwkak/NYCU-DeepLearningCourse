import torch
import numpy as np

def overlay_mask(image, mask, alpha=0.3):
    if torch.is_tensor(image):
        image = image.cpu().numpy().transpose(1, 2, 0)
        mask = mask.cpu().numpy()
        
    overlay = image.copy()
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + alpha * mask, a_min=0, a_max=1)
    
    return overlay

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    assert pred_mask.shape == gt_mask.shape
    pred_mask = torch.sigmoid(pred_mask)
    return (2 * torch.sum(pred_mask * gt_mask, dim=(2, 3))) / (torch.sum(pred_mask, dim=(2, 3)) + torch.sum(gt_mask, dim=(2, 3)))

def dice_loss(pred_mask, gt_mask):
    return 1 - dice_score(pred_mask, gt_mask).mean()

def dice_loss_quantized(pred_mask, gt_mask):
    pred_mask = torch.sigmoid(pred_mask) > 0.5
    return 1 - ((2 * torch.sum(pred_mask * gt_mask, dim=(2, 3))) / (torch.sum(pred_mask, dim=(2, 3)) + torch.sum(gt_mask, dim=(2, 3)))).mean()

def dice_scoretest(targets, predictions):
    return (targets == predictions).sum() / (65536)

if __name__ == "__main__":
    t = torch.randint(0, 2, (1, 1, 256, 256))
    p = torch.randint(0, 2, (1, 1, 256, 256))
    
    print(dice_score(p, t), dice_scoretest(p, t))