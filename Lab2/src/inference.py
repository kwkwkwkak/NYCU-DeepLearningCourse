import argparse
import os
import torch
import cv2
import numpy as np

from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import UNet
from models.resnet34_unet import ResNet34Unet
from utils import overlay_mask, dice_loss_quantized
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default='..\\dataset\\oxford-iiit-pet\\', help='path to the input data')
    parser.add_argument('--output', action='store_true', help='saves inference results if set')

    #parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = 'cuda'
    if args.model.startswith('unet'):
        model = UNet()
    else:
        model = ResNet34Unet()

    model.load_state_dict(torch.load(f"..\\saved_models\\{args.model}", weights_only=True))
    model.to(device)

    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(os.getcwd(), args.data_path)

    dataset = load_dataset(data_path, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    mean, std = torch.tensor(dataset.mean), torch.tensor(dataset.std)
    with torch.no_grad():
        eval_loss = 0.0
        for batch in tqdm(loader):
            image = batch['image'].to(device)
            label = None if args.output else batch['mask'].to(device)
            
            output = model(image)
            pred_mask = torch.sigmoid(output) > 0.5          

            if args.output:
                mask_output = (pred_mask.cpu().numpy().astype(np.uint8) * 255)[0, 0]
                cv2.imwrite(f"..\\outputs\\mask\\mask_{batch[0]['filename']}.png", mask_output)
                
                reverted_image = image[0].cpu() * std[:, None, None] + mean[:, None, None]
                overlay = (255 * np.flip(overlay_mask(reverted_image, pred_mask[0]), axis=2)).astype(np.uint8)
                cv2.imwrite(f"..\\outputs\\overlay\\overlay_{batch[0]['filename']}.png", overlay)
            else:
                eval_loss += dice_loss_quantized(output, label)

        if not args.output:
            eval_loss /= len(loader)
            print(f"Eval Loss on set (why is there gt data on the testing set): {eval_loss:.4f}")
