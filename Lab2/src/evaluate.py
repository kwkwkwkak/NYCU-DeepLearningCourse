import torch
import os

from models.unet import UNet
from models.resnet34_unet import ResNet34Unet
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from utils import dice_loss_quantized
from tqdm import tqdm

def evaluate(net, name, data, device):
    # implement the evaluation function here
    net.eval()
    
    with torch.no_grad():
        eval_loss = 0.0
        for batch in tqdm(data):
            images, lables = batch['image'].to(device), batch['mask'].to(device)
            outputs = net(images)
            
            eval_loss += dice_loss_quantized(outputs, lables)
            
        eval_loss /= len(data)
        
        print(f"{name} Eval Loss: {eval_loss:.4f}")
        
        return eval_loss
        
if __name__ == "__main__":
    
    device = "cuda"
    dataset = load_dataset(os.path.join(os.getcwd(), '..\\dataset\\oxford-iiit-pet\\'), mode='valid')
    dataloader = DataLoader(dataset)
    
    best_ckpt = ""
    best_loss = 1

    for checkpoint in os.listdir("..\\saved_models\\"):
        if checkpoint.startswith("unet"):
            model = UNet()
        elif checkpoint.startswith("res"):
            model = ResNet34Unet()
        else:
            raise Exception("Model name does not start with unet or resnet.")
        
        model.load_state_dict(torch.load(f"..\\saved_models\\{checkpoint}", weights_only=True))
        model.to(device)
        eval_loss = evaluate(model, checkpoint, dataloader, device)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_ckpt = checkpoint
        
    print(f"Best Checkpoint: {best_ckpt}, Eval Loss: {best_loss}")