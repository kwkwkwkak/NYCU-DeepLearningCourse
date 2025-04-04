import argparse
import os
import torch
import random
import sys

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34Unet
from torch.utils.data import DataLoader
from utils import dice_loss
from tqdm import tqdm
from evaluate import evaluate

import matplotlib.pyplot as plt

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)

def train(args):
    # implement the training function here
    if args.model == 'unet':
        model = UNet()
        model_name = "unet"
    else:
        model = ResNet34Unet()
        model_name = "resnet34+unet"
    
    #print(model)
    model.to('cuda')
    
    g = torch.Generator()
    g.manual_seed(0)
    
    dataset = load_dataset(os.path.join(os.getcwd(), '..\\dataset\\oxford-iiit-pet\\'), mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, worker_init_fn=seed_worker, generator=g)

    eval_dataset = load_dataset(os.path.join(os.getcwd(), '..\\dataset\\oxford-iiit-pet\\'), mode='valid')
    eval_dataloader = DataLoader(eval_dataset)

    optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate, decoupled_weight_decay=True, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5 * args.learning_rate, epochs=args.epochs, steps_per_epoch=len(dataloader))
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.77)
    lowest_eval_loss = 1

    training_losses = []
    valid_losses = []
    for epoch in tqdm(range(args.epochs)):
        epoch_loss = 0
        model.train()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = batch['image'].to('cuda'), batch['mask'].to('cuda')
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            assert outputs.shape == labels.shape
            
            #loss = dice_loss(outputs, labels)
            loss = 0.7 * criterion(outputs, labels) + 0.3 * dice_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} Training loss (BCE+Dice): {avg_loss:.4f}")
        training_losses.append(avg_loss)
        
        eval_loss = evaluate(model, '', eval_dataloader, 'cuda')
        valid_losses.append(eval_loss.cpu().numpy())
        if eval_loss < lowest_eval_loss:
            lowest_eval_loss = eval_loss
            model_path = f"..\\saved_models\\{model_name}_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Best model updated.")

    plt.plot(training_losses, label="Training Loss (BCE+Dice)", color="green")
    plt.plot(valid_losses, label="Validation Loss (Dice)", color="orange")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(args.model)
    plt.legend()
    
    plt.savefig(f"{args.model}_loss_trend.png")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'res'], help='unet or resnet34+unet')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    # torch.manual_seed(448936829540800,)
    # torch.cuda.manual_seed(8976052598629137)
    # random.seed(85380090418149233)
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    
    train(args)
    print(f"Training dataset random seed: {seed}")
    print(f"Torch initial seed: {torch.initial_seed()}, Cuda initial seed: {torch.cuda.initial_seed()}")