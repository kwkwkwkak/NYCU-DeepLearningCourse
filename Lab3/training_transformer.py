import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from functools import partial

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS, train_loader, test_loader, start_from_last):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.total_steps = len(train_loader) * args.epochs // args.accum_grad
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.epochs = args.epochs
        self.device = args.device
        self.lowest_loss = 1e9
        self.save_path = args.checkpoint_path
        self.acc_steps = args.accum_grad
        
        if start_from_last:
            self.model.load_transformer_checkpoint('./checkpoints/last_ckpt.pt')
        
    @staticmethod
    def prepare_training():
        os.makedirs("checkpoints", exist_ok=True)

    def train_one_epoch(self):
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(tqdm(self.train_loader)):
            batch = batch.to(self.device)
            logits, targets, mask = self.model(batch)
            #print(logits.shape, targets.shape, mask.shape)
            flat_mask = mask.view(-1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1))[flat_mask], targets.reshape(-1)[flat_mask])
            loss.backward()
            
            if (step + 1) % self.acc_steps == 0:  
                self.optim.step()
                self.scheduler.step()
                self.optim.zero_grad()
                
            total_loss += loss.item()
            num_batches += 1
            
        if (step + 1) % self.acc_steps != 0:
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Training Loss: {avg_loss:.4f}")
        
        torch.save(self.model.transformer.state_dict(), './checkpoints/last_ckpt.pt')
            
    @torch.no_grad()
    def eval_one_epoch(self):
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.test_loader):
            batch = batch.to(self.device)
            logits, targets, mask = self.model(batch)

            flat_mask = mask.view(-1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1))[flat_mask], targets.reshape(-1)[flat_mask])
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Evaluation Loss: {avg_loss:.4f}")
        
        if avg_loss < self.lowest_loss:
            self.lowest_loss = avg_loss
            torch.save(self.model.transformer.state_dict(), './checkpoints/best_ckpt.pt')
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.transformer.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.total_steps // 10, self.total_steps)
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    parser.add_argument('--start_from_last', action='store_true', help='Continue training from last.ckpt')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS, train_loader, val_loader, args.start_from_last)
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f"Epoch {epoch}")
        train_transformer.train_one_epoch()
        train_transformer.eval_one_epoch()