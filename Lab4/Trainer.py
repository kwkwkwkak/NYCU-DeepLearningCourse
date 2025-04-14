import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10, isnan
import pickle

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.current_epoch = current_epoch
        self.use_schedule = True

        if args.kl_anneal_type == "Cyclical":
            self.schedule = self.frange_cycle_linear(args.num_epoch, 
                                                     n_cycle=args.kl_anneal_cycle, 
                                                     ratio=args.kl_anneal_ratio)
        elif args.kl_anneal_type == "Monotonic":
            self.schedule = self.frange_cycle_linear(args.num_epoch, ratio=args.kl_anneal_ratio)
        elif args.kl_anneal_type == "Without":
            self.use_schedule = False
        else:
            raise Exception("Unsupported annealing strategy")
        
    def update(self):
        self.current_epoch += 1
    
    def get_beta(self):
        return self.schedule[self.current_epoch] if self.use_schedule else 1

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        T = n_iter / n_cycle
        inc_T = T * ratio
        m = (stop - start) / inc_T
        betas = [stop] * n_iter

        T = int(T)
        inc_T = int(inc_T)
        for cycle in range(n_cycle):
            accum = start
            for i in range(inc_T):
                betas[cycle * T + i] = accum
                accum += m

        return betas
# warmup_steps = 10
# peak_lr = 5e-5
# initial_lr = 1e-5

# def warmup_lambda(step):
#     if step >= warmup_steps:
#         return peak_lr / initial_lr  # Keep at peak after warmup is done
#     else:
#         # Linear ramp from initial_lr to peak_lr
#         return ((peak_lr - initial_lr) * step / warmup_steps + initial_lr) / initial_lr
    
class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=args.lr)
        
        # warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=warmup_lambda)

        # decay_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optim,
        #     T_max=60,
        #     eta_min=5e-6
        # )
        
        # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        # self.scheduler = optim.lr_scheduler.SequentialLR(
        #     self.optim,
        #     schedulers=[warmup_scheduler, decay_scheduler],
        #     milestones=[10]
        # )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.6, patience=4)
        # fine tuning scheduler
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.lowest_eval_loss = float('inf')
        self.training_losses = []
        self.eval_losses = []
        self.tfrs = []
        
        
    def forward(self, x, x_next, p_next):
        x_transform = self.frame_transformation(x)
        x_next_transform = self.frame_transformation(x_next)
        p_next_transform = self.label_transformation(p_next)

        z, mu, logvar = self.Gaussian_Predictor(x_next_transform, p_next_transform)

        decoder_out = self.Decoder_Fusion(x_transform, p_next_transform, z)
        x_next_pred = self.Generator(decoder_out)

        return mu, logvar, x_next_pred

    def forward_eval(self, x, p_next):
        x_transform = self.frame_transformation(x)
        p_next_transform = self.label_transformation(p_next)

        z = torch.randn((1, self.args.N_dim, self.args.frame_H, self.args.frame_W), device=self.args.device)

        decoder_out = self.Decoder_Fusion(x_transform, p_next_transform, z)
        x_next_pred = self.Generator(decoder_out)

        return x_next_pred
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            self.train()

            training_loss = 0.0
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device) # (batch, numFrames, c, h, w)
                label = label.to(self.args.device)
                beta = self.kl_annealing.get_beta()

                loss = self.training_one_step(img, label, adapt_TeacherForcing, beta)
                training_loss += loss.item()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.4f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.4f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            training_loss /= len(train_loader)
            if isnan(training_loss):
                raise ValueError("Nan training loss.")
            
            self.training_losses.append(training_loss)
            self.tfrs.append(self.tfr)
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            e_loss = self.evaluate()
            self.current_epoch += 1
            self.scheduler.step(e_loss)
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
        plt.figure()
        plt.plot(self.training_losses)
        plt.title("Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("training_losses.png")
        plt.close()

        plt.figure()
        plt.plot(self.tfrs)
        plt.title("TFR")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.savefig("tfr.png")
        plt.close()
        
        data = {
            "training_losses": self.training_losses,
            "eval_losses": self.eval_losses,
            "tfr": self.tfrs
        }

        with open("losses_data.pkl", "wb") as f:
            pickle.dump(data, f)
        
    @torch.no_grad()
    def evaluate(self):
        val_loader = self.val_dataloader()
        self.eval()

        eval_loss = 0.0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)

            eval_loss += loss.item()
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

        eval_loss /= len(val_loader)
        self.eval_losses.append(eval_loss)
        if eval_loss < self.lowest_eval_loss:
            self.lowest_eval_loss = eval_loss
            self.save(os.path.join(self.args.save_root, f"best.ckpt"))
        return eval_loss
    
    def training_one_step(self, img, label, adapt_TeacherForcing, beta):
        self.optim.zero_grad()

        mse_loss = 0.0
        kl_loss = 0.0

        x = img[:, 0]
        len_to_predict = img.shape[1] - 1

        for i in range(len_to_predict):
            if adapt_TeacherForcing:
                x = img[:, i]
            x_next = img[:, i + 1]
            p_next = label[:, i + 1]
            mu, logvar, x_next_pred = self.forward(x, x_next, p_next)
            kl_loss += kl_criterion(mu, logvar, self.batch_size)
            mse_loss += self.mse_criterion(x_next_pred, x_next)
            
            x = x_next_pred
        
        loss = mse_loss + beta * kl_loss
        loss.backward()
        self.optimizer_step()

        return loss
    
    def val_one_step(self, img, label):
        mse_loss = 0.0

        x = img[:, 0]
        len_to_predict = img.shape[1] - 1

        for i in range(len_to_predict):
            x_next = img[:, i + 1]
            p_next = label[:, i + 1]

            x_next_pred = self.forward_eval(x, p_next)
            mse_loss += self.mse_criterion(x_next_pred, x_next)

            x = x_next_pred

        return mse_loss
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch > self.tfr_sde:
            self.tfr = max(0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.4f}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            #self.args.lr = checkpoint['lr']
            #self.tfr = checkpoint['tfr']
            
            #self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-4)
            #self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-4)
            #self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
            #self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=4)
    parser.add_argument('--lr',            type=float,  default=1e-4,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu", "cuda:1"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=5,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=0.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=0,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.0,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=100,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Monotonic',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=1,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.2,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
