import torch
import torch.nn.functional as F
import numpy as np
import math

from tqdm import tqdm
from models.unet import UNetDiffuser
from data.utils import IclevrDataset, get_testing_labels
from testing.evaluator import evaluation_model
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from pathlib import Path


class DDPMTrainer:
    def __init__(
        self,
        batch_size=64,
        noise_steps=300,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        testing_label_path=Path("./testing/new_test.json"),
    ):
        self.model = UNetDiffuser(
            in_channels=3, out_channels=3, embed_dim=128
        ).to(device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5
        )
        self.loader = DataLoader(
            IclevrDataset(), batch_size=batch_size, shuffle=True
        )
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.batch_size = batch_size

        self.cfg_scale = 5
        
        self.betas = self.prepare_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_inv_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.testing_label_path = testing_label_path
        self.testing_labels = get_testing_labels(testing_label_path).to(self.device)
        self.testing_multi_hot_labels = get_testing_labels(
            self.testing_label_path, as_multi_hot=True
        ).to("cuda")

        self.eval_model = evaluation_model()

    def prepare_noise_schedule(self):
        return torch.linspace(
            self.beta_start, self.beta_end, self.noise_steps, device=self.device
        )

    def add_noise(self, imgs, t_steps):
        sqrt_a_bar = self.sqrt_alphas_cumprod[t_steps][:, None, None, None]
        sqrt_one_m_a_bar = self.sqrt_one_minus_alphas_cumprod[t_steps][
            :, None, None, None
        ]

        noise = torch.randn_like(imgs, device=self.device)

        return sqrt_a_bar * imgs + sqrt_one_m_a_bar * noise, noise

    def gen_batch_timesteps(self, n):
        return torch.randint(0, self.noise_steps - 1, size=(n,), device=self.device)

    @torch.no_grad()
    def sample(self, labels):
        self.model.eval()
        
        n_samples = labels.shape[0]
        
        x = torch.randn(
            (n_samples, 3, self.img_size, self.img_size), device=self.device
        )

        intermediates = []

        for i in tqdm(
            range(self.noise_steps - 1, -1, -1), desc="Sampling image from noise"
        ):
            t_steps = torch.full((n_samples,), i, device=self.device, dtype=torch.long)

            betas = self.betas[t_steps][:, None, None, None]
            sqrt_inv_alphas = self.sqrt_inv_alphas[t_steps][:, None, None, None]
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t_steps][
                :, None, None, None
            ]
            posterior_variances = self.posterior_variance[t_steps][:, None, None, None]

            predicted_noise = self.model(x, t_steps, labels)

            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                probs = self.eval_model.resnet18(x_in)
                log_p = -F.binary_cross_entropy_with_logits(probs, self.testing_multi_hot_labels.float(), reduction="none")
                log_p_y = log_p.sum(dim=1) 
                
                grad = torch.autograd.grad(log_p_y.sum(), x_in)[0]
                noise_level = self.sqrt_alphas_cumprod[t_steps][:, None, None, None]
                scaled_grad = grad * noise_level
                
                guided_noise = predicted_noise - self.cfg_scale * scaled_grad
                
            if i == 0:
                x = sqrt_inv_alphas * (
                    x - betas / sqrt_one_minus_alphas_cumprod * guided_noise
                )
            else:
                x = sqrt_inv_alphas * (
                    x - betas / sqrt_one_minus_alphas_cumprod * guided_noise
                ) + torch.sqrt(posterior_variances) * torch.randn_like(x)

            if i % (self.noise_steps // 7) == 0:
                intermediates.append(x.cpu())
        
        # add final image to list
        intermediates.pop() # update last with newest
        intermediates.append(x.cpu())
        self.model.train()

        return x, torch.stack(intermediates)

    def train_step(self, epoch):
        self.model.train()
        pbar = tqdm(self.loader)

        training_loss = 0
        for sample in pbar:
            images, labels = sample["images"].to(self.device), sample["labels"].to(
                self.device
            )

            t_steps = self.gen_batch_timesteps(images.shape[0])
            x_t, noise = self.add_noise(images, t_steps)

            predicted_noise = self.model(x_t, t_steps, labels)

            loss = F.mse_loss(predicted_noise, noise)

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()

            training_loss += loss.item()

            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        training_loss /= len(self.loader)
        pbar.set_description(f"Epoch {epoch} | Loss: {training_loss:.4f}")

    def train(self, starting_epoch, end_epoch):
        for epoch in range(starting_epoch, end_epoch):
            self.train_step(epoch)

            if (epoch + 1) % 10 == 0:
                self.save_model(f"./saves/checkpoints/ddpm_model_epoch_{epoch+1}.pt")
                samples, _ = self.eval()
                # samples = self.sample(self.testing_labels, return_all=False).clamp(-1, 1)
                self.output_samples(samples, f"samples_epoch_{epoch+1}")

    def eval(self, output_intermediates=False, fname_intermediates="eval"):
        samples, intermediates = self.sample(self.testing_labels)
        samples, intermediates = samples.clamp(-1, 1), intermediates.clamp(-1, 1)
        
        acc = self.eval_model.eval(samples, self.testing_multi_hot_labels)
        print(f"Eval Accuracy: {acc}")

        if output_intermediates:
            self.output_intermediates(intermediates, fname_intermediates)
            
        return samples, acc

    def output_intermediates(self, intermediates, fname="eval"):
        # single Image
        # step, batch, c, h, w
        
        grid = make_grid(intermediates[-1], normalize=True, nrow=8, value_range=(-1, 1))
        img = to_pil_image(grid)
        img.save(Path("./saves/samples") / f"{fname}.png")
        # steps
        intermediates = intermediates.permute(1, 0, 2, 3, 4)
        
        for i, sample in enumerate(intermediates):
            grid = make_grid(sample, nrow=8, normalize=True, value_range=(-1, 1))
            img = to_pil_image(grid)
            img.save(Path("./saves/samples") / f"{fname}_{i}_steps.png")
                
    def output_samples(self, samples, fname):
        # single Image
        grid = make_grid(samples, normalize=True, nrow=8, value_range=(-1, 1))
        img = to_pil_image(grid)
        img.save(Path("./saves/samples") / f"{fname}.png")

        # each imamge
        for i, img_tensor in enumerate(samples):
            img_tensor = (img_tensor + 1) * 0.5
            img = to_pil_image(img_tensor)
            img.save(Path("./saves/samples") / f"{fname}_num_{i}.png")

        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    trainer = DDPMTrainer()
    trainer.load_model("saves/checkpoints/ddpm_model_epoch_230.pt")
    # trainer.train(starting_epoch=100, end_epoch=250)

    # x = trainer.sample(len(trainer.testing_labels), trainer.testing_labels, return_all=True).clamp(-1, 1)
    # trainer.output_samples(x, "testing.png")
    samples, acc = trainer.eval(output_intermediates=True, fname_intermediates="test")
    trainer.output_samples(samples, "test")
