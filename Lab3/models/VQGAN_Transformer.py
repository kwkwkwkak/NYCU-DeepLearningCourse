import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        codebook_indices = codebook_indices.view(codebook_mapping.shape[0], -1)
        return codebook_mapping, codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda ratio : 1 - ratio
        elif mode == "cosine":
            return lambda ratio : np.cos(np.pi * ratio * 0.5)
        elif mode == "square":
            return lambda ratio : 1 - ratio ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        
        _, z_indices = self.encode_to_z(x) #ground truth
        
        mask_ratio = self.gamma(np.random.uniform())
        # mask = torch.rand_like(z_indices) < mask_ratio
        # masked_z_indices = z_indices.clone()
        # masked_z_indices[mask] = self.mask_token_id
        num_mask_tokens = math.ceil(mask_ratio * z_indices.shape[1])
        
        mask_positions = torch.rand(z_indices.shape, device=x.device).topk(
            num_mask_tokens, 
            dim=1
        ).indices
        
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=x.device)
        mask.scatter_(dim=1, index=mask_positions, value=True)
        
        masked_z_indices = torch.where(mask, self.mask_token_id, z_indices)

        logits = self.transformer(masked_z_indices) #transformer predict the probability of tokens
        return logits, z_indices, mask
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, tokens, mask, total_masked, step, total_iters):
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        logits = self.transformer(masked_tokens)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        #raise NotImplementedError
        probs = torch.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = probs.max(dim=-1)

        if step < total_iters - 1:
            ratio = self.gamma(step / (total_iters - 1)) 
            next_ratio = self.gamma((step + 1) / (total_iters - 1))
            
            #predicted probabilities add temperature annealing gumbel noise as confidence
            g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob) + 1e-10) + 1e-10)  # gumbel noise
            temperature = self.choice_temperature * (1 - ratio)
            confidence = z_indices_predict_prob + temperature * g
            
            #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
            #sort the confidence for the rank 
            #define how much the iteration remain predicted tokens by mask scheduling
            ##At the end of the decoding process, add back the original(non-masked) token values
            confidence[~mask] = float('inf')
            
            # Calculate number of tokens to unmask
            #num_masked = mask.sum().item()
            num_to_unmask = mask.shape[1] - int(total_masked * (next_ratio))
            top_confidence_positions = confidence.topk(num_to_unmask, sorted=False).indices
            #print(top_confidence_positions)
            mask_bc = mask.clone()
            mask_bc[0, top_confidence_positions] = False
            z_indices_predict[~mask] = tokens[~mask]
            return z_indices_predict, mask_bc
        else:
            z_indices_predict[~mask] = tokens[~mask]
            return z_indices_predict, mask
        #print(mask_bc)
        
        
        
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
