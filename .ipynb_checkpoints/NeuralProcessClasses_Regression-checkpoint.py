from architecture_classes import *
import torch


from architecture_classes import *
from utils import *
import torch


class ConditionalNeuralProcess(torch.nn.Module):
    """
    Class defining conditional neural processes for regression problems with arbitrary X-dimension

    Parameters
    ----------

    encoder: module, MLP
    decoder: module, MLP
    last_decoder_layer_dim: output of last decoder layer
    y_dim: output_dimension, int

    """

    def __init__(self, encoder, decoder,last_decoder_layer_dim, y_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.lastLayer_mu = torch.nn.Linear(last_decoder_layer_dim, y_dim)
        self.lastLayer_log_var = torch.nn.Linear(last_decoder_layer_dim, y_dim)
        

    def forward(self, context_x, context_y, target_x, target_y = None):
        full_context = torch.concat([context_x, context_y], dim=-1)

        encoder_output = self.encoder(full_context)
        encoder_output = torch.mean(encoder_output, dim=1)

        encoder_output = encoder_output.unsqueeze(dim=1).tile((1,target_x.shape[1],1))

        decoder_input = torch.concat([target_x, encoder_output], dim=-1)

        decoder_output = self.decoder(decoder_input)

        prediction_mu = self.lastLayer_mu(decoder_output)
        prediction_log_var = self.lastLayer_log_var(decoder_output)

        prediction_var = 0.1 + 0.9*torch.exp(prediction_log_var/2)

        if target_y is not None:

            p_y_target = torch.distributions.Normal(prediction_mu, prediction_var)
            log_likelihood = p_y_target.log_prob(target_y).mean(dim=0).sum()
            return prediction_mu, prediction_var, -log_likelihood
            
        else:
            
            return prediction_mu, prediction_var


class LatentNeuralProcess(torch.nn.Module):
    """
    Class defining latent neural processes for regression problems with arbitrary X-dimension

    Parameters
    ----------

    deterministic_encoder: module, MLP
    latent_encoder: module, MLP
    last_encoder_layer_dim: out dim of last encoder layer, int
    latent_dim: dimension of latent, int
    decoder: module, MLP
    last_decoder_layer_dim: output of last decoder layer, int
    y_dim: output_dimension, int

    """

    def __init__(self, deterministic_encoder, latent_encoder, last_encoder_layer_dim, latent_dim, decoder,last_decoder_layer_dim, y_dim, **kwargs):
        
        super().__init__(**kwargs)
        
        self.deterministic_encoder = deterministic_encoder
        
        self.latent_encoder = latent_encoder
        self.latentLayer_mu = torch.nn.Linear(last_encoder_layer_dim, latent_dim)
        self.latentLayer_log_var = torch.nn.Linear(last_encoder_layer_dim, latent_dim)
        
        self.decoder = decoder
        self.lastLayer_mu = torch.nn.Linear(last_decoder_layer_dim, y_dim)
        self.lastLayer_log_var = torch.nn.Linear(last_decoder_layer_dim, y_dim)
        

    def forward(self, context_x, context_y, target_x, target_y = None):
        full_context = torch.concat([context_x, context_y], dim=-1)

        deterministic_encoder_output = self.deterministic_encoder(full_context)
        deterministic_encoder_output = torch.mean(deterministic_encoder_output, dim=1)

        r = deterministic_encoder_output.unsqueeze(dim=1).tile((1,target_x.shape[1],1))

        latent_encoder_output = self.latent_encoder(full_context)
        latent_encoder_output = torch.mean(latent_encoder_output, dim=1)

        z_mu = self.latentLayer_mu(latent_encoder_output)
        z_log_var = self.latentLayer_log_var(latent_encoder_output)

        z = torch.randn_like(z_log_var) * torch.exp(z_log_var/2) + z_mu

        z = z.unsqueeze(dim=1).tile((1, target_x.shape[1],1))

        decoder_input = torch.concat([target_x, r, z], dim=-1)

        decoder_output = self.decoder(decoder_input)

        prediction_mu = self.lastLayer_mu(decoder_output)
        prediction_log_var = self.lastLayer_log_var(decoder_output)

        prediction_var = 0.1 + 0.9*torch.exp(prediction_log_var/2)

        if target_y is not None:

            p_y_target = torch.distributions.Normal(prediction_mu, prediction_var)
            log_likelihood = p_y_target.log_prob(target_y).mean(dim=0).sum()

            full_XY = torch.concat([target_x, target_y], dim=-1)

            target_latent_encoder_output = self.latent_encoder(full_XY)
            target_latent_encoder_output = torch.mean(target_latent_encoder_output, dim=1)

            z_mu_target = self.latentLayer_mu(target_latent_encoder_output)
            z_log_var_target = self.latentLayer_log_var(target_latent_encoder_output)

            kl = self.kl_div_calc(z_mu, z_log_var, z_mu_target, z_log_var_target)
            
            return prediction_mu, prediction_var, -log_likelihood + torch.sum(kl)
            
        else:
            
            return prediction_mu, prediction_var



    def kl_div_calc(self, z_mu_context, z_log_var_context, z_mu_target, z_log_var_target):
        q_context = torch.distributions.Normal(z_mu_context, torch.exp(z_log_var_context/2))
        q_target = torch.distributions.Normal(z_mu_target, torch.exp(z_log_var_target/2))
        
        kl = torch.distributions.kl.kl_divergence(q_context, q_target)
        return kl



class LatentNeuralProcess(torch.nn.Module):
    """
    Class defining latent neural processes for regression problems with arbitrary X-dimension

    Parameters
    ----------

    deterministic_encoder: module, MLP
    latent_encoder: module, MLP
    last_encoder_layer_dim: out dim of last encoder layer, int
    latent_dim: dimension of latent, int
    decoder: module, MLP
    last_decoder_layer_dim: output of last decoder layer, int
    y_dim: output_dimension, int

    """

    def __init__(self, deterministic_encoder, latent_encoder, last_encoder_layer_dim, latent_dim, decoder,last_decoder_layer_dim, y_dim, **kwargs):
        
        super().__init__(**kwargs)
        
        self.deterministic_encoder = deterministic_encoder
        
        self.latent_encoder = latent_encoder
        self.latentLayer_mu = torch.nn.Linear(last_encoder_layer_dim, latent_dim)
        self.latentLayer_log_var = torch.nn.Linear(last_encoder_layer_dim, latent_dim)
        
        self.decoder = decoder
        self.lastLayer_mu = torch.nn.Linear(last_decoder_layer_dim, y_dim)
        self.lastLayer_log_var = torch.nn.Linear(last_decoder_layer_dim, y_dim)
        

    def forward(self, context_x, context_y, target_x, target_y = None):
        full_context = torch.concat([context_x, context_y], dim=-1)

        deterministic_encoder_output = self.deterministic_encoder(full_context)
        deterministic_encoder_output = torch.mean(deterministic_encoder_output, dim=1)

        r = deterministic_encoder_output.unsqueeze(dim=1).tile((1,target_x.shape[1],1))

        latent_encoder_output = self.latent_encoder(full_context)
        latent_encoder_output = torch.mean(latent_encoder_output, dim=1)

        z_mu = self.latentLayer_mu(latent_encoder_output)
        z_log_var = self.latentLayer_log_var(latent_encoder_output)

        z = torch.randn_like(z_log_var) * torch.exp(z_log_var/2) + z_mu

        z = z.unsqueeze(dim=1).tile((1, target_x.shape[1],1))

        decoder_input = torch.concat([target_x, r, z], dim=-1)

        decoder_output = self.decoder(decoder_input)

        prediction_mu = self.lastLayer_mu(decoder_output)
        prediction_log_var = self.lastLayer_log_var(decoder_output)

        prediction_var = 0.1 + 0.9*torch.exp(prediction_log_var/2)

        if target_y is not None:

            p_y_target = torch.distributions.Normal(prediction_mu, prediction_var)
            log_likelihood = p_y_target.log_prob(target_y).mean(dim=0).sum()

            full_XY = torch.concat([target_x, target_y], dim=-1)

            target_latent_encoder_output = self.latent_encoder(full_XY)
            target_latent_encoder_output = torch.mean(target_latent_encoder_output, dim=1)

            z_mu_target = self.latentLayer_mu(target_latent_encoder_output)
            z_log_var_target = self.latentLayer_log_var(target_latent_encoder_output)

            kl = self.kl_div_calc(z_mu, z_log_var, z_mu_target, z_log_var_target)
            
            return prediction_mu, prediction_var, -log_likelihood + torch.sum(kl)
            
        else:
            
            return prediction_mu, prediction_var



    def kl_div_calc(self, z_mu_context, z_log_var_context, z_mu_target, z_log_var_target):
        q_context = torch.distributions.Normal(z_mu_context, torch.exp(z_log_var_context/2))
        q_target = torch.distributions.Normal(z_mu_target, torch.exp(z_log_var_target/2))
        
        kl = torch.distributions.kl.kl_divergence(q_context, q_target)
        return kl

        

    