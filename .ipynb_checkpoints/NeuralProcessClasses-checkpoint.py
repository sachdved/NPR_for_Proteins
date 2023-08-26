from architecture_classes import *
import torch

class ConditionalNeuralProcess(torch.nn.Module):
    """Basic Determinstic Conditional Neural Process. Takes a decoder and encoder of the form of MLPs and trains. Current use case for proteins, but can and should be generalized to continuous data of time series type.
    input:
        encoder: MLP
        decoder: MLP

        for forward method:
            context_x, context_y, target_x, target_y (optional)

    output:
        y_pred: probabilistic predictions of amino acids of size (batch size, target points, number of amino acids)
        loss: cross entropy loss of probabilistic predictions. Only given if target_y is specified
    """
    def __init__(self,
                 encoder,
                 decoder,
                 **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, context_x, context_y, target_x, target_y = None):
        full_context = torch.concat([context_x, context_y], dim=-1)

        encoder_output = self.encoder(full_context)
        encoder_output = torch.mean(encoder_output, dim=1)

        encoder_output = encoder_output.unsqueeze(dim=1).tile((1,target_x.shape[1],1))

        decoder_input = torch.concat([target_x, encoder_output], dim=-1)
        decoder_output = self.decoder(decoder_input)

        decoder_output = torch.nn.Softmax(dim=-1)(decoder_output)

        
        if target_y is not None:
            loss = self.cross_entropy_loss(decoder_output, target_y)
            return decoder_output, loss
        else:
            return decoder_output
        
    def cross_entropy_loss(self, output, target_y):
        assert output.shape == target_y.shape

        cross_entropy_per_site_per_seq = -torch.sum(target_y * torch.log(output+1e-6), dim=-1)
        cross_entropy = torch.mean(cross_entropy_per_site_per_seq)
        
        return cross_entropy

class LatentNeuralProcess(torch.nn.Module):
    """Basic Latent Neural Process. Takes a decoder and encoder of the form of MLPs, creates a latent variable, and then predicts. Current use case for proteins, but can and should be generalized to continuous data of time series type.
    input:
        determinstic_encoder: MLP
        latent_encoder: MLP
        latentLayer_mu: MLP
        latentLayer_log_var: MLP
        decoder: MLP

        for forward method:
            context_x, context_y, target_x, target_y (optional)

    output:
        y_pred: probabilistic predictions of amino acids of size (batch size, target points, number of amino acids)
        loss: cross entropy loss of probabilistic predictions. Only given if target_y is specified
    """
    def __init__(self,
                 deterministic_encoder,
                 latent_encoder,
                 latentLayer_mu,
                 latentLayer_log_var,
                 decoder,
                 **kwargs
    ):
        super().__init__(**kwargs)
        
        self.deterministic_encoder = deterministic_encoder

        self.latent_encoder = latent_encoder
        
        self.latentLayer_mu = latentLayer_mu
        self.latentLayer_log_var = latentLayer_log_var

        self.decoder = decoder

    def forward(self, context_x, context_y, target_x, target_y = None):
        full_context = torch.concat([context_x, context_y], dim=-1)

        deterministic_encoder_output = self.deterministic_encoder(full_context)
        deterministic_encoder_output = torch.mean(deterministic_encoder_output, dim=1)

        latent_encoder_output = self.latent_encoder(full_context)
        latent_encoder_output = torch.mean(latent_encoder_output, dim=1)

        z_mu = self.latentLayer_mu(latent_encoder_output)
        z_log_var = self.latentLayer_log_var(latent_encoder_output)

        z = torch.randn_like(z_log_var) * torch.exp(z_log_var/2) + z_mu

        z = z.unsqueeze(dim=1).tile((1, target_x.shape[1],1))

        r = deterministic_encoder_output.unsqueeze(dim=1).tile((1, target_x.shape[1],1))

        decoder_input = torch.concat([target_x, r, z], dim=-1)
        decoder_output = self.decoder(decoder_input)

        decoder_output = torch.nn.Softmax(dim=-1)(decoder_output)

        if target_y is not None:
            loss = self.cross_entropy_loss(decoder_output, target_y)
            
            X_full = torch.concat([context_x, target_x], dim=1)
            Y_full = torch.concat([context_y, target_y], dim=1)
            
            full_XY = torch.concat([X_full, Y_full], dim=-1)
            
            target_latent_encoder_output = self.latent_encoder(full_XY)
            target_latent_encoder_output = torch.mean(target_latent_encoder_output, dim=1)
            
            z_mu_target = self.latentLayer_mu(target_latent_encoder_output)
            z_log_var_target = self.latentLayer_log_var(target_latent_encoder_output)
            
            
            kl = self.kl_div_calc(z_mu, z_log_var, z_mu_target, z_log_var_target)
            return decoder_output, loss, kl
        else:
            return decoder_output        
        
    def cross_entropy_loss(self, output, target_y):
        assert output.shape == target_y.shape

        cross_entropy_per_site_per_seq = -torch.sum(target_y * torch.log(output+1e-6), dim=-1)
        cross_entropy = torch.mean(cross_entropy_per_site_per_seq)
        
        return cross_entropy

    def kl_div_calc(self, z_mu_context, z_log_var_context, z_mu_target, z_log_var_target):
        q_context = torch.distributions.Normal(z_mu_context, torch.exp(z_log_var_context/2))
        q_target = torch.distributions.Normal(z_mu_target, torch.exp(z_log_var_target/2))
        
        kl = torch.distributions.kl.kl_divergence(q_context, q_target)
        return kl

class AttentiveNeuralProcess(torch.nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.target_projection = MLP(1, [128,128,128,128])
        self.context_projection = MLP(1, [128,128,128,128])

    def forward(self, context_x, context_y, target_x, target_y = None):
        concat_input = torch.concat([context_x, context_y], dim=-1)
        encoder_input = self.encoder[0](concat_input)
        for layer in self.encoder[1:]:
            encoder_input, _ = layer(encoder_input, encoder_input, encoder_input)
            encoder_input = encoder_input.permute((0,2,1,3)).reshape((encoder_input.shape[0], encoder_input.shape[2], encoder_input.shape[1]*encoder_input.shape[3]))

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        for layer in self.decoder[:-1]:
            query , _ = layer(query, keys, encoder_input)
            query = query.permute((0,2,1,3)).reshape((query.shape[0], query.shape[2], query.shape[1]*query.shape[3]))

        concatenated_final_entry = torch.concat([query, target_x], dim=-1)
        output = self.decoder[-1](concatenated_final_entry)

        output = torch.nn.Softmax(dim=-1)(output)

        if target_y is not None:
            loss = self.cross_entropy_loss(output, target_y)
            return output, loss
        else:
            return output
        

        

    def cross_entropy_loss(self, output, target_y):
        assert output.shape == target_y.shape

        cross_entropy_per_site_per_seq = torch.sum(-target_y * torch.log(output + 1e-6), dim=-1)
        cross_entropy_per_seq = torch.mean(cross_entropy_per_site_per_seq, dim=-1)
        cross_entropy = torch.mean(cross_entropy_per_seq)

        return cross_entropy
        
        