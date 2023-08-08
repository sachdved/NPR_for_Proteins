from architecture_classes import *
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

    def __init__(self, encoder, decoder, last_decoder_layer_dim, y_dim, **kwargs):
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

        if target_y:
            prediction_var_reshaped = prediction_var.reshape((prediction_var.shape[0]* prediction_var.shape[1],1))
            cov_mat = torch.zeros((prediction_var_reshaped.shape[0], prediction_var_reshaped[0]))
            torc
        

        

    