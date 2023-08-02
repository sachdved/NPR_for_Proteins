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

        cross_entropy = torch.mean(torch.sum(-target_y * torch.log(output + 1e-6) - (1-target_y) * torch.log(1 - output + 1e-6), dim=1))
        return cross_entropy



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
            print('got here')
            print(query.shape)
            print(keys.shape)
            print(encoder_input.shape)
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

        cross_entropy = torch.mean(torch.sum(-target_y * torch.log(output + 1e-6) - (1-target_y) * torch.log(1 - output + 1e-6), dim=1))
        return cross_entropy
        
        