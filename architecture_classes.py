class MLP(torch.nn.Module):
    def __init__(self, 
        input_size,
        output_sizes,
        **kwargs
    ):
        super(MLP, self).__init__(**kwargs)
        self.input_size = input_size
        self.output_sizes = output_sizes
        
        self.mlp = torch.nn.Sequential()

        self.mlp.add_module('input_layer', torch.nn.Linear(input_size, output_sizes[0]))
        self.mlp.add_module('relu', torch.nn.ReLU())

        for index, output in range(1, output_sizes):
            self.mlp.add_module('hidden_layer_{}'.format(index), torch.nn.Linear(output_sizes[index-1], output_sizes[index]))
            self.mlp.add_module('relu_{}'.format(index+1), torch.nn.ReLU())


    def forward(self, x):
        assert x.shape[-1] == self.input_size, "Input to MLP not the correct dimension"

        return self.mlp(x)


class DotProductAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, queries, keys, values, d_k, mask=None):
        scores = torch.matmul(queries, keys.transpose(-1,-2))/torch.sqrt(d_k)
        if mask is not None:
            scores += -1e9*mask
        attention = torch.nn.Softmax(dim=-1)(scores)

        return torch.matmul(attention, values), attention

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, 
                 h,
                 d_input, 
                 d_k, 
                 d_model,
                 activation = torch.nn.ReLU,
                 **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()
        self.heads = h
        self.d_k = d_k
        self.d_input = d_input
        self.W_q = torch.nn.Linear(d_input, d_k*h)
        self.W_k = torch.nn.Linear(d_input, d_k*h)
        self.W_v = torch.nn.Linear(d_input, d_k*h)
        self.W_o = torch.nn.Linear(d_k, d_model)     

    def reshape_tensor(self, x, heads, flag):
        if flag:
            x = torch.reshape(x, shape = (x.shape[0], x.shape[1], heads, self.d_k))
            x = x.permute(0,2,1,3)
        else:
            x = x.transpose(-1,-2)
            x = torch.reshape(x, shape = (x.shape[0], x.shape[1], heads*self.d_k))
        return x

    def forward(self, queries, keys, values, mask = None):
        n_context = queries.shape[1]
        n_target = keys.shape[1]

        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)

        
        output, attention = self.attention(q_reshaped, k_reshaped, v_reshaped, torch.tensor(self.d_k), mask)

        return self.W_o(output), output    