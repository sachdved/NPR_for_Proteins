import torch
import numpy as np

class ProteinDataset(torch.utils.data.Dataset):
    """Dataset structure for protein data. Offers one hot encoding from sequence data
    input:
        data: np.array of sequences, with data[i,j] is the jth amino acid of the ith sequence
    output:
        X: indices of positions
        one_hot_Y: torch.tensor of size (length of sequence, number of amino acids)
    """
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data

        self.AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV-"
        self.IDX_TO_AA = list(self.AMINO_ACIDS)
        self.AA_TO_IDX = {aa: i for i, aa in enumerate(self.IDX_TO_AA)}

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        X = torch.unsqueeze(torch.arange(start = 0, end = self.data.shape[1]),-1)

        Y = self.data[index]

        one_hot_Y = torch.tensor(self._to_one_hot(Y))
        return X, one_hot_Y

    def _to_one_hot(self, seq):
        one_hot_encoded = np.zeros((seq.shape[0],len(self.IDX_TO_AA)))

        for index, char in enumerate(seq):
            one_hot_encoded[index, self.AA_TO_IDX[char]]=1
        return one_hot_encoded
        

def context_target_splitter(batch, min_context, max_context, len_seq):
    """Splits batch into context and target splits
    input:
        batch: generated from pytorch DataLoader
        min_context: minimum number of context points
        max_context: maximum number of context points
        len_seq: length of sequence

    output:
        (((X_context, Y_context), X_target), Y_target)
    """
    num_context = torch.randint(low=min_context, high=max_context, size=[1])

    X, Y = batch

    X_context = torch.zeros(size=(X.shape[0], num_context, X.shape[-1]))
    Y_context = torch.zeros(size=(Y.shape[0], num_context, Y.shape[-1]))

    
    X_target = torch.zeros(size=(X.shape[0], len_seq - num_context, X.shape[-1]))
    Y_target = torch.zeros(size=(Y.shape[0], len_seq - num_context, Y.shape[-1]))
    
    

    for index in range(Y.shape[0]):
        seq = Y[index]

        
        shuffled_indices = torch.randperm(len_seq)
    
        context_indices = shuffled_indices[:num_context]
        target_indices = shuffled_indices[num_context:]
        
        X_context[index] = context_indices.unsqueeze(-1)
        X_target[index]  = target_indices.unsqueeze(-1)
        
        Y_context[index] = seq[context_indices]
        Y_target[index] = seq[target_indices]

    return (((X_context, Y_context), X_target), Y_target)
