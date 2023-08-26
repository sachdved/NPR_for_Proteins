import torch
import numpy as np
import sklearn.utils
import sklearn.gaussian_process


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

        X = torch.arange(start = 0, end = self.data.shape[1])

        Y = self.data[index]

        one_hot_Y = torch.tensor(self._to_one_hot(Y))
        return X, one_hot_Y

    def _to_one_hot(self, seq):
        one_hot_encoded = np.zeros((seq.shape[0],len(self.IDX_TO_AA)))

        for index, char in enumerate(seq):
            one_hot_encoded[index, self.AA_TO_IDX[char]]=1
        return one_hot_encoded
        

def context_target_splitter(batch, min_context, max_context, len_seq, x_dim, y_dim):
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

    X = X.squeeze()

    X_context = torch.zeros(size=(X.shape[0], num_context, x_dim))
    Y_context = torch.zeros(size=(Y.shape[0], num_context, y_dim))

    
    X_target = torch.zeros(size=(X.shape[0], len_seq, x_dim))
    Y_target = torch.zeros(size=(Y.shape[0], len_seq, y_dim))
    
    

    for index in range(Y.shape[0]):
        seq = Y[index]

        
        shuffled_indices = torch.randperm(len_seq)
    
        context_indices = shuffled_indices[:num_context]
        target_indices = shuffled_indices
        
        X_context[index] = X[index,context_indices].unsqueeze(-1)
        X_target[index]  = X[index, target_indices].unsqueeze(-1)
        
        Y_context[index] = seq[context_indices]
        Y_target[index] = seq[target_indices]

    return (((X_context, Y_context), X_target), Y_target)


class One_D_Datasets(torch.utils.data.Dataset):
    """
    Dataset of one-dimensional functions, generated from varying gaussian processes for testing my NPR methods

    Parameters
    ----------
    num_samples: how many samples from family to collect

    n_same_samples: how many samples to collect with the same hyper-parameters

    n_points: how many points to to sample the process at within range, X

    min_max: tuple, min_max x range to evaluate the gaussian process at

    kernel: kernels for gaussian process regression. Can be ['RBF()', 'ConstantKernel()', 'DotProduct()', 'ExpSineSquared()', 'Matern()', 'WhiteKernel()']

    vary_kernel_hyp: do we vary kernel hyperparameters or nah?
    
    **kwargs: additional arguments to GaussianProcessRegressor
    """
    def __init__(self,
                 num_samples = 10000,
                 n_same_samples = 20,
                 n_points =      128,
                 min_max  =   (-2,2),
                 kernel   = sklearn.gaussian_process.kernels.RBF(length_scale = 0.4, length_scale_bounds = (0.01,1)),
                 vary_kernel_hyp = True,
                 **kwargs
    ):
        super().__init__()
        self.n_points = n_points
        self.num_samples = num_samples
        self.n_same_samples = n_same_samples
        self.min_max = min_max
        self.is_vary_kernel_hyp = vary_kernel_hyp

        if not vary_kernel_hyp:
            # only fit hyperparam when predicting if using various hyperparam
            kwargs["optimizer"] = None

            # we also fix the bounds as these will not be needed
            for hyperparam in kernel.hyperparameters:
                kernel.set_params(**{f"{hyperparam.name}_bounds": "fixed"})

        self.generator = sklearn.gaussian_process.GaussianProcessRegressor(kernel = kernel, alpha=0.005, **kwargs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def vary_kernel_hyperparam(self):
        K = self.generator.kernel
        for hyperparam in K.hyperparameters:
            K.set_params(
                **{hyperparam.name: np.random.uniform(*hyperparam.bounds.squeeze())}
            )

    def get_samples(
        self,
        n_samples,
        test_min_max,
        n_points
    ):
        test_min_max = test_min_max if test_min_max is not None else self.min_max
        n_points = n_points if n_points is not None else self.n_points
        n_samples = n_samples if n_samples is not None else self.num_samples
        """
        n_samples: number of samples to generate data for
        test_min_max: range of x values to generate sample for
        n_points: number of points to generate 
        """
        X = self._sample_features(test_min_max, n_points, n_samples)

        self.data, self.targets = self._sample_targets(X, n_samples)

        return self.data, self.targets
        

    def _sample_features(self, min_max, n_points, n_samples):
        X = np.random.uniform(min_max[1], min_max[0], size=(n_samples, n_points))
        X.sort(axis=-1)
        return X

    def _sample_targets(self, X, n_samples):
        targets = X.copy()
        n_samples, n_points = X.shape
        for i in range(0, n_samples, self.n_same_samples):
            if self.is_vary_kernel_hyp:
                self.vary_kernel_hyperparam()
            for attempt in range(self.n_same_samples):
                try:
                    n_same_samples = targets[i:i+self.n_same_samples,:].shape[0]
                    targets[i:i+self.n_same_samples,:] = self.generator.sample_y(
                        X[i+attempt,:,np.newaxis],
                        n_samples = n_same_samples,
                        random_state = None,
                    ).transpose(1,0)
                    X[i:i+self.n_same_samples,:] = X[i+attempt,:]
                except np.linalg.LinAlgError:
                    continue
                else:
                    break
        X, targets = sklearn.utils.shuffle(X, targets)
        targets = torch.from_numpy(targets)
        targets = targets.view(n_samples, n_points, 1).float()
        return X, targets
            
        
            
                 
