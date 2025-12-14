import random
import numpy as np
import torch

def set_seed(seed):
    """
    More robust CUDA-safe seed setup.
    Uses the exact logic from your original script.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()        # ensure CUDA context exists
            torch.cuda.empty_cache()           # clear cache
            torch.tensor([0.], device='cuda')  # force GPU initialization
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except RuntimeError as e:
            print(f"[WARNING] CUDA seed setup failed: {e}")


def get_device(prefer_gpu: bool = True):
    """
    Returns CUDA if available and prefer_gpu=True, else CPU.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def findDuplicated(df):
    """
    Identifies and removes columns (genes) with duplicate names in a DataFrame.
    Keeps the first occurrence of a duplicated gene name (case-insensitive).
    """
    df_transposed = df.T
    gene_names_upper = df_transposed.index.str.upper()
    is_duplicated = gene_names_upper.duplicated(keep='first')
    df_unique_genes_transposed = df_transposed[~is_duplicated]
    df_final = df_unique_genes_transposed.T
    return df_final


def high_var_npdata(data_np, num_to_keep, gene_names_index=None, return_indices=False):
    """
    Selects the top 'num_to_keep' genes (columns) with the highest variance
    from a numpy array (cells x genes).
    """
    datavar = np.var(data_np, axis=0) * (-1)
    ind_maxvar = np.argsort(datavar)
    gene_indices_high_var = ind_maxvar[:num_to_keep]
    gene_indices_high_var.sort()

    results = [gene_indices_high_var]
    if gene_names_index is not None:
        selected_gene_names = gene_names_index[gene_indices_high_var]
        results.append(selected_gene_names)

    if len(results) == 1:
        return results[0]
    return tuple(results)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def process_features_and_adj(features_df, full_adj_sparse, gene_indices_to_use, max_val_for_scaling):
    """
    Subsets features and adjacency matrix based on pre-selected gene indices.
    Performs log1p and max-scaling on the features.
    """
    features_np = np.asarray(features_df).astype(np.float32)
    subset_features_np = features_np[:, gene_indices_to_use]
    transformed_features = np.log1p(subset_features_np)
    if max_val_for_scaling > 0:
        scaled_features = transformed_features / max_val_for_scaling
    else:
        scaled_features = transformed_features
    final_adj_sparse = full_adj_sparse[gene_indices_to_use, :][:, gene_indices_to_use]

    return scaled_features, final_adj_sparse


