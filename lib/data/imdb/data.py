import numpy as np
import torch
from torch_geometric.datasets import IMDB
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from sklearn.model_selection import train_test_split
import numba as nb

from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import SVDEncodingsGraphDataset
from ..graph_dataset import StructuralDataset


class IMDBDataset(DatasetBase):
    def __init__(self,
                 dataset_path,
                 dataset_name='IMDB',
                 subset_size=50,
                 **kwargs
                 ):
        super().__init__(dataset_name=dataset_name,
                         **kwargs)
        self.dataset_path = dataset_path
        self.subset_size = subset_size

    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            self._dataset = IMDB(root=self.dataset_path)
            return self._dataset

    def create_mam_graph(self, data):
        if self.subset_size is None:
            subset_indices = torch.arange(data['movie'].num_nodes)
        else:
            subset_indices = torch.arange(self.subset_size)

        # Ottieni gli edge di tipo 'movie', 'actor' e 'actor', 'movie'
        ma_edges = data['movie', 'actor'].edge_index

        # Filtra gli edge per includere solo quelli che coinvolgono i film selezionati
        mask = torch.isin(ma_edges[0], subset_indices)
        ma_edges = ma_edges[:, mask]

        am_edges = ma_edges.flip(0)

        # Crea una matrice di adiacenza sparse per entrambi i tipi di edge
        ma_adj = torch.sparse_coo_tensor(ma_edges, torch.ones(ma_edges.size(1)),
                                         (subset_indices.size(0), data['actor'].num_nodes))
        am_adj = torch.sparse_coo_tensor(am_edges, torch.ones(am_edges.size(1)),
                                         (data['actor'].num_nodes, subset_indices.size(0)))

        # Moltiplica le due matrici di adiacenza per ottenere il grafo MAM
        mam_adj = torch.sparse.mm(ma_adj, am_adj)

        # Ottieni gli edge index del grafo MAM
        mam_edge_index = mam_adj.coalesce().indices()

        return Data(edge_index=to_undirected(mam_edge_index), num_nodes=subset_indices.size(0))

    @property
    def record_tokens(self):
        try:
            return self._record_tokens
        except AttributeError:
            split = {'training': 'train',
                     'validation': 'val',
                     'test': 'test'}[self.split]

            # Creiamo uno split manualmente
            data = self.dataset[0]

            # Crea il grafo MAM
            mam_data = self.create_mam_graph(data)

            # Suddividere manualmente gli edge in training, validation e test set
            edge_index = mam_data.edge_index.T.numpy()
            train_val_edges, test_edges = train_test_split(edge_index, test_size=0.2, random_state=42)
            train_edges, val_edges = train_test_split(train_val_edges, test_size=0.1, random_state=42)

            train_mask = np.zeros(edge_index.shape[0], dtype=bool)
            val_mask = np.zeros(edge_index.shape[0], dtype=bool)
            test_mask = np.zeros(edge_index.shape[0], dtype=bool)

            train_mask[np.isin(edge_index, train_edges).all(axis=1)] = True
            val_mask[np.isin(edge_index, val_edges).all(axis=1)] = True
            test_mask[np.isin(edge_index, test_edges).all(axis=1)] = True

            mam_data.train_mask = torch.tensor(train_mask)
            mam_data.val_mask = torch.tensor(val_mask)
            mam_data.test_mask = torch.tensor(test_mask)

            if split == 'train':
                self._record_tokens = mam_data.train_mask.nonzero(as_tuple=False).view(-1)
            elif split == 'val':
                self._record_tokens = mam_data.val_mask.nonzero(as_tuple=False).view(-1)
            elif split == 'test':
                self._record_tokens = mam_data.test_mask.nonzero(as_tuple=False).view(-1)

            return self._record_tokens

    def read_record(self, token):
        data = self.dataset[0]
        mam_data = self.create_mam_graph(data)
        if self.subset_size is None:
            num_nodes = data['movie'].num_nodes
            node_features = data['movie'].x
            target = data['movie'].y
        else:
            num_nodes = self.subset_size
            node_features = data['movie'].x[:self.subset_size]
            target = data['movie'].y[:self.subset_size]

        graph = {
            'num_nodes': np.array(num_nodes, dtype=np.int16),
            'edges': mam_data.edge_index.T.to(torch.int16).numpy(),
            'edge_features': mam_data.edge_attr.to(torch.int16).numpy() if mam_data.edge_attr is not None else np.array(
                []),
            'node_features': node_features.to(torch.int16).numpy() if node_features is not None else np.array([]),
            'target': target.to(torch.float32).numpy()
        }
        return graph


# Aggiorna anche la funzione calculate_svd_encodings se necessario
def calculate_svd_encodings(edges, num_nodes, calculated_dim):
    # Implementazione della funzione SVD
    pass


class IMDBGraphDataset(GraphDataset, IMDBDataset):
    pass


class IMDBSVDGraphDataset(SVDEncodingsGraphDataset, IMDBDataset):
    pass


class IMDBStructuralGraphDataset(StructuralDataset, IMDBGraphDataset):
    pass


class IMDBStructuralSVDGraphDataset(StructuralDataset, IMDBSVDGraphDataset):
    pass
