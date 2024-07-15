import numpy as np
import torch
from torch_geometric.datasets import IMDB
from torch_geometric.data import DataLoader
from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import SVDEncodingsGraphDataset
from ..graph_dataset import StructuralDataset


class IMDBDataset(DatasetBase):
    def __init__(self,
                 dataset_path,
                 dataset_name='IMDB',
                 subset_size=100,  # Aggiunto parametro subset_size
                 **kwargs):
        super().__init__(dataset_name=dataset_name,
                         **kwargs)
        self.dataset_path = dataset_path
        self.subset_size = subset_size  # Salva il parametro subset_size

    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            self._dataset = IMDB(root=self.dataset_path)
            return self._dataset

    @property
    def record_tokens(self):
        try:
            return self._record_tokens
        except AttributeError:
            split = {'training': 'train',
                     'validation': 'valid',
                     'test': 'test'}[self.split]
            num_samples = self.dataset[0].num_nodes
            if self.subset_size:  # Se subset_size è definito, usa solo un sottoinsieme dei nodi
                num_samples = min(self.subset_size, num_samples)
            if split == 'train':
                self._record_tokens = torch.arange(0, int(0.7 * num_samples))
            elif split == 'valid':
                self._record_tokens = torch.arange(int(0.7 * num_samples), int(0.85 * num_samples))
            elif split == 'test':
                self._record_tokens = torch.arange(int(0.85 * num_samples), num_samples)
            return self._record_tokens

    def read_record(self, token):
        data = self.dataset[0]
        movie_idx = token.numpy() if isinstance(token, torch.Tensor) else token
        mam_edge_index = self.extract_mam(data, movie_idx)

        if mam_edge_index.size == 0:
            print(f"No MAM edges found for movie index {movie_idx}")

        graph = {
            'num_nodes': np.array(data['movie'].num_nodes, dtype=np.int16),
            'edges': mam_edge_index.T.astype(np.int16) if mam_edge_index.size > 0 else np.empty((0, 2), dtype=np.int16),
            'edge_features': np.ones((mam_edge_index.shape[1], 1),
                                     dtype=np.int16) if mam_edge_index.size > 0 else np.empty((0, 1), dtype=np.int16),
            # dummy edge features
            'node_features': data['movie'].x[:,:5].numpy().astype(np.int16),
            'target': np.array(data['movie'].y[movie_idx], np.float32)
        }
        return graph

    def extract_mam(self, data, movie_idx):
        #print("Available edge types:", data.edge_index_dict.keys())
        # Verifica quali chiavi sono disponibili
        if ('movie', 'to', 'actor') in data.edge_index_dict and ('actor', 'to', 'movie') in data.edge_index_dict:
            movie_actor_edges = data['movie', 'to', 'actor'].edge_index
            actor_movie_edges = data['actor', 'to', 'movie'].edge_index
        else:
            print("Required edge types ('movie', 'to', 'actor') or ('actor', 'to', 'movie') not found in data")
            return np.array([], dtype=np.int64)

        mam_edge_index = []
        for movie_actor in movie_actor_edges.T:
            if movie_actor[0].numpy() == movie_idx:
                actor = movie_actor[1].numpy()
                for actor_movie in actor_movie_edges.T:
                    if actor_movie[0].numpy() == actor:
                        mam_edge_index.append([movie_actor[0], actor_movie[1]])

        if not mam_edge_index:
            print(f"No MAM edges found for movie index {movie_idx}")

        return np.array(mam_edge_index).T if mam_edge_index else np.array([], dtype=np.int64)


class IMDBGraphDataset(GraphDataset, IMDBDataset):
    pass


class IMDBSVDGraphDataset(SVDEncodingsGraphDataset, IMDBDataset):
    pass


class IMDBStructuralGraphDataset(StructuralDataset, IMDBGraphDataset):
    pass


class IMDBStructuralSVDGraphDataset(StructuralDataset, IMDBSVDGraphDataset):
    pass
