import numpy as np
import torch
from torch_geometric.datasets import DBLP
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from sklearn.model_selection import train_test_split
import random

from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import SVDEncodingsGraphDataset
from ..graph_dataset import StructuralDataset


class DBLPDataset(DatasetBase):
    def __init__(self,
                 dataset_path,
                 dataset_name='DBLP',
                 subset_size=50,
                 metapath='APA',
                 **kwargs
                 ):
        super().__init__(dataset_name=dataset_name,
                         **kwargs)
        self.dataset_path = dataset_path
        self.subset_size = subset_size
        self.metapath = metapath

    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            self._dataset = DBLP(root=self.dataset_path)
            return self._dataset

    def sample_subset(self, data, sample_size, seed=None):
        # Campiona un sottoinsieme di autori
        random.seed(seed)
        authors = data['author'].num_nodes
        sampled_authors = random.sample(range(authors), sample_size)
        
        mask = torch.zeros(authors, dtype=torch.bool)
        mask[sampled_authors] = True
        
        return mask

    def create_apa_graph(self, data, mask):
        # Crea il grafo APA utilizzando matrici sparse
        ap_edges = data['author', 'paper'].edge_index
        
        # Filtra gli edge per includere solo gli autori selezionati
        author_mask = mask[ap_edges[0]]
        ap_edges = ap_edges[:, author_mask]

        num_authors = data['author'].num_nodes
        num_papers = data['paper'].num_nodes

        ap_adj = torch.sparse_coo_tensor(ap_edges, torch.ones(ap_edges.size(1)),
                                         (num_authors, num_papers))
        pa_adj = torch.sparse_coo_tensor(ap_edges.flip(0), torch.ones(ap_edges.size(1)),
                                         (num_papers, num_authors))

        apa_adj = torch.sparse.mm(ap_adj, pa_adj)
        apa_edge_index = apa_adj.coalesce().indices()

        # Rimuovi i self-loops
        apa_edge_index = apa_edge_index[:, apa_edge_index[0] != apa_edge_index[1]]

        return apa_edge_index

    def create_aptpa_graph(self, data, mask):
        # Crea il grafo APTPA utilizzando matrici sparse
        ap_edges = data['author', 'paper'].edge_index
        pt_edges = data['paper', 'term'].edge_index

        # Filtra gli edge per includere solo gli autori selezionati
        author_mask = mask[ap_edges[0]]
        ap_edges = ap_edges[:, author_mask]

        num_authors = data['author'].num_nodes
        num_papers = data['paper'].num_nodes
        num_terms = data['term'].num_nodes

        ap_adj = torch.sparse_coo_tensor(ap_edges, torch.ones(ap_edges.size(1)),
                                         (num_authors, num_papers))
        pt_adj = torch.sparse_coo_tensor(pt_edges, torch.ones(pt_edges.size(1)),
                                         (num_papers, num_terms))
        tp_adj = torch.sparse_coo_tensor(pt_edges.flip(0), torch.ones(pt_edges.size(1)),
                                         (num_terms, num_papers))
        pa_adj = torch.sparse_coo_tensor(ap_edges.flip(0), torch.ones(ap_edges.size(1)),
                                         (num_papers, num_authors))

        aptpa_adj = torch.sparse.mm(ap_adj, torch.sparse.mm(pt_adj, torch.sparse.mm(tp_adj, pa_adj)))
        aptpa_edge_index = aptpa_adj.coalesce().indices()

        # Rimuovi i self-loops
        aptpa_edge_index = aptpa_edge_index[:, aptpa_edge_index[0] != aptpa_edge_index[1]]

        return aptpa_edge_index

    def create_apcpa_graph(self, data, mask):
        # Crea il grafo APCPA utilizzando matrici sparse
        ap_edges = data['author', 'paper'].edge_index
        cp_edges = data['conference', 'paper'].edge_index

        # Filtra gli edge per includere solo gli autori selezionati
        author_mask = mask[ap_edges[0]]
        ap_edges = ap_edges[:, author_mask]

        num_authors = data['author'].num_nodes
        num_papers = data['paper'].num_nodes
        num_conferences = data['conference'].num_nodes

        ap_adj = torch.sparse_coo_tensor(ap_edges, torch.ones(ap_edges.size(1)),
                                         (num_authors, num_papers))
        pc_adj = torch.sparse_coo_tensor(cp_edges.flip(0), torch.ones(cp_edges.size(1)),
                                         (num_papers, num_conferences))
        cp_adj = torch.sparse_coo_tensor(cp_edges, torch.ones(cp_edges.size(1)),
                                         (num_conferences, num_papers))
        pa_adj = torch.sparse_coo_tensor(ap_edges.flip(0), torch.ones(ap_edges.size(1)),
                                         (num_papers, num_authors))

        apcpa_adj = torch.sparse.mm(ap_adj, torch.sparse.mm(pc_adj, torch.sparse.mm(cp_adj, pa_adj)))
        apcpa_edge_index = apcpa_adj.coalesce().indices()

        # Rimuovi i self-loops
        apcpa_edge_index = apcpa_edge_index[:, apcpa_edge_index[0] != apcpa_edge_index[1]]

        return apcpa_edge_index

    def create_metapath_graph(self, data):
        # Crea il grafo del metapath specificato
        mask = self.sample_subset(data, self.subset_size, seed=42)
        
        if self.metapath == 'APA':
            edge_index = self.create_apa_graph(data, mask)
        elif self.metapath == 'APTPA':
            edge_index = self.create_aptpa_graph(data, mask)
        elif self.metapath == 'APCPA':
            edge_index = self.create_apcpa_graph(data, mask)
        else:
            raise ValueError(f"Metapath {self.metapath} non supportato")

        return Data(edge_index=to_undirected(edge_index), num_nodes=self.subset_size)

    @property
    def record_tokens(self):
        # Genera i token per i record del dataset
        try:
            return self._record_tokens
        except AttributeError:
            split = {'training': 'train',
                     'validation': 'val',
                     'test': 'test'}[self.split]

            data = self.dataset[0]
            metapath_data = self.create_metapath_graph(data)

            # Crea gli split per train, validation e test
            edge_index = metapath_data.edge_index.T.numpy()
            train_val_edges, test_edges = train_test_split(edge_index, test_size=0.2, random_state=42)
            train_edges, val_edges = train_test_split(train_val_edges, test_size=0.1, random_state=42)

            train_mask = np.zeros(edge_index.shape[0], dtype=bool)
            val_mask = np.zeros(edge_index.shape[0], dtype=bool)
            test_mask = np.zeros(edge_index.shape[0], dtype=bool)

            train_mask[np.isin(edge_index, train_edges).all(axis=1)] = True
            val_mask[np.isin(edge_index, val_edges).all(axis=1)] = True
            test_mask[np.isin(edge_index, test_edges).all(axis=1)] = True

            metapath_data.train_mask = torch.tensor(train_mask)
            metapath_data.val_mask = torch.tensor(val_mask)
            metapath_data.test_mask = torch.tensor(test_mask)

            if split == 'train':
                self._record_tokens = metapath_data.train_mask.nonzero(as_tuple=False).view(-1)
            elif split == 'val':
                self._record_tokens = metapath_data.val_mask.nonzero(as_tuple=False).view(-1)
            elif split == 'test':
                self._record_tokens = metapath_data.test_mask.nonzero(as_tuple=False).view(-1)

            return self._record_tokens

    def read_record(self, token):
        # Legge un record specifico dal dataset
        data = self.dataset[0]
        metapath_data = self.create_metapath_graph(data)
        
        num_nodes = self.subset_size
        node_features = data['author'].x[:self.subset_size]
        target = data['author'].y[:self.subset_size]

        graph = {
            'num_nodes': np.array(num_nodes, dtype=np.int16),
            'edges': metapath_data.edge_index.T.to(torch.int16).numpy(),
            'edge_features': metapath_data.edge_attr.to(torch.int16).numpy() if metapath_data.edge_attr is not None else np.array([]),
            'node_features': node_features.to(torch.int16).numpy() if node_features is not None else np.array([]),
            'target': target.to(torch.float32).numpy()
        }
        return graph

# Le classi derivate rimangono le stesse
class DBLPGraphDataset(GraphDataset, DBLPDataset):
    pass

class DBLPSVDGraphDataset(SVDEncodingsGraphDataset, DBLPDataset):
    pass

class DBLPStructuralGraphDataset(StructuralDataset, DBLPGraphDataset):
    pass

class DBLPStructuralSVDGraphDataset(StructuralDataset, DBLPSVDGraphDataset):
    pass