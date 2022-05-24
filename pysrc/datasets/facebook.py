import os
import ssl
import torch
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import subgraph, from_scipy_sparse_matrix
from torch_geometric.data import Data, InMemoryDataset, download_url


class Facebook(InMemoryDataset):
    url = 'https://github.com/sisaman/pyg-datasets/raw/main/datasets/facebook100/'
    targets = ['status', 'gender', 'major', 'minor', 'housing', 'year']
    available_datasets = [
        "Villanova62", "UCLA26", "Tennessee95", "NYU9", "Carnegie49", "GWU54", "USF51", "Vanderbilt48", 
        "USC35", "Rutgers89", "UConn91", "MIT8", "USFCA72", "UChicago30", "UIllinois20", "UC61", "Cal65", 
        "Yale4", "Northeastern19", "Dartmouth6", "Vermont70", "Northwestern25", "William77", "Harvard1", 
        "Princeton12", "UC64", "Middlebury45", "Haverford76", "Bingham82", "UNC28", "Berkeley13", "Rochester38", 
        "Swarthmore42", "Virginia63", "WashU32", "Columbia2", "NotreDame57", "Bucknell39", "UVA16", "Maine59", 
        "MU78", "Simmons81", "MSU24", "Colgate88", "Temple83", "Cornell5", "Indiana69", "Oklahoma97", "Michigan23", 
        "BU10", "Brown11", "Auburn71", "FSU53", "UGA50", "UCF52", "Howard90", "UCSD34", "Vassar85", "Tufts18", 
        "UPenn7", "Baylor93", "UMass92", "Bowdoin47", "Maryland58", "Penn94", "Wesleyan43", "UC33", 
        "Rice31", "UCSC68", "Smith60", "Caltech36", "Hamilton46", "Oberlin44", "American75", "Mich67", 
        "Mississippi66", "Williams40", "UCSB37", "Amherst41", "Duke14", "Pepperdine86", "Wake73", "Lehigh96", 
        "Reed98", "Tulane29", "Texas84", "Wellesley22", "JMU79", "Santa74", "Wisconsin87", "Stanford3", 
        "Texas80", "UF21", "JohnsHopkins55", "Syracuse56", "BC17", "Georgetown15", "Trinity100", "Brandeis99", "Emory27"
    ]

    def __init__(self, root: str, name: str, target='status', transform=None, pre_transform=None):
        self.name = name
        self.target = target
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        return self.name + '.mat'

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'data-{self.name}-{self.target}.pt'

    def download(self):
        context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        download_url(f'{self.url}/{self.raw_file_names}', self.raw_dir)
        ssl._create_default_https_context = context

    def process(self):
        mat = loadmat(os.path.join(self.raw_dir, self.raw_file_names))
        features = pd.DataFrame(mat['local_info'][:, :-1], columns=self.targets)
        y = torch.from_numpy(LabelEncoder().fit_transform(features[self.target]))
        if 0 in features[self.target].values:
            y = y - 1

        x = features.drop(columns=self.target).replace({0: pd.NA})
        x = torch.tensor(pd.get_dummies(x).values, dtype=torch.float)
        edge_index = from_scipy_sparse_matrix(mat['A'])[0]

        # removed unlabled nodes
        subset = y >= 0
        edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=len(y))
        x = x[subset]
        y = y[subset]

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'Facebook100-{self.name}()'

