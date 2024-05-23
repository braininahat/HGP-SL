from torch_geometric.data import InMemoryDataset
import torch
from typing import List, Optional, Callable


class BreastMAPDataset(InMemoryDataset):
    """
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk. (default: :obj:`None`)
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ) -> None:
        self.name = name
        super(BreastMAPDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        names = ["A", "graph_indicator"]
        return [f"{self.name}_{name}.txt" for name in names]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    def process(self):
        data = torch.load(self.raw_paths[0])
        data = data if isinstance(data, list) else [data]
        data = [data] if isinstance(data[0], Data) else data

        data = [self.collate(data)]
        torch.save(self.collate(data), self.processed_paths[0])

    def collate(self, data_list):
        data = Batch.from_data_list(data_list)
        data.y = data.y.view(-1)
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
