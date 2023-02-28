from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import (
    filter_data,
    filter_feature_store,
    filter_hetero_data,
)
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor


class LinkNeighborLoader(torch.utils.data.DataLoader):
    r"""A link-based data loader derived as an extension of the node-based
    :class:`torch_geometric.loader.NeighborLoader`.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, this loader first selects a sample of edges from the
    set of input edges :obj:`edge_label_index` (which may or not be edges in
    the original graph) and then constructs a subgraph from all the nodes
    present in this list by sampling :obj:`num_neighbors` neighbors in each
    iteration.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import LinkNeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = LinkNeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            edge_label_index=data.edge_index,
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128])

    It is additionally possible to provide edge labels for sampled edges, which
    are then added to the batch:

    .. code-block:: python

        loader = LinkNeighborLoader(
            data,
            num_neighbors=[30] * 2,
            batch_size=128,
            edge_label_index=data.edge_index,
            edge_label=torch.ones(data.edge_index.size(1))
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128], edge_label=[128])

    The rest of the functionality mirrors that of
    :class:`~torch_geometric.loader.NeighborLoader`, including support for
    heterogenous graphs.

    .. note::
        :obj:`neg_sampling_ratio` is currently implemented in an approximate
        way, *i.e.* negative edges may contain false negatives.

        :obj:`time_attr` is currently implemented such that for an edge
        `(src_node, dst_node)`, the neighbors of `src_node` can have a later
        timestamp than `dst_node` or vice-versa.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices for which neighbors are sampled to create
            mini-batches.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)
        edge_label (Tensor): The labels of edge indices for which neighbors are
            sampled. Must be the same length as the :obj:`edge_label_index`.
            If set to :obj:`None` then no labels are returned in the batch.
        num_src_nodes (optional, int): The number of source nodes in the edge
            label index. Inferred if not provided.
        num_dst_nodes (optional, int): The number of destination nodes in the
            edge label index. Inferred if not provided.
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges.
            If :obj:`edge_label` does not exist, it will be automatically
            created and represents a binary classification task
            (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`edge_label` exists, it has to be a categorical label from
            :obj:`0` to :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges.
            Note that returned labels are of type :obj:`torch.float` for binary
            classification (to facilitate the ease-of-use of
            :meth:`F.binary_cross_entropy`) and of type
            :obj:`torch.long` for multi-class classification (to facilitate the
            ease-of-use of :meth:`F.cross_entropy`). (default: :obj:`0.0`).
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier timestamp than the center node. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column. This avoids internal
            re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` is generally not recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down data loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighbors,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        num_src_nodes: Optional[int] = None,
        num_dst_nodes: Optional[int] = None,
        replace: bool = False,
        directed: bool = True,
        neg_sampling_ratio: float = 0.0,
        time_attr: Optional[str] = None,
        transform: Callable = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        neighbor_sampler: Optional[LinkNeighborSampler] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.replace = replace
        self.directed = directed
        self.neg_sampling_ratio = neg_sampling_ratio
        self.transform = transform
        self.filter_per_worker = filter_per_worker
        self.neighbor_sampler = neighbor_sampler

        edge_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)

        if neighbor_sampler is None:
            self.neighbor_sampler = LinkNeighborSampler(
                data,
                num_neighbors,
                replace,
                directed,
                input_type=edge_type,
                is_sorted=is_sorted,
                neg_sampling_ratio=self.neg_sampling_ratio,
                num_src_nodes=num_src_nodes,
                num_dst_nodes=num_dst_nodes,
                time_attr=time_attr,
                share_memory=kwargs.get('num_workers', 0) > 0,
            )

        super().__init__(Dataset(edge_label_index, edge_label),
                         collate_fn=self.collate_fn, **kwargs)

    def filter_fn(self, out: Any) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            (node, row, col, edge, edge_label_index, edge_label) = out
            data = filter_data(self.data, node, row, col, edge,
                               self.neighbor_sampler.perm)
            data.edge_label_index = edge_label_index
            if edge_label is not None:
                data.edge_label = edge_label

        elif isinstance(self.data, HeteroData):
            (node_dict, row_dict, col_dict, edge_dict, edge_label_index,
             edge_label) = out
            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                      edge_dict,
                                      self.neighbor_sampler.perm_dict)
            edge_type = self.neighbor_sampler.input_type
            data[edge_type].edge_label_index = edge_label_index
            if edge_label is not None:
                data[edge_type].edge_label = edge_label
        else:
            (node_dict, row_dict, col_dict, edge_dict, edge_label_index,
             edge_label) = out
            feature_store, _ = self.data
            data = filter_feature_store(feature_store, node_dict, row_dict,
                                        col_dict, edge_dict)
            edge_type = self.neighbor_sampler.input_type
            data[edge_type].edge_label_index = edge_label_index
            if edge_label is not None:
                data[edge_type].edge_label = edge_label

        return data if self.transform is None else self.transform(data)

    def collate_fn(self, index: Union[List[int], Tensor]) -> Any:
        out = self.neighbor_sampler(index)
        if self.filter_per_worker:
            # We execute `filter_fn` in the worker process.
            out = self.filter_fn(out)
        return out

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        # We execute `filter_fn` in the main process.
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'