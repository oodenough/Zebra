Zebra: When Temporal Graph Neural Networks Meet Temporal Personalized PageRank
=============================================================================

## Dataset
6 datasets were used in this paper:

- Wikipedia: downloadable from http://snap.stanford.edu/jodie/.
- Reddit: downloadable from http://snap.stanford.edu/jodie/.
- MOOC: downloadable from http://snap.stanford.edu/jodie/.
- AskUbuntu: downloadable from http://snap.stanford.edu/data/sx-askubuntu.html.
- SuperUser: downloadable from http://snap.stanford.edu/data/sx-superuser.html.
- Wiki-Talk: downloadable from http://snap.stanford.edu/data/wiki-talk-temporal.html.

## Preprocessing
If edge features or nodes features are absent, they will be replaced by a vector of zeros. Example usage:
```sh
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_custom_data.py --data superuser
```


## Usage
```sh
Optional arguments:
    --data                  Dataset name
    --bs                    Batch size
    --n_head                Number of attention heads used in neighborhood aggregation
    --n_epoch               Number of training epochs
    --n_layer               Number of network layers
    --lr                    Learning rate
    --gpu                   GPU id
    --patience              Patience for early stopping
    --enable_random         Use random seeds
    --topk                  Top-k threshold
    --tppr_strategy         Strategy used for answering top-k T-PPR query [streaming|pruning]
    --alpha_list            Alpha values used in T-PPR metrics
    --beta_list             Beta values used in T-PPR metrics
    
Example usage:
    python train.py --n_epoch 50 --bs 200 --data wikipedia --enable_random  --tppr_strategy streaming  --topk 20 --alpha_list 0.1 0.1 --beta_list 0.5 0.95 --gpu 0
```



## FileNotFoundError

```bash
INFO:root:Namespace(data='wikipedia', bs=200, n_degree=10, n_head=2, n_epoch=50, n_layer=2, lr=0.0001, patience=5, n_runs=1, drop_out=0.3, gpu=0, use_memory=True, use_destination_embedding_in_message=False, use_source_embedding_in_message=False, message_function='identity', memory_updater='gru', embedding_module='diffusion', enable_random=True, aggregator='last', save_best=False, tppr_strategy='streaming', topk=20, alpha_list=[0.1, 0.1], beta_list=[0.5, 0.95], ignore_edge_feats=False, ignore_node_feats=False, node_dim=100, time_dim=100, memory_dim=100)
./saved_checkpoints/1714142065.540677.pth
wikipedia_streaming_topk_20_alpha_[0.1, 0.1]_beta_[0.5, 0.95]_bs_200_layer_2_epoch_50_lr_0.0001_random_seed
Traceback (most recent call last):
  File "/Users/saul/Projects/Zebra/train.py", line 123, in <module>
    full_data, full_train_data, full_val_data, test_data, new_node_val_data, new_node_test_data, n_nodes, n_edges = get_data(DATA)
                                                                                                                    ^^^^^^^^^^^^^^
  File "/Users/saul/Projects/Zebra/utils/data_processing.py", line 81, in get_data
    graph_df = pd.read_csv('../data/{}/ml_{}.csv'.format(dataset_name,dataset_name))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/saul/Projects/Zebra/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/saul/Projects/Zebra/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/saul/Projects/Zebra/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/saul/Projects/Zebra/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ^^^^^^^^^^^
  File "/Users/saul/Projects/Zebra/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
             ^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '../data/wikipedia/ml_wikipedia.csv'
```

fixed in commit `f87dabd`

## TypeError

```bash
INFO:root:Namespace(data='wikipedia', bs=200, n_degree=10, n_head=2, n_epoch=50, n_layer=2, lr=0.0001, patience=5, n_runs=1, drop_out=0.3, gpu=0, use_memory=True, use_destination_embedding_in_message=False, use_source_embedding_in_message=False, message_function='identity', memory_updater='gru', embedding_module='diffusion', enable_random=True, aggregator='last', save_best=False, tppr_strategy='streaming', topk=20, alpha_list=[0.1, 0.1], beta_list=[0.5, 0.95], ignore_edge_feats=False, ignore_node_feats=False, node_dim=100, time_dim=100, memory_dim=100)
./saved_checkpoints/1714143308.62215.pth
wikipedia_streaming_topk_20_alpha_[0.1, 0.1]_beta_[0.5, 0.95]_bs_200_layer_2_epoch_50_lr_0.0001_random_seed
Traceback (most recent call last):
  File "/Users/saul/Projects/Zebra/train.py", line 123, in <module>
    full_data, full_train_data, full_val_data, test_data, new_node_val_data, new_node_test_data, n_nodes, n_edges = get_data(DATA)
                                                                                                                    ^^^^^^^^^^^^^^
  File "/Users/saul/Projects/Zebra/utils/data_processing.py", line 102, in get_data
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/saul/.pyenv/versions/3.12.0/lib/python3.12/random.py", line 413, in sample
    raise TypeError("Population must be a sequence.  "
TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).
```

fixed in `bcaf5f8`


