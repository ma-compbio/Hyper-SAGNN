## Hyper-SAGNN: a self-attention based graph neural network for hypergraphs 

This is an implementation of "Hyper-SAGNN: a self-attention based graph neural network for hypergraphs" (ICLR2020)

The datasets included in this repo are originally from DHNE (https://github.com/tadpole/DHNE)

## Requirements

python >= 3.6.8

Tensorflow >= 1.0.0 (< 2.0.0)

Pytorch >= 1.0



## Usage

To run the code:

```
cd Code
python main.py --data wordnet -f adj
```



Change the following arguments to reproduce corresponding results from the manuscript,

The **--data** argument can take "GPS", "drug", "MovieLens", "wordnet". This argument is case sensitive

The **-f, --feature** argument can take "adj" or "walk" represents encoder based approach and random walk based approach respectively.



Other arguments are as followed:

```
parser.add_argument('--dimensions', type=int, default=64,
                    help='Number of dimensions. Default is 64.')

parser.add_argument('-l', '--walk-length', type=int, default=40,
                    help='Length of walk per source. Default is 40.')

parser.add_argument('-r', '--num-walks', type=int, default=10,
                    help='Number of walks per source. Default is 10.')

parser.add_argument('-k', '--window-size', type=int, default=10,
                    help='Context size for optimization. Default is 10.')
```



## Cite

If you want to cite our paper:

```
@article{zhang2019hyper,
  title={Hyper-SAGNN: a self-attention based graph neural network for hypergraphs},
  author={Zhang, Ruochi and Zou, Yuesong and Ma, Jian},
  journal={arXiv preprint arXiv:1911.02613},
  year={2019}
}
```