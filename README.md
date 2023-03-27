# Rebalancing and Node Placement in PCNs

This repository was made to study and test different node placement and rebalancing techniques in payment channel networks. This is an ongoing work.

## Requirements

This code uses snapshots from the Lightning Network to test different strategies. Currently, part of the code related to the node placement considers an undirected graph and the part of the code related to rebalancing considers a directed graph. As this is an ongoing project, this will be corrected in the future. On the same note, part of the LN snapshot is available in the repository under the directory `ln-snapshot`. However, the snapshots used in the rebalancing part is not available because of their size. Users may download the rest of the snapshots [here](https://gta.ufrj.br/~gabriel/files/ln-graphs.tar.gz) or by cloning and running the code present in [here](https://github.com/gfrebello/topology). All snapshots were obtained from the [lnresearch repository](https://github.com/lnresearch/topology).

The code requires some Python3 libraries as well. They can be installed by running the following command in the root repository directory:

    pip install -r requirements.txt

The program also requires two datasets which are sampled to simulate payments. The first one is a credit-card dataset that can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The second one is a Ripple dataset available [here](https://crysp.uwaterloo.ca/software/speedymurmurs/download.php). Both datasets must be downloaded and moved to a directory called `datasets` on the root repository directory.

## Running LN-Looprebalance

LN-Looprebalance allows multiple simulation scenarios. To view simulation options, run the following command:

    python3 get-results.py --help

