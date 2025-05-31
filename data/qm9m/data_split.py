"""
Save data_split.pkl

- Total # of data: 133,885
- train/val/test = 100,000 / 23,885 / 10,000
    (Basis: In QM-9 the dataset was randomly split to 100k molecules in the train set,
    with 10% in the test set, and the remaining in the validation set.
    We removed the 3054 molecules that failed consistency requirements [Ramakrishnan et al., 2014],
    [Anderson et al. 2019]; https://arxiv.org/abs/1906.04015)
"""

import pickle
import random

random.seed(42)

if __name__ == "__main__":
    # Generate data indices
    total_indices = list(range(133885))  # Total of 133,885 indices

    # Shuffle the indices randomly
    random.shuffle(total_indices)

    # Define number of samples for train/val/test splits
    # train_count = 99000
    # val_count = 1000
    # test_count = 33885
    train_count = 100000
    val_count = 23885
    test_count = 10000

    # Split indices for each dataset
    train_indices = total_indices[:train_count]
    val_indices = total_indices[train_count : train_count + val_count]
    test_indices = total_indices[
        train_count + val_count : train_count + val_count + test_count
    ]

    assert len(total_indices) == len(train_indices) + len(val_indices) + len(
        test_indices
    )

    rxn_index = {}
    rxn_index["train_index"] = train_indices
    rxn_index["valid_index"] = val_indices
    rxn_index["test_index"] = test_indices

    print(rxn_index)
    save_filename = "./data_split.pkl"
    with open(save_filename, "wb") as f:
        pickle.dump(rxn_index, f)
    print("Dumped to ", save_filename)
