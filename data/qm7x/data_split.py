import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, required=True)
    parser.add_argument("--num_val", type=int, required=True)
    parser.add_argument("--num_test", type=int, required=True)
    parser.add_argument("--excluding", type=str)
    args = parser.parse_args()
    
    # Generate indices
    train_indices = list(range(0, args.num_train))
    val_indices = list(range(args.num_train, args.num_train + args.num_val))
    test_indices = list(range(args.num_train + args.num_val, args.num_train + args.num_val + args.num_test))
    
    # Remove excluding indices if provided
    if args.excluding:
        with open(args.excluding, 'rb') as f:
            excluding = set(pickle.load(f))
        train_indices = [i for i in train_indices if i not in excluding]
        val_indices = [i for i in val_indices if i not in excluding]
        test_indices = [i for i in test_indices if i not in excluding]
        print(f"Excluded {len(excluding)} indices")
    
    split_index = {
        "train_index": train_indices,
        "valid_index": val_indices,
        "test_index": test_indices
    }
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    with open("./data_split.pkl", "wb") as f:
        pickle.dump(split_index, f)
    print("Saved to ./data_split.pkl")

if __name__ == "__main__":
    main()