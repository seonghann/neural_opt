# QM7X Data Processing
This directory contains scripts for processing QM7X molecular data, including structure extraction, SMARTS generation, and data splitting.


## 1. Save processed data that includes non-eq structures.

Process data to include corresponding non-equilibrium structures.
Change paths to fit your environment.

```bash
# 1. Train dataloader 
python run_dataloader.py \
  --datapath /home/jeheon/programs/MoreRed/src/morered/data \
  --split_file /home/jeheon/programs/MoreRed/src/morered/configs/data/split.npz \
  --batch_size 100 \
  --seed 42 \
  --sample_mode random \
  --dataloader_split train \
  --results_path train.random.pt

# 2. Validation dataloader
python run_dataloader.py \
  --datapath /home/jeheon/programs/MoreRed/src/morered/data \
  --split_file /home/jeheon/programs/MoreRed/src/morered/configs/data/split.npz \
  --batch_size 100 \
  --seed 42 \
  --sample_mode random \
  --dataloader_split val \
  --results_path val.random.pt

# 3. Test dataloader
python run_dataloader.py \
  --datapath /home/jeheon/programs/MoreRed/src/morered/data \
  --split_file /home/jeheon/programs/MoreRed/src/morered/configs/data/split.npz \
  --batch_size 100 \
  --seed 42 \
  --sample_mode random \
  --dataloader_split test \
  --results_path test.random.pt
```

For a detailed analysis of RMSD statistics, see `statistics.ipynb`.


## 2. Denoising Experiment (Optional)
To run denoising experiments from preprocessed data:

```bash
python run_dataloader.py \
  --datapath /home/jeheon/programs/MoreRed/src/morered/data \
  --split_file /home/jeheon/programs/MoreRed/src/morered/configs/data/split.npz \
  --model_path /home/jeheon/programs/MoreRed/src/morered/runs/a179c79e-41e1-11f0-9674-b44506f055df/best_model \
  --seed 42 \
  --denoising \
  --preprocessed_data test.random.pt \
  --results_path denoise.random.pt
```

To evaluate the denoised results:

```bash
python evaluate_accuracy_morered.py \
  --config_yaml ../../configs/sampling.qm7x.rdsm.yaml \
  --prb_pt denoise.random.pt
```


## 3. Convert PyTorch Data to XYZ Format
Convert `.pt` files to `.xyz` format with SMARTS generation.
Dataset split: train/val/test = 25431/7391/6815 (refer to split.npz).
train/val/test=25431,7391,6815 (refer to `split.npz`)

```bash
# Convert to xyz data
python pt_to_xyz.py --save_dir ./qm7x_xyz --input_file ./train.random.pt --start_idx 0 --end_idx 25430
python pt_to_xyz.py --save_dir ./qm7x_xyz --input_file ./val.random.pt --start_idx 25431 --end_idx 32821
python pt_to_xyz.py --save_dir ./qm7x_xyz --input_file ./test.random.pt --start_idx 32822 --end_idx 39636
```

## 4. Create data split
Generate train/validation/test splits excluding problematic samples.

```bash
# Find samples where SMARTS generation failed and save their indices
python process_wrong_samples_qm7x.py --prb_dir ./qm7x_xyz --save_index wrong_samples.pkl

# Write `./data_split.pkl` 
python data_split.py --num_train 25431 --num_val 7391 --num_test 6815 --excluding wrong_samples.pkl
# Excluded 32 indices
# Train: 25412, Val: 7383, Test: 6810
# Saved to ./data_split.pkl
```
