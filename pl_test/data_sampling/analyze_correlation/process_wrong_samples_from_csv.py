"""
QM9M 데이터 중, MMFF와 DFT의 3차원 구조가 동일한 samples이 있음.
이들을 제거하기 위한 과정.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="input csv filepath")
parser.add_argument("--output_csv", type=str, help="output csv filepath")
args = parser.parse_args()
print(args)


import pandas as pd

# df = pd.read_csv("./qm9m.csv")
df = pd.read_csv(args.input_csv)
print(df)

filtered_df = df[df["dmae"] > 1e-10]
print(filtered_df)
print(len(filtered_df))
# filtered_df.to_csv("./qm9m.csv", index=False)
filtered_df.to_csv(args.output_csv, index=False)
