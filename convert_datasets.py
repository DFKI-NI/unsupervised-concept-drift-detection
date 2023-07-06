from os import path
import pandas as pd
from scipy.io import arff

datasets = [
    "NOAA.arff",
    "INSECTS-abrupt_balanced_norm.arff",
    "INSECTS-gradual_balanced_norm.arff",
    "INSECTS-incremental-abrupt_balanced_norm.arff",
    "INSECTS-incremental_balanced_norm.arff",
    "INSECTS-incremental-reoccurring_balanced_norm.arff",
    "outdoor.arff",
    "poker-lsn.arff",
    "powersupply.arff",
    "rialto.arff",
    "sensorstream.arff",
]
base_path = "datasets/files"


def main():
    print("Converting datasets to .csv")
    for dataset in datasets:
        try:
            full_path = path.join(base_path, dataset)
            raw_data = arff.loadarff(full_path)
            new_path = f"{full_path[:-5]}.csv"
            df = pd.DataFrame(raw_data[0])
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    df[col] = df[col].apply(lambda val: val.decode("utf-8"))
            df.to_csv(new_path, index=False)
            print(f"{dataset}")
        except Exception as e:
            print(dataset, e)


if __name__ == "__main__":
    main()
