import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def grouped_train_validation_test_split(dataframe, group_column="case_group_id",
                                        train_size=0.70, validation_size=0.15,
                                        test_size=0.15, random_state=42):
    if abs(train_size + validation_size + test_size - 1.0) > 1e-8:
        raise ValueError("Split proportions must sum to 1.0")
    if group_column not in dataframe.columns:
        raise ValueError(f"Missing grouping column: {group_column}")
    if dataframe[group_column].isna().any():
        raise ValueError("Resolve missing case_group_id values before splitting")
    groups = dataframe[group_column]
    first = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, remainder_idx = next(first.split(dataframe, groups=groups))
    remainder = dataframe.iloc[remainder_idx]
    relative_test = test_size / (validation_size + test_size)
    second = GroupShuffleSplit(n_splits=1, train_size=relative_test, random_state=random_state)
    test_idx, val_idx = next(second.split(remainder, groups=remainder[group_column]))
    return dataframe.iloc[train_idx].copy(), remainder.iloc[val_idx].copy(), remainder.iloc[test_idx].copy()
