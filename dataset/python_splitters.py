# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from dataset.split_utils import (
    process_split_ratio,
    min_rating_filter_pandas,
    split_pandas_data_with_ratios
)


def _do_stratification(
    config,
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    is_random=False
):
    # A few preliminary checks.
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    if 'userID' not in data.columns:
        raise ValueError("Schema of data not valid. Missing User Col")

    if 'itemID' not in data.columns:
        raise ValueError("Schema of data not valid. Missing Item Col")

    if not is_random:
        if 'timestamp' not in data.columns:
            raise ValueError("Schema of data not valid. Missing Timestamp Col")

    multi_split, ratio = process_split_ratio(ratio)

    split_by_column = 'userID' if filter_by == "user" else 'itemID'

    ratio = ratio if multi_split else [ratio, 1 - ratio]

    if min_rating > 1:
        data = min_rating_filter_pandas(
            data,
            min_rating=min_rating,
            filter_by=filter_by
        )

    # Split by each group and aggregate splits together.
    splits = []

    # If it is for chronological splitting, the split will be performed in a random way.
    df_grouped = (
        data.sort_values('timestamp').groupby(split_by_column)
        if is_random is False
        # group by users
        else data.groupby(split_by_column)
    )

    for _, group in df_grouped:
        group_splits = split_pandas_data_with_ratios(
            config, group, ratio, shuffle=is_random
        )

        # Concatenate the list of split dataframes.
        concat_group_splits = pd.concat(group_splits)

        splits.append(concat_group_splits)

    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits)

    # Take split by split_index
    splits_list = [
        splits_all[splits_all["split_index"] == x].drop("split_index", axis=1)
        for x in range(len(ratio))
    ]

    return splits_list


def python_chrono_split(
    config,
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user"
):
    """Pandas chronological splitter.

    This function splits data in a chronological manner. That is, for each user / item, the
    split function takes proportions of ratings which is specified by the split ratio(s).
    The split is stratified.

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    return _do_stratification(
        config,
        data,
        ratio=ratio,
        min_rating=min_rating,
        filter_by=filter_by,
        is_random=False
    )


def python_stratified_split(
    config,
    data,
    ratio=0.8,
    min_rating=1,
    filter_by="user"
):
    """Pandas stratified splitter.

    For each user / item, the split function takes proportions of ratings which is
    specified by the split ratio(s). The split is stratified.

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.

    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    return _do_stratification(
        config,
        data,
        ratio=ratio,
        min_rating=min_rating,
        filter_by=filter_by,
        is_random=True
    )


