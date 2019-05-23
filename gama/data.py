""" This module contains functions for loading data. """
from typing import Optional, Tuple, Union

import arff
import pandas as pd


def get_data_from_arff(file_path: str,
                       split_column: Optional[str] = 'last'
                       ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """ Loads data from the ARFF file into pandas DataFrame.
    Optionally splits off a column as series.

    :param file_path: str
        path to the ARFF file.
    :param split_column: str, optional (default='last')
        Column to split and return separately (e.g. target column).
        Value should either match a column name or 'last'.
        If 'last' is specified, the last column is returned separately.

    :returns
        pd.DataFrame if split_column is None.
        Tuple[pd.DataFrame, pd.Series] if split_column is specified.
    """
    with open(file_path, 'r') as arff_file:
        arff_dict = arff.load(arff_file)

    attribute_names, data_types = zip(*arff_dict['attributes'])
    data = pd.DataFrame(arff_dict['data'], columns=attribute_names)
    for attribute_name, dtype in arff_dict['attributes']:
        # 'real' and 'numeric' are probably interpreted correctly, date support needs to be added.
        if isinstance(dtype, list):
            data[attribute_name] = data[attribute_name].astype('category')

    if split_column is None:
        return data
    elif split_column == 'last':
        return data.iloc[:, :-1], data.iloc[:, -1]
    else:
        if split_column in attribute_names:
            return data.loc[:, data.columns != split_column], data.loc[:, split_column]
        else:
            raise ValueError("No column with name {} found in ARFF file {}"
                             .format(split_column, file_path))
