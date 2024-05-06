import io
import json
import re
from typing import List, Optional, Union, Dict, Literal, Any

import pandas as pd
import pandera as pa

from extralit.schema.checks.utils import make_same_length_arguments


def drop_single_value_index_names(df: pd.DataFrame) -> pd.Index:
    names_to_drop = []
    for name in df.index.names:
        num_uniques = df.index.get_level_values(name).nunique()
        
        if num_uniques <= 1:
            names_to_drop.append(name)

    if names_to_drop and len(names_to_drop) == len(df.index.names):
        names_to_drop = names_to_drop[1:]

    return pd.Index(names_to_drop)


def preprocess(df: pd.DataFrame,
               required_columns: List[str]=[],
               drop_columns: Optional[List[str]] = None):
    """
    Preprocess a DataFrame before utils.
    """
    assert isinstance(required_columns, list), 'required_columns must be a list'

    if drop_columns:
        drop_columns = pd.Index(drop_columns).difference(required_columns)
        drop_index_levels = set(df.index.names or [df.index.name]) & set(drop_columns)
        if drop_index_levels:
            df.index = df.index.droplevel(list(drop_index_levels))
        df = df.drop(columns=drop_columns, errors='ignore')

    # Drop non-unique index levels
    if df.shape[0] > 1:
        single_value_index_levels = drop_single_value_index_names(df).difference(required_columns)
        if len(single_value_index_levels):
            df.index = df.index.droplevel(single_value_index_levels)

    # Replace values in string columns
    df_str_cols = df.select_dtypes(include=['O', 'string'])
    df.loc[:, df_str_cols.columns] = df_str_cols.replace({None: 'NA', '': 'NA'})

    # Drop columns with all "NA"
    # all_na_columns = df.columns.difference(required_columns)
    # all_na_columns = all_na_columns[(df[all_na_columns].isna() | (df[all_na_columns] == 'NA')).all(axis=0)]
    # df = df.drop(columns=all_na_columns, errors='ignore')

    return df


def standardize_values(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    # Capitalize string values if the schema has `isin` checks with capital letters
    for column_name, columnSchema in schema.columns.items():
        if column_name not in df.columns: continue
        isin_checks = [check for check in columnSchema.checks if isinstance(check, pa.Check) and hasattr(check, 'isin')]
        if not isin_checks or 'allowed_values' not in isin_checks[0].statistics:
            continue

        if all([value.istitle() for value in isin_checks[0].statistics['allowed_values']]):
            df[column_name] = df[column_name].apply(lambda x: x.capitalize() if pd.notna(x) and x != "NA" else x)

    return df


def get_required_columns(schema: pa.DataFrameSchema):
    required_columns = [name for name, column in schema.columns.items() if not column.nullable]
    index_columns = [index.name for index in schema.index.indexes] if hasattr(schema.index, 'indexes') else []
    required_columns = required_columns + index_columns
    return required_columns


def df_to_json(df: pd.DataFrame,
               schema: Union[pa.DataFrameModel, pa.DataFrameSchema],
               drop_columns=None,
               transpose=False,
               metadata: Optional[Dict[str, Any]] = None,
               **kwargs) -> str:
    """
    Convert a DataFrame to a JSON string with utils information.
    """
    dataframe_schema = schema if isinstance(schema, pa.DataFrameSchema) else schema.to_schema()
    required_columns = get_required_columns(dataframe_schema)

    df = preprocess(df, required_columns=required_columns, drop_columns=drop_columns)

    try:
        df = standardize_values(df, dataframe_schema)
    except Exception as e:
        print('Failed to standardize values:', e)

    # if df.columns.size == 0 or df.shape[0] == 0:
    #     return 'NA'

    if transpose:
        df = df.T
        df.index.name = df.index.name or 'reference'

    df_json = json.loads(
        df.to_json(orient='table', 
                   index=bool(df.index.name) or len(df.index.names)>1))
    if dataframe_schema is not None:
        df_json['validation'] = schema_to_json(dataframe_schema, )

    if metadata:
        df_json = {**metadata, **df_json}
    
    return json.dumps(df_json)


def schema_to_json(dataframe_schema: pa.DataFrameSchema) -> Dict[str, Any]:
    schema_specs = json.loads(dataframe_schema.to_json())

    for check in (schema_specs['checks'] or []):
        if check not in ['check_less_than', 'check_greater_than', 'check_between']:
            continue
        schema_specs['checks'][check] = {
            k: v for k, v in
            zip(schema_specs['checks'][check].keys(),
                make_same_length_arguments(**schema_specs['checks'][check]))}

    for index in schema_specs['index']:
        index['required'] = True

    return schema_specs


def json_to_df(json_string: str,
               schema: Optional[pa.DataFrameSchema] = None) -> pd.DataFrame:
    """
    Convert a JSON string to a DataFrame. If a schema is provided, the DataFrame will be validated against the schema.

    Args:
        json_string: JSON string
        schema: DataFrameSchema
        index_level_rename: Dictionary to rename index levels

    Returns:
        pd.DataFrame
    """
    if not json_string:
        return pd.DataFrame()

    if schema is not None:
        index_cols = [name for name in schema.index.names if name]
        required_columns = get_required_columns(schema)
    else:
        index_cols = []
        required_columns = []

    try:
        try:
            df = pd.read_json(io.StringIO(json_string) if isinstance(json_string, str) else json_string,
                              orient='table', convert_dates=False)
        except:
            json_string = re.sub(r'"type":"\w+"', '"type":"string"', json_string)
            json_string = re.sub(r',\s*"extDtype":"[^"]+"', '', json_string)
            df = pd.read_json(io.StringIO(json_string) if isinstance(json_string, str) else json_string,
                              orient='table', convert_dates=False)

        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        if schema is not None:
            df = convert_to_schema_dtypes(df, schema)

        if index_cols and df.columns.intersection(index_cols).size:
            df = df.set_index(index_cols)
    except Exception as e:
        print(f"Failed to load DataFrame from JSON: {e}")
        raise e

    return df


def convert_to_schema_dtypes(df: pd.DataFrame, schema: pa.DataFrameSchema,
                             errors: Literal['raise', 'ignore'] = 'ignore') -> pd.DataFrame:
    dtype_map = {'int64': 'Int64', 'int': 'Int64', 'int32': 'Int64'}

    data_types = {col: dtype_map.get(str(datatype), str(datatype)) \
                  for col, datatype in schema.dtypes.items() if col in df.columns}

    df = df.astype(data_types, errors=errors)
    return df


def is_json_table(json_string: str) -> bool:
    if not isinstance(json_string, str) or not json_string.startswith('{') or not json_string.endswith('}'):
        return False

    try:
        json.dumps(json_string)
        return True
    except:
        return False
