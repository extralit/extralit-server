from typing import Dict, Optional

import pandas as pd
import pandera as pa


def suggestion(series: pd.Series, values: Dict[str, Optional[Dict[str, str]]]):
    mask = series.isin(values)
    if not mask.all():
        print(f"Warning: Some values are not in the suggested list: {series[~mask].unique()}")

    return True
