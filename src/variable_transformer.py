import numpy as np
import pandas as pd

def transform_value(variable_spec, date, base_df):
    """
    Transforms a single value according to the variable specification.
    
    Parameters
    ----------
    variable_spec : VariableSpec
        Specification of the transformations to apply
    date : pd.Timestamp
        The date corresponding to the value
    base_df : pd.DataFrame
        The base DataFrame containing the original data
        
    Returns
    -------
    float
        The transformed value
    """
    date = pd.to_datetime(date)
    
    # Check if the column exists in the dataframe
    if variable_spec.name not in base_df.columns:
        raise ValueError(f"Column {variable_spec.name} not found in base dataframe")
    
    # Apply transformations in sequence
    
    # 1. Start with the date we want to transform
    current_date = date
    
    # 2. If lag is specified, shift back in time to get the effective date
    if variable_spec.lag_order > 0:
        current_date = current_date - pd.DateOffset(months=variable_spec.lag_order)
        if current_date not in base_df.index:
            return np.nan # Return NaN for missing lag values
    
    # 3. Get the value at the current (possibly shifted) date
    current_value = base_df.loc[current_date, variable_spec.name]
    
    # 4. Apply log transform if specified
    if variable_spec.log_transform:
        current_value = np.log(current_value)
    
    # 5. Apply differencing if specified
    if variable_spec.diff_order > 0:
        diff_date = current_date - pd.DateOffset(months=variable_spec.diff_order)
        if diff_date not in base_df.index:
            return np.nan  
           
        # Get the value to subtract (from diff_date)
        diff_value = base_df.loc[diff_date, variable_spec.name]
        
        # Apply log transform to the diff value if needed
        if variable_spec.log_transform:
            diff_value = np.log(diff_value)
        
        # Apply the differencing
        current_value = current_value - diff_value
    
    return current_value


def reverse_transform_value(variable_spec, date, transformed_value, base_df):
    """
    Reverses the transformations applied by transform_value to get back the original value.
    
    Parameters
    ----------
    variable_spec : VariableSpec
        Specification of the transformations to undo
    date : pd.Timestamp or str
        The date corresponding to the transformed value
    transformed_value : float
        The transformed value
    base_df : pd.DataFrame
        The original DataFrame containing values needed for undoing transformations
        
    Returns
    -------
    float
        The original value at the original date
    """
    date = pd.to_datetime(date)
    
    # Check if the column exists in the dataframe
    if variable_spec.name not in base_df.columns:
        raise ValueError(f"Column {variable_spec.name} not found in base dataframe")
    
    # Check if transformed_value is NaN
    if pd.isna(transformed_value):
        raise ValueError("Cannot reverse transform a NaN value. The transformed value must be a valid number.")

    # Apply reverse transformations in the reverse sequence
    
    # 1. Start with the date we want to reverse transform
    current_date = date
    
    # 2. If lag is specified, shift back in time to get the effective date
    # (same as in transform_value)
    if variable_spec.lag_order > 0:
        current_date = current_date - pd.DateOffset(months=variable_spec.lag_order)
        if current_date not in base_df.index:
            raise ValueError(f"Missing required lag value for date {current_date}")
    
    # 3. Start with the transformed value
    result_value = transformed_value
    
    # 4. Undo differencing if it was applied
    if variable_spec.diff_order > 0:
        diff_date = current_date - pd.DateOffset(months=variable_spec.diff_order)
        if diff_date not in base_df.index:
            raise ValueError(f"Missing required base value for date {diff_date}")
        
        # Get the value to add back (from diff_date)
        diff_value = base_df.loc[diff_date, variable_spec.name]
        
        # Apply log transform to the diff value if needed
        if variable_spec.log_transform:
            diff_value = np.log(diff_value)
        
        # Undo the differencing by adding back the diff value
        result_value = result_value + diff_value
    
    # 5. Undo log transform if it was applied
    if variable_spec.log_transform:
        result_value = np.exp(result_value)
    
    # Note for lag: We've already accounted for lag by using the effective date
    # No further transformation needed
    
    return result_value

# TODO: DEze moet eigenlijk die andere gebruiken maar ik ben er even helemaal klaar mee
def transform_column(df, col_name, diff_order=0, take_log=False, lag_order=0):
    """
    Apply any combination of a logarithmic transformation, a difference transformation, and/or a lag transformation to a column of a Pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the column to be transformed.
    col_name : str
        The name of the column to be transformed.
    diff_order : int, optional
        The order of the difference transformation to be applied (default is 0, which means no transformation).
    take_log : bool, optional
        If True, apply a logarithmic transformation to the column (default is False).
    lag_order : int, optional
        The number of lags to apply to the transformed column (default is 0, which means no lags).

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the transformed column(s).
    str
        The name of the transformed column.
        
    Examples
    --------
    >>> df = pd.DataFrame({'date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01'],
                           'values': [3, 3, 7, 2, 4, 7]})

    # Transform column 'values' with a logarithmic transformation, a difference transformation of order 1, and a lag of 1
    >>> new_df, transformed_col = transform_column(df, 'values', diff_order=1, take_log=True, lag_order=1)

    # Print the new DataFrame and the name of the transformed column
    >>> print(new_df)
           date  values  log(values)(d1)(-1)
    0 2022-01-01       3                  NaN
    1 2022-02-01       3                  NaN
    2 2022-03-01       7             0.000000
    3 2022-04-01       2             0.847298
    4 2022-05-01       4            -1.252763
    5 2022-06-01       7             0.693147
    >>> print(transformed_col)
    log(values)(d1)(-1)
    """
    # Make a copy of the original DataFrame to avoid modifying the original data
    transformed_df = df.copy()
    transformed_df[col_name] = pd.to_numeric(transformed_df[col_name])
    original_columns = transformed_df.columns

    # Apply optional logarithm transformation to the specified column
    if take_log:
        transformed_df[f"log({col_name})"] = np.log(transformed_df[col_name])
        transformed_col = f"log({col_name})"
    else:
        transformed_col = col_name
    
    # Apply difference transformation to the transformed column if diff_order > 0
    if diff_order > 0:
        transformed_df[f"{transformed_col}(d{diff_order})"] = transformed_df[transformed_col].diff(diff_order)
        transformed_col = f"{transformed_col}(d{diff_order})"
    else:
        transformed_col = transformed_col
    
    # Apply lag transformation to the transformed column if lag_order > 0
    if lag_order > 0:
        transformed_df[f"{transformed_col}({-lag_order})"] = transformed_df[transformed_col].shift(lag_order)
        transformed_col = f"{transformed_col}({-lag_order})"
    
    if transformed_col not in list(original_columns):
        # Return the new DataFrame with the transformed column(s)
        return transformed_df[list(original_columns)+[transformed_col]], transformed_col
    else:
        # This prevents duplicate rows when nothing is done
        return transformed_df[list(original_columns)], transformed_col
