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