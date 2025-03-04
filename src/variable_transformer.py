import numpy as np
import pandas as pd

# Function that converts a single value to a transformed value
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

    current_value = base_df.loc[date, variable_spec.name]
    
    # 1. Apply log transform if specified
    if variable_spec.log_transform:
        current_value = np.log(current_value)
    
    # 2. Apply differencing if specified
    if variable_spec.diff_order > 0:
        look_back_date = date - pd.DateOffset(months=variable_spec.diff_order)
        if look_back_date not in base_df.index:
            raise ValueError(
                f"Missing required base value for date {look_back_date}"
            )
        
        # Get the appropriate previous value from the base column
        previous_value = base_df.loc[look_back_date, variable_spec.name]
        
        # Apply log transform to the previous value if specified
        if variable_spec.log_transform:
            previous_value = np.log(previous_value)
        
        current_value = current_value - previous_value
    
    # 3. Apply lag order if specified
    if variable_spec.lag_order > 0:
        lag_date = date - pd.DateOffset(months=variable_spec.lag_order)
        if lag_date not in base_df.index:
            raise ValueError(
                f"Missing required lag value for date {lag_date}"
            )
        
        current_value = base_df.loc[lag_date, variable_spec.name]
    
    return current_value