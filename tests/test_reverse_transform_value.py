import pytest
import pandas as pd
import numpy as np
from itertools import product
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.variable_spec import VariableSpec
from src.variable_transformer import transform_value, reverse_transform_value


@pytest.fixture
def test_data():
    """Create test data with quadratic pattern and transformed values."""
    # Create a date range for 3 years of monthly data
    date_range = pd.date_range(start='2020-01-01', periods=36, freq='MS')
    
    # Triangular numbers with linearly increasing differences
    quadratic = np.array([1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 
                          66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 
                          231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 
                          496, 528, 561, 595, 630, 666])
    
    # Create the DataFrame
    test_df = pd.DataFrame({
        'value': quadratic
    }, index=date_range)
    
    # Create all specifications
    all_specs = []
    for diff, log, lag in product([0, 1], [False, True], [0, 1]):
        spec = VariableSpec(
            name='value',
            diff_order=diff,
            log_transform=log,
            lag_order=lag
        )
        all_specs.append(spec)
    
    # Apply transformations
    transformed_df = test_df.copy()
    for spec in all_specs:
        column_name = spec.get_transformed_column_name()
        transformed_values = pd.Series(index=test_df.index, dtype=float)
        
        for date in test_df.index:
            transformed_values[date] = transform_value(
                variable_spec=spec,
                date=date,
                base_df=test_df
            )
        
        transformed_df[column_name] = transformed_values
    
    return transformed_df

# It is important to note that we handle the lag order when assigning it to specific dates not in the transformation
def test_reverse_transform_lag(test_data):
    """Test reverse transform for lag only."""
    spec = VariableSpec(name='value', diff_order=0, log_transform=False, lag_order=1)

    # General part
    last_index = test_data.index[-1]
    transformed_value = test_data.loc[last_index, spec.get_transformed_column_name()]
    result = reverse_transform_value(
        variable_spec=spec,
        date=last_index,
        transformed_value=transformed_value,
        base_df=test_data[['value']]
    )

    # We expect the value from the previous month, the transformation shouldn't change it
    expected_output = test_data.loc[last_index - pd.DateOffset(months=1), 'value']
    assert result == expected_output, f"Failed to reverse lag transform. Expected: {expected_output}, Got: {result}"

def test_reverse_transform_log(test_data):
    """Test reverse transform for log only."""
    spec = VariableSpec(name='value', diff_order=0, log_transform=True, lag_order=0)
    
    # General part
    last_index = test_data.index[-1]
    transformed_value = test_data.loc[last_index, spec.get_transformed_column_name()]
    result = reverse_transform_value(
        variable_spec=spec,
        date=last_index,
        transformed_value=transformed_value,
        base_df=test_data[['value']]
    )

    # We expect the value from the final index
    expected_output = test_data.loc[last_index, 'value']
    assert np.isclose(result, expected_output), f"Failed to reverse log transform. Expected: {expected_output}, Got: {result}"


def test_reverse_transform_log_lag(test_data):
    """Test reverse transform for log and lag."""
    spec = VariableSpec(name='value', diff_order=0, log_transform=True, lag_order=1)

    # General part
    last_index = test_data.index[-1]
    transformed_value = test_data.loc[last_index, spec.get_transformed_column_name()]
    result = reverse_transform_value(
        variable_spec=spec,
        date=last_index,
        transformed_value=transformed_value,
        base_df=test_data[['value']]
    )

    # We expect the value from the previous month, the transformation shouldn't change it
    expected_output = test_data.loc[last_index - pd.DateOffset(months=1), 'value']
    assert np.isclose(result, expected_output), f"Failed to reverse lag transform. Expected: {expected_output}, Got: {result}"

def test_reverse_transform_diff(test_data):
    """Test reverse transform for diff only."""
    spec = VariableSpec(name='value', diff_order=1, log_transform=False, lag_order=0)

    # General part
    last_index = test_data.index[-1]
    transformed_value = test_data.loc[last_index, spec.get_transformed_column_name()]
    result = reverse_transform_value(
        variable_spec=spec,
        date=last_index,
        transformed_value=transformed_value,
        base_df=test_data[['value']]
    )

    # We expect the value from the final index
    expected_output = test_data.loc[last_index, 'value']
    assert np.isclose(result, expected_output), f"Failed to reverse lag transform. Expected: {expected_output}, Got: {result}"

def test_reverse_transform_diff_lag(test_data):
    """Test reverse transform for diff and lag."""
    spec = VariableSpec(name='value', diff_order=1, log_transform=False, lag_order=1)

    # General part
    last_index = test_data.index[-1]
    transformed_value = test_data.loc[last_index, spec.get_transformed_column_name()]
    result = reverse_transform_value(
        variable_spec=spec,
        date=last_index,
        transformed_value=transformed_value,
        base_df=test_data[['value']]
    )

    # We expect the value from the previous month, the transformation shouldn't change it
    expected_output = test_data.loc[last_index - pd.DateOffset(months=1), 'value']
    assert np.isclose(result, expected_output), f"Failed to reverse lag transform. Expected: {expected_output}, Got: {result}"

def test_reverse_transform_log_diff(test_data):
    """Test reverse transform for log and diff."""
    spec = VariableSpec(name='value', diff_order=1, log_transform=True, lag_order=0)

    # General part
    last_index = test_data.index[-1]
    transformed_value = test_data.loc[last_index, spec.get_transformed_column_name()]
    result = reverse_transform_value(
        variable_spec=spec,
        date=last_index,
        transformed_value=transformed_value,
        base_df=test_data[['value']]
    )

    # We expect the value from the final index
    expected_output = test_data.loc[last_index, 'value']
    assert np.isclose(result, expected_output), f"Failed to reverse lag transform. Expected: {expected_output}, Got: {result}"

def test_reverse_transform_log_diff_lag(test_data):
    """Test reverse transform for log, diff, and lag."""
    spec = VariableSpec(name='value', diff_order=1, log_transform=True, lag_order=1)

    # General part
    last_index = test_data.index[-1]
    transformed_value = test_data.loc[last_index, spec.get_transformed_column_name()]
    result = reverse_transform_value(
        variable_spec=spec,
        date=last_index,
        transformed_value=transformed_value,
        base_df=test_data[['value']]
    )

    # We expect the value from the previous month, the transformation shouldn't change it
    expected_output = test_data.loc[last_index - pd.DateOffset(months=1), 'value']
    assert np.isclose(result, expected_output), f"Failed to reverse lag transform. Expected: {expected_output}, Got: {result}"

def test_reverse_transform_nan_handling(test_data):
    """Test that reverse_transform_value raises an error when given NaN values."""
    spec = VariableSpec(name='value', diff_order=1, log_transform=True, lag_order=0)
    
    with pytest.raises(ValueError) as excinfo:
        last_index = test_data.index[-1]
        transformed_value = np.nan
        result = reverse_transform_value(
            variable_spec=spec,
            date=last_index,
            transformed_value=transformed_value,
            base_df=test_data[['value']]
        )
    
    # Check that the error message is appropriate
    assert "NaN" in str(excinfo.value), "Error message should mention NaN values"


def test_reverse_transform_missing_column(test_data):
    """Test that reverse_transform_value returns NaN for a missing column."""
    spec = VariableSpec(name='non_existent_column', diff_order=0, log_transform=False, lag_order=0)

    with pytest.raises(ValueError) as excinfo:
        last_index = test_data.index[-1]
        transformed_value = 420 # Some arbitrary value, NOTE: it is important that the check for the column is done the other checks otherwise this test will fail
        result = reverse_transform_value(
            variable_spec=spec,
            date=last_index,
            transformed_value=transformed_value,
            base_df=test_data[['value']]
        )
    
    # Check that the error message is appropriate
    assert "Column" in str(excinfo.value), "Error message should mention a missing column"

def test_property_transform_and_reverse_are_inverse(test_data):
    """Test the property that transform followed by reverse transform returns the original value."""
    # Get all available specs from column names
    # Skip the 'value' column which is the original data
    transformed_columns = [col for col in test_data.columns if col != 'value']
    
    # Create specs for each transformation
    specs = []
    for col in transformed_columns:
        diff_order = 1 if '_d1' in col else 0
        log_transform = 'log' in col
        lag_order = 1 if '_l1' in col else 0
        
        spec = VariableSpec(
            name='value',
            diff_order=diff_order,
            log_transform=log_transform,
            lag_order=lag_order
        )
        specs.append(spec)
    
    # Test a few specific dates in the middle of the series
    # Avoid first date due to differencing and lagging issues
    test_dates = [
        pd.Timestamp('2021-01-01'),  # 13th month
        pd.Timestamp('2021-06-01'),  # 18th month
        pd.Timestamp('2022-01-01')   # 25th month
    ]
    
    for date in test_dates:
        # The original value at this date
        original_value = test_data.loc[date, 'value']
        
        for spec in specs:
            # Skip lag on first date if needed
            if spec.lag_order > 0 and date == test_data.index[0]:
                continue
                
            # Find transformed value from the dataframe
            column_name = spec.get_transformed_column_name()
            transformed_value = test_data.loc[date, column_name]
            
            # Skip if transform is NaN
            if pd.isna(transformed_value):
                continue
            
            # Special case: If lag is applied skip the check
            # because lag transformation isn't mathematically reversible without
            # knowing the original value
            if spec.lag_order > 0:
                continue

            # Now reverse transform this value
            reversed_value = reverse_transform_value(
                variable_spec=spec,
                date=date,
                transformed_value=transformed_value,
                base_df=test_data[['value']]  # Pass only the original data column
            )
            
            # Check if we get back the original value
            assert np.isclose(reversed_value, original_value), \
                f"Transform-reverse property failed for {spec} at {date}. " \
                f"Original: {original_value}, Transformed: {transformed_value}, " \
                f"Reversed: {reversed_value}, Diff: {original_value - reversed_value}"