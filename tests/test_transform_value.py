import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.variable_spec import VariableSpec
from src.variable_transformer import transform_value

@pytest.fixture
def test_dataframe():
    """Create a test dataframe for use in tests."""
    values = [i for i in range(1, 25)]
    return pd.DataFrame(
        data={'value': values}, 
        index=pd.date_range(start='2022-01-01', periods=len(values), freq='MS')
    )


def test_log_diff_normal_case(test_dataframe):
    """Test 1: log diff works in the normal case."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=1, 
        log_transform=True,
        lag_order=0,
    )
    expected = np.log(2) - np.log(1)
    output = transform_value(variable_spec=variable_spec, date='2022-02-01', base_df=test_dataframe)
    assert output == expected, 'Taking the log difference with order 1 failed for simple values'

def test_log_diff_missing_base_value(test_dataframe):
    """Test 2: check that the function returns nan when the base value for the differencing is missing."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=1, 
        log_transform=True,
        lag_order=0,
    )
    output = transform_value(variable_spec=variable_spec, date='2022-01-01', base_df=test_dataframe)
    assert pd.isna(output), 'Function should return nan when required base value for differencing is missing'

def test_log_diff_order_12(test_dataframe):
    """Test 3: check that the function works for a differencing order of 12."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=12, 
        log_transform=True,
        lag_order=0,
    )
    expected = np.log(13) - np.log(1)
    output = transform_value(variable_spec=variable_spec, date='2023-01-01', base_df=test_dataframe)
    assert output == expected, 'Taking the log difference with order 12 failed for simple values'


def test_differencing_only(test_dataframe):
    """Test 4: check that the function works for differencing only."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=1, 
        log_transform=False,
        lag_order=0,
    )
    expected = 2 - 1 
    output = transform_value(variable_spec=variable_spec, date='2023-01-01', base_df=test_dataframe)
    assert output == expected, 'Taking the first difference without taking the logarithm failed'


def test_log_transform_only(test_dataframe):
    """Test 5: check that the function works for log_transform only."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=0, 
        log_transform=True,
        lag_order=0,
    )
    expected = np.log(13)
    output = transform_value(variable_spec=variable_spec, date='2023-01-01', base_df=test_dataframe)
    assert output == expected, 'Taking the logarithm without differencing failed'


def test_no_transformation(test_dataframe):
    """Test 6: check that doing nothing changes nothing."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=0, 
        log_transform=False,
        lag_order=0,
    )
    expected = 13
    output = transform_value(variable_spec=variable_spec, date='2023-01-01', base_df=test_dataframe)
    assert output == expected, 'Doing nothing changed the input'


def test_lag_order_1(test_dataframe):
    """Test 7: check that the function works for lag order of 1."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=0, 
        log_transform=False,
        lag_order=1,
    )
    expected = 1
    output = transform_value(variable_spec=variable_spec, date='2022-02-01', base_df=test_dataframe)
    assert output == expected, 'Lag order of 1 failed'



def test_lag_order_missing_value(test_dataframe):
    """Test 8: check that the function returns nan when the lag order looks for a value that is not in the dataframe."""
    variable_spec = VariableSpec(
        name='value',
        diff_order=0, 
        log_transform=False,
        lag_order=1,
    )
    output = transform_value(variable_spec=variable_spec, date='2022-01-01', base_df=test_dataframe)
    assert pd.isna(output), 'Function should return nan when required lag value is missing'

