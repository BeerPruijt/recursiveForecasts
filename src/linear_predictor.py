import pandas as pd
from typing import List, Union
from datetime import datetime
from typing import Iterable
from copy import deepcopy
from src.variable_spec import VariableSpec
import matplotlib.pyplot as plt 
from src.variable_transformer import transform_column, transform_value, reverse_transform_value
import statsmodels.api as sm

class LinearPredictor:
    """A custom class for making linear predictions using Ordinary Least Squares (OLS) method. We need full control to implement recursive error correction and things like that"""
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the LinearPredictor.

        params:
        df = input dataframe with datetime index
        last_month = last month for which we have observations, start forecasting the month after 
        first_month = first month for which we want to start estimation (i.e. we disregard all before this) defaults to simply the first non_na observation
        exogenous_colnames = a list with the column names of the predictive variables as they are generated by the transform column function
        endogenous_colname = the column name of the variable to be predicted after all specified transformations are applied
        """
        self.df = df.copy()
        self.exogenous_colnames: List[str] = []  # predictive variables
        self.endogenous_colname: str | None = None  # variable to be predicted
        self.last_month: Union[datetime, str, None] = None
        self.first_month: Union[datetime, str, None] = None
        self.estimation_window: Union[Iterable, None] = None
        self.endogenous_specification: Union[VariableSpec, None] = None  # specification for variable to be predicted
        self.exogenous_specification: Union[VariableSpec, Iterable[VariableSpec], None] = None  # specification for predictive variables
        self.include_constant: bool = False
        self.fitted_model = None

    def _construct_exogenous_columns(self):
        """
        Add the predictive (exogenous) variables as columns to the dataframe.
        """
        # Add the constant if instructed
        if self.include_constant:
            self.exogenous_colnames.insert(0, 'const')
            self.df.insert(0, 'const', 1.0) 

        # Apply the transformations from the variable specifications to the relevant columns
        for var_spec in self.exogenous_specification:
            self.df, colname_temp = transform_column(
                self.df,
                var_spec.name,
                var_spec.diff_order,
                var_spec.log_transform,
                var_spec.lag_order
            )
            self.exogenous_colnames.append(colname_temp)

    def _construct_endogenous_column(self):
        """
        Add the transformed target (endogenous) variable as a column to the dataframe.
        """
        # Apply the transformations from the variable specification to the relevant column
        self.df, self.endogenous_colname = transform_column(
            self.df,
            self.endogenous_specification.name,
            self.endogenous_specification.diff_order,
            self.endogenous_specification.log_transform,
            self.endogenous_specification.lag_order
        )

    def _save_residual_plot(self):
        fig = plt.figure()
        plt.plot(self.estimation_window, self.fitted_model.resid)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(self.endogenous_specification.name)
        plt.xlabel("Time")
        plt.ylabel("Residuals")
        fig.savefig(r'G:\EBO\ECMO\NIPE\NIPE_automated\_temp' + '\\' + self.endogenous_specification.name + '.jpg')
        plt.close()

    def fit(self, endogenous_specification: VariableSpec, exogenous_specification: Union[VariableSpec, Iterable[VariableSpec]], last_month: Union[datetime, str], first_month: Union[datetime, str, None] = None, include_constant: bool = False):
        """
        Fit the linear model.
        
        Parameters:
        -----------
        endogenous_specification : VariableSpec
            Specification for the target variable to be predicted
        exogenous_specification : Union[VariableSpec, Iterable[VariableSpec]]
            Specification(s) for the predictive variables
        last_month : Union[datetime, str]
            Last month of data to use in estimation
        first_month : Union[datetime, str, None]
            First month of data to use in estimation
        include_constant : bool
            Whether to include a constant term in the regression
        """
        # Initialize
        self.endogenous_specification = deepcopy(endogenous_specification)
        self.exogenous_specification = deepcopy(exogenous_specification)
        if isinstance(self.exogenous_specification, VariableSpec):
            self.exogenous_specification = [self.exogenous_specification]
        self.include_constant = include_constant

        # Add the relevant columns from the specification to the dataframe, save the column names 
        self._construct_endogenous_column()
        self._construct_exogenous_columns()

        # Ensure they are in datetime format
        self.last_month = pd.to_datetime(last_month)
        self.first_month = pd.to_datetime(first_month) if first_month is not None else None

        # If first month is left unspecified we use the first index for which all relevant variables are present
        if self.first_month is None:
            self.first_month = self.df.loc[:, self.exogenous_colnames + [self.endogenous_colname]].dropna().index[0]

        # Construct the estimation window
        self.estimation_window = pd.date_range(
            start=self.first_month,
            end=self.last_month,
            freq='MS'
        )

        # Fit the model using basic statsmodels implementation
        self.fitted_model = sm.OLS(
            endog=self.df.loc[self.estimation_window, [self.endogenous_colname]],
            exog=self.df.loc[self.estimation_window, self.exogenous_colnames]
        ).fit()

        # Save the residual plots
        self._save_residual_plot()

    def _make_prediction_for_idx(self, index):
        exog_for_idx = self.df.loc[index, self.exogenous_colnames]
        return self.fitted_model.predict(exog=exog_for_idx).values[0]

    def predict(self, start: Union[datetime, str, None] = None, end: Union[datetime, str, None] = None, steps_ahead: int = 12):

        # Steps ahead is initialized in the class as 12, here we might overwrite it if explitly provided
        if steps_ahead is not None:
            self.steps_ahead = steps_ahead

        # If the start date is specified we use it otherwise we start after last_month
        if start is None:
            self.start = self.last_month + pd.DateOffset(months=1)
        else:
            self.start = start

        # If the end date is specified we go with that, otherwise we use h periods
        if end is not None:
            self.end = end
            self.forecast_window = pd.date_range(start=self.start, end=self.end, freq='MS')
        else:
            self.forecast_window = pd.date_range(start=self.start, periods=self.steps_ahead, freq='MS')

        # TODO
        # Here we validate the inputs, specifically we check that the first forecast is possible and all non-recursive columns contain no NaN's

        # Make the prediction for each index in the forecast window
        for index in self.forecast_window:
            
            # Make prediction for the endogenous variable
            prediction = self._make_prediction_for_idx(index)
            self.df.loc[index, self.endogenous_colname] = prediction

            # Convert it back to a level prediction for the base column
            level_prediction = reverse_transform_value(
                variable_spec=self.endogenous_specification,
                transformed_value=prediction,
                base_df=self.df,
                date=index
            )
            self.df.loc[index, self.endogenous_specification.name] = level_prediction


            # Construct any derivative explanatory variables that might depend on this forecast
            # If NAME of the exogenous variable is the same as the endogenous variable, we need to update the exogenous variable
            for var_spec in self.exogenous_specification:
                if var_spec.name == self.endogenous_specification.name:
                    # We transform the value
                    transformed_val = transform_value(
                        variable_spec=var_spec,
                        date=index,
                        base_df=self.df
                    )
                    # We determinen the relevant index for the transformed value (lag order requires us to assign it to a future date)
                    relevant_index = index + pd.DateOffset(months=var_spec.lag_order)
                    self.df.loc[relevant_index, var_spec.get_transformed_column_name()] = transformed_val

            # TODO: implement custom functionality for the error correction terms for example

        return self.df.loc[self.forecast_window, self.endogenous_specification.name]
        