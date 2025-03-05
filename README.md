# recursiveForecasts
Implement functionality to make recursive forecasts for a linear model that allows for custom recursive relations.

```{python}
import pandas as pd
from src.variable_spec import VariableSpec
from src.linear_predictor import LinearPredictor

# Load some data
data = pd.read_excel(r"G:\EBO\ECMO\NIPE\NIPE_automated\Output\2025-Q1\29-01-2025 (MPE MAR25 - save sarimas)\Results\df_original.xlsx", index_col=0, parse_dates=True)

# Define the endogenous and exogenous variables
X = VariableSpec(
    name='SA03',
    diff_order=1, 
    log_transform=True,
    lag_order=1
)
y = VariableSpec(
    name='SA03',
    diff_order=1, 
    log_transform=True,
    lag_order=0,
)

# Fit the model
predictor = LinearPredictor(df=data)
predictor.fit(
    endogenous_specification=y,
    exogenous_specification=X, 
    last_month='2023-12-01',
    include_constant=True
)

# Predict
predictor.predict(steps_ahead=15)
```