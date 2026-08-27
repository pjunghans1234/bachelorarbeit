from model.stats._parallel_linear_regression import ParLinearRegression
from model.stats._parallel_linear_regression_stack import ParLinearRegressionStack
from model.stats._parallel_polynomial_regression import ParPolyRegression
from model.stats._linear_regression import LinearRegression

__all__ = [
    #linear regression
    "LinearRegression",
    "ParLinearRegression",
    "ParLinearRegressionStack",
    "ParPolyRegression"
    ]
