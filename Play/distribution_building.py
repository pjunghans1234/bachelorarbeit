import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import calendar
import numpy as np
import math
import xarray as xr
import sklearn


from sklearn.linear_model import LinearRegression
from Play import shape_data as shape
from Play import plot_one_var
from Play import dt_functions 
from Play import LinReg_analise
from Play import LinReg_building
from Play import shape_data


def residuals_to_variance_regr(residuals, output_var = "mrsol", input_vars = ["tas","pr"]):
    input_arr = residuals[input_vars].to_array().stack(features=tuple(d for d in ["time", "run", "lat", "lon"] if d in residuals[input_vars].dims)).transpose("features", "variable") 
    output_arr = residuals[f"{output_var}_res"].stack(features = ("time", "run"))
    output_values = output_arr.values[np.isfinite(output_arr)]
    input_values = input_arr.values[np.isfinite(output_arr)]
    var_regr = LinearRegression()
    var_regr.fit(input_values, output_values**2)
    return var_regr


def distribution_predictor(scen = "historical", input_vars = ["tas", "pr"], output_var = "mrsol",
                            lat_idx = 16, lon_idx = 32 , depth = 0, r = 0,
                            start ="1850-01-01", end = "1900-01-01", time_step = "1ME",  Month_idx = 1, hist = 1, 
                            mu_min_run_idx = 1, mu_max_run_idx = 2,
                            var_min_run_idx = 2, var_max_run_idx = 5
                           ):

    #linearabhängigen Teil schätzen
    mu_regr = LinReg_building.local_vars_to_one_dim(scen = scen, variables = input_vars, output_var = output_var,
                            lat_idx = lat_idx, lon_idx = lon_idx, depth = depth, r = r,
                            start =start, end = end, time_step = time_step,  Month_idx = Month_idx, hist = hist, 
                            run_idx = mu_min_run_idx,
                            )             ### ToDo local_vars_to_dim mit mehreren runs möglich machen
    
    #residuen ausrechnen
    residuals = shape_data.create_residuals (mu_regr, scen = scen, input_vars = input_vars, output_var = output_var,
                            lat_idx = lat_idx, lon_idx = lon_idx, depth = depth, r = r,
                            start =start, end = end, time_step = time_step, Month_idx = Month_idx, hist = hist,
                            min_run_idx = var_min_run_idx, max_run_idx = var_max_run_idx)
    
    #residuals fitten
    var_regr = residuals_to_variance_regr(residuals=residuals, output_var = output_var, input_vars = input_vars)

    return (mu_regr,var_regr)




