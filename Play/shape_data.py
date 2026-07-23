import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from Play import config as conf
from Play import dt_functions 



xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)

def Transform_ds(ds):
    ds_T = ds
    ds_T["mrsol"] = (ds.mrsol/100)    
    ds_T["mrsol"] = np.log(ds_T.mrsol/(1-ds_T.mrsol))
    print("yeaay")
    return ds_T
        

#hold_out integrieren
def show_data_set(var = "tas", scen = "historical", test = False,run_idx = 1, Transform = False):
    if Transform:
        return Transform_ds( xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.all_runs[run_idx]}_g025.nc'))
    if test :
        return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.test_run}_g025.nc')
    return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.all_runs[run_idx]}_g025.nc')

def make_concat_set(var = "tas", scen = "historical", set = "all_runs", min_run_idx = 11, max_run_idx = 21, Transform = False):
    if Transform :
        data_sets = [Transform_ds(xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc'))  for run in getattr(conf, set)[min_run_idx:max_run_idx]]
    
    else : 
        data_sets = [xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc')  for run in getattr(conf, set)[min_run_idx:max_run_idx]]
    
    
    #merged variante
    #for i in range(0,10):
    #    data_sets[i] = data_sets[i].rename({var: f"{var}_{i}"})
    #return xr.merge(data_sets, compat = "identical")

    return xr.concat(data_sets, "run")

def start_with_hist(start,time_step,hist):

    datetime = pd.to_datetime(start)
    offset = pd.tseries.frequencies.to_offset(time_step)
    new_date = datetime - hist * offset
    return new_date.strftime("%Y-%m-%d")

def place_with_radius(ds, lat_idx, lon_idx, r):
    print(ds.isel(lat=slice(lat_idx-r,lat_idx+r+1),lon=slice(lon_idx-r,lon_idx+r+1)))
    return ds.isel(lat=slice(lat_idx-r,lat_idx+r+1),lon=slice(lon_idx-r,lon_idx+r+1))

def first_not_none_element(matrix):
    for row in matrix:
        for val in row:
            if val is not None:
                return val
    return None

def create_empty_prediction(pred_mat, input):
    x =  0
    for row in pred_mat:
        y = 0
        for pred in row:
            if pred is not None:
                arr = pred.predict(input.isel(lat = x, lon_idx = y).to_array().T)
                arr.fill(0)
                return arr
            y = y + 1
        x = x + 1
    return None

def prune_group_ds_timespan(ds, 
                        start = "1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None):
    ds = ds.sel(time=slice(start, end)).resample(time=time_step).mean()
    if Month_idx != None:
        ds = ds.sel(time= ds.time.dt.month == Month_idx)
    return ds

def prune_location(ds, lat_idx, lon_idx, r = 0):        
    min_lat_idx = max(0,lat_idx - r)
    max_lat_idx = min(lat_idx + r + 1, ds.sizes["lat"])
    if lon_idx - r < 0:
        ds = ds.roll(lon = r)
        ds=ds.isel(lat=slice(min_lat_idx,max_lat_idx),lon=slice(lon_idx,lon_idx+2*r+1))
    elif lon_idx + r >= ds.sizes["lon"]:
        ds = ds.roll(lon = - r)
        ds=ds.isel(lat=slice(min_lat_idx,max_lat_idx),lon=slice(lon_idx-2*r,lon_idx+1))
    else:
        ds=ds.isel(lat=slice(min_lat_idx,max_lat_idx),lon=slice(lon_idx-r,lon_idx+r+1))
    return ds

def load_create_datatree(scenarios = ["historical"], variables = ["tas", "pr", "mrsol"], 
                        start = "1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None, 
                        test = False,run_idx = None, min_run_idx = 1, max_run_idx = 2):
    if run_idx != None:
        min_run_idx = run_idx
        max_run_idx = run_idx + 1
    dt = xr.DataTree()

    for scen in scenarios:
        dt[scen] = xr.DataTree()
        for var in variables:
            if max_run_idx-min_run_idx == 1:
                ds = show_data_set(var=var, scen=scen, test = test, run_idx = min_run_idx)
            else:
                ds = make_concat_set(var=var, scen=scen, min_run_idx=min_run_idx,max_run_idx=max_run_idx)   #exkluded test
            ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
            ds_pruned = prune_group_ds_timespan(ds, start= start, end= end, time_step = time_step, Month_idx = Month_idx)
            dt[scen][var] = xr.DataTree(ds_pruned)

    return dt

def dt_to_features (dt, 
                        lat_idx , lon_idx, r = 0, 
                        Month_idx = None, hist = 1, 
                        prune = False, start = "1850-01-01", end = "1900-01-01", time_step = "1ME", intern_run_idx = 0):
    
    datasets = [subtree.to_dataset() for subtree in dt.subtree if not subtree.is_empty]

    data_ds = xr.merge(datasets)


    min_lat_idx = max(0,lat_idx - r)
    max_lat_idx = min(lat_idx + r + 1, data_ds.sizes["lat"])
    if lon_idx - r < 0:
        data_ds = data_ds.roll(lon = r)
        data_ds=data_ds.isel(lat=slice(min_lat_idx,max_lat_idx),lon=slice(lon_idx,lon_idx+2*r+1))
    elif lon_idx + r >= data_ds.sizes["lon"]:
        data_ds = data_ds.roll(lon = - r)
        data_ds=data_ds.isel(lat=slice(min_lat_idx,max_lat_idx),lon=slice(lon_idx-2*r,lon_idx+1))
    else:
        data_ds=data_ds.isel(lat=slice(min_lat_idx,max_lat_idx),lon=slice(lon_idx-r,lon_idx+r+1))

    
    if prune:
        start = start_with_hist(start, time_step, hist)
        data_ds = data_ds.sel(time=slice(start, end)).resample(time=time_step).mean()
    
    history = []
    for i in range(0,hist):
        tmp = data_ds.shift(time=i)
        history.append(tmp)

    data_ds = xr.concat(history, "hist")

    data_ds = data_ds.isel(time=slice(hist-1,None))

    if Month_idx != None:
        data_ds = data_ds.sel(time= data_ds.time.dt.month == Month_idx)
    
    if "run" in data_ds.dims:
        data_ds = data_ds.isel(run = intern_run_idx)


    features_arr = data_ds.to_array().transpose("time","variable","hist","lat","lon").stack(features=("variable","hist","lat","lon"))

    return features_arr


def res_to_features (residuals, output_var = "mrsol",input_vars = ["tas","pr"]):

    input_arr = residuals[input_vars].to_array().stack(features = ("time", "run", "lat", "lon")).transpose("features", "variable") 
    output_arr = residuals[f"{output_var}_res"].stack(features = ("time", "run"))
    output_values = output_arr.values[np.isfinite(output_arr)]
    input_values = input_arr.values[np.isfinite(output_arr)]
    return input_values, output_values

def residuals_to_variance_regr(residuals, output_var = "mrsol", input_vars = ["tas","pr"]):
    input_arr = residuals[input_vars].to_array().stack(features=tuple(d for d in ["time", "run", "lat", "lon"] if d in residuals[input_vars].dims)).transpose("features", "variable") 
    output_arr = residuals[f"{output_var}_res"].stack(features = ("time", "run"))
    output_values = output_arr.values[np.isfinite(output_arr)]
    input_values = input_arr.values[np.isfinite(output_arr)]
    var_regr = LinearRegression()
    var_regr.fit(input_values**2, output_values)
    return var_regr


    #gives tas and pr only for r = 0, hist = 1
    #Realoutcome minus estimated
    #number_runs_hat format leicht geändert
def create_residuals(regr, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol",
                            lat_idx = 30, lon_idx = 0, depth = 0, r = 0,
                            start ="1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = 1, hist = 1,
                            test = True,  min_run_idx = 1, max_run_idx = 2
                            , Transform = False):
    
    
    
    data_tree = xr.DataTree()

    #real outcome
    output = make_concat_set(var = output_var, scen = scen, min_run_idx=min_run_idx, max_run_idx=max_run_idx,Transform = Transform)      #exkludet test
    output = prune_group_ds_timespan(output,start= start,end= end,time_step = time_step, Month_idx = Month_idx)
    output = output.isel(lat = lat_idx, lon = lon_idx)

    output = output.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")
    if output_var == "mrsol":
        output= output.isel(depth = depth)



    #input
    start_hist = start_with_hist(start=start,time_step= time_step,hist= hist)
    data_tree["input"] = load_create_datatree(scenarios = [scen],variables = input_vars, start = start_hist,end = end,time_step= time_step, test=test, min_run_idx=min_run_idx, max_run_idx=max_run_idx)
    features = []
    for intern_run_idx in range(max_run_idx-min_run_idx):
        features.append(dt_to_features (dt=data_tree["input"],lat_idx= lat_idx,lon_idx= lon_idx,r= r,Month_idx= Month_idx,hist= hist,start= start,end= end,time_step= time_step, intern_run_idx = intern_run_idx))

    output = output.sel(time=features[0].time)
    

    #estimated outcome
    output_estimated = xr.zeros_like(output)
    for intern_run_idx in range(max_run_idx-min_run_idx):
        output_estimated[output_var].loc[{"run":intern_run_idx}][:] = regr.predict(features[intern_run_idx].values)

    #resuduals
    output_residuals = output - output_estimated

    #merge
    data_tree = dt_functions.map_over_datasets(prune_location, data_tree, lat_idx, lon_idx , kwargs=None)
    data_tree = dt_functions.map_over_datasets(prune_group_ds_timespan, data_tree, kwargs = {"start" : start, "end" : end, "time_step" : "1ME" , "Month_idx" : Month_idx })


    output_estimated = output_estimated.rename({output_var: f"{output_var}_est"})
    output_residuals = output_residuals.rename({output_var: f"{output_var}_res"})
    
    data_tree["output"] = xr.DataTree()
    data_tree["output"]["real"] = xr.DataTree(output)
    data_tree["output"]["estimated"] = xr.DataTree(output_estimated)
    data_tree["output"]["residuals"] = xr.DataTree(output_residuals)


    return  xr.merge([subtree.to_dataset() for subtree in data_tree.subtree if not subtree.is_empty])

    

