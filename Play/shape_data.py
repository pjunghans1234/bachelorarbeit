import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from Play import confic as conf

xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)



def show_data_set(var = "tas", scen = "historical", test = False):
    if test :
        return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.test_run}_g025.nc')
    return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.run}_g025.nc')

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
            y = y+1
        x = x + 1
    return None

def prune_group_ds_timespan(ds, 
                        start = "1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None):
    ds = ds.sel(time=slice(start, end)).resample(time=time_step).mean()
    if Month_idx != None:
        ds = ds.sel(time= ds.time.dt.month == Month_idx)
    return ds

def load_create_datatree(scenarios = ["historical"], variables = ["tas", "pr", "mrsol"], 
                        start = "1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None, 
                        test = False):
    
    dt = xr.DataTree()

    for scen in scenarios:
        dt[scen] = xr.DataTree()
        for var in variables:
            ds = show_data_set(var=var, scen=scen, test = test)
            ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
            ds_pruned = prune_group_ds_timespan(ds, start= start, end= end, time_step = time_step, Month_idx = Month_idx)
            dt[scen][var] = xr.DataTree(ds_pruned)

    return dt

def dt_to_features (dt, 
                        lat_idx , lon_idx, r = 0, 
                        Month_idx = None, hist = 1, 
                        prune = False, start = "1850-01-01", end = "1900-01-01", time_step = "1ME"):
    
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
    
    features_arr = data_ds.to_array().transpose("time","variable","hist","lat","lon").stack(features=("variable","hist","lat","lon"))

    return features_arr

    
