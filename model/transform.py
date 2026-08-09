import numpy as np
import xarray as xr
from model import config as conf

#Does nothing is just a helper vor NB adaption speed
def none_Transform_ds(ds):
    return ds

def none_Transform_ds_inv(ds):
    return ds


#wendet die logit Tranformation auf die mrsolwerte an.
def Logit_Transform_ds(ds, var = "mrsol"):
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=var)
    ds_T = ds.copy(deep=True)
    ds_T[var] = (ds[var]/100)    
    ds_T[var] = np.log(ds_T[var]/(1-ds_T[var])) 

    units = ds_T[var].attrs.get("units")
    if units is not None:
        ds_T[var].attrs["units"] = f"logit-transform of({units}/100)"
    return ds_T

#wendet die logit Tranformation auf die mrsolwerte an.
def Logit_Transform_ds_inv(ds_T, var = "mrsol"):
    if isinstance(ds_T, xr.DataArray):
        ds_T = ds_T.to_dataset(name=var)
    ds = ds_T.copy(deep=True)
    ds[var] = 100 / (1 + np.exp(-ds_T[var]))    

    units_T = ds[var].attrs.get("units")
    if units_T is not None:
        ds[var].attrs["units"] = units_T.removeprefix("logit-transform of(").removesuffix("/100)") 
    return ds


#wendet die log Tranformation auf die mrsolwerte an.
def Log_Transform_ds(ds, var = "mrsol"):
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=var)
    ds_T = ds.copy(deep=True)
    ds_T[var] = np.log(ds[var]) 

    units = ds_T[var].attrs.get("units")
    if units is not None: 
        ds_T[var].attrs["units"] = f"log({units})"
    return ds_T

#wendet die log Tranformation auf die mrsolwerte an.
def Log_Transform_ds_inv(ds_T, var = "mrsol"):
    if isinstance(ds_T, xr.DataArray):
        ds_T = ds_T.to_dataset(name=var)
    ds = ds_T.copy(deep=True)
    ds[var] = np.exp(ds_T[var])  
    
    units_T = ds[var].attrs.get("units")
    if units_T is not None:
        ds[var].attrs["units"] = units_T.removeprefix("log(").removesuffix(")")
    return ds
