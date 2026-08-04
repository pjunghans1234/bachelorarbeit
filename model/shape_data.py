import numpy as np
import xarray as xr
from model import config as conf




#Läd beispiel datensatz
def load_data_set(var = "tas", scen = "historical",run_idx = 1, local = True, model = "MPI-ESM1-2-LR", timescale = "mon", run = None):
    if local:
        return xr.load_dataset(conf.path_to_local_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.all_runs[run_idx]}_g025.nc')
    else:
        if run is None:
            run = conf.all_runs[run_idx]  
        return  xr.load_dataset(conf.path_to_online_data / f"{var}/{timescale}/g025/{var}_{timescale}_{model}_{scen}_{run}_g025.nc")



"""Vermutlich eher zu "multifunktional" wird also vermutlich gelöscht"""
#Läd für gewünschte Angaben das ensprechende Dataset und gibt dies zurück
def show_data_set(var = "tas", scen = "historical", test = False,run_idx = 1, Transform = None):
    if Transform == "Logit":
        return Logit_Transform_ds(xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.all_runs[run_idx]}_g025.nc'))
    if Transform == "Log":
            return Log_Transform_ds(xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.all_runs[run_idx]}_g025.nc'))
    if test :
        return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.test_run}_g025.nc')
    return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.all_runs[run_idx]}_g025.nc')

#Läd Daten aus verschiedenen Runs und packt sie zusammen zu einem Datenset
def make_concat_set(var = "tas", scen = "historical", set = "all_runs", min_run_idx = 11, max_run_idx = 21, Transform = None):
    if Transform == "Logit":
        data_sets = [Logit_Transform_ds(xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc'))  for run in getattr(conf, set)[min_run_idx:max_run_idx]]
    if Transform == "Log":
            data_sets = [Log_Transform_ds(xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc'))  for run in getattr(conf, set)[min_run_idx:max_run_idx]]
        
    else : 
        data_sets = [xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc')  for run in getattr(conf, set)[min_run_idx:max_run_idx]]

    #merged variante
    #for i in range(0,10):
    #    data_sets[i] = data_sets[i].rename({var: f"{var}_{i}"})
    #return xr.merge(data_sets, compat = "identical")

    return xr.concat(data_sets, "run")


def prune_group_ds_timespan(ds, 
                            start = None, end = None, time_step = "1ME", Month_idx = None):
    ds = ds.sel(time=slice(start, end)).resample(time=time_step).mean()
    if Month_idx != None:
        ds = ds.sel(time= ds.time.dt.month == Month_idx)
    return ds