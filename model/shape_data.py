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


def load_params(var = "", scen = "",folder = "", name_prefix = "", name = "", name_postfix = ""):

    if folder != "":
        file_path = conf.path_to_parametere_output / folder
    else:
        file_path = conf.path_to_parametere_output

    if var != "":
        var = "_" + var
    if scen != "":
        scen = "_" + scen
    if name_prefix != "":
        name_prefix = "_" +  name_prefix  
    if name != "":
        name = "_" + name
    if name_postfix != "":
        name_postfix = "_" + name_postfix

    return xr.load_dataset(file_path / f'mean_paramerters{var}{scen}{name}{name_postfix}.nc'), xr.load_dataset(file_path / f'variance_paramerters{var}{scen}{name}{name_postfix}.nc')

def load_scores(var = "", scen = "",folder = "", name_prefix = "", name = "", name_postfix = ""):

    if folder != "":
        file_path = conf.path_to_scores_output / folder
    else:
        file_path = conf.path_to_scores_output

    if var != "":
        var = "_" + var
    if scen != "":
        scen = "_" + scen
    if name_prefix != "":
        name_prefix = "_" +  name_prefix  
    if name != "":
        name = "_" + name
    if name_postfix != "":
        name_postfix = "_" + name_postfix

    return xr.load_dataset(file_path / f'crps_scores{var}{scen}{name}{name_postfix}.nc')


def prune_group_ds_timespan(ds, 
                            start = None, end = None, time_step = "1ME", Month_idx = None):
    ds = ds.sel(time=slice(start, end)).resample(time=time_step).mean()
    if Month_idx != None:
        ds = ds.sel(time= ds.time.dt.month == Month_idx)
    return ds