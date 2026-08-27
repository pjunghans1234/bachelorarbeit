import numpy as np
import xarray as xr
from model import config as conf
from PIL import Image



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

def load_plot(var = "", scen = "",folder = "", name_prefix = "", name = "", name_postfix = ""):

    if folder != "":
        file_path = conf.path_to_plot_output / folder
    else:
        file_path = conf.path_to_plot_output

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

    return Image.open(file_path / f'plot{var}{scen}{name}{name_postfix}.png')




def prune_group_ds_timespan(ds, 
                            start = None, end = None, time_step = "1ME", Month_idx = None):
    ds = ds.sel(time=slice(start, end)).resample(time=time_step).mean()
    if Month_idx != None:
        ds = ds.sel(time= ds.time.dt.month == Month_idx)
    return ds

def make_concat_set(var = "tas", scen = "historical", set = "all_runs", min_run_idx = 11, max_run_idx = 21, local = True, model = "MPI-ESM1-2-LR", timescale = "mon", min_run = None, max_run = None):
    if local:
        data_sets = [xr.load_dataset(conf.path_to_local_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc')  for run in getattr(conf, set)[min_run_idx:max_run_idx]]
    else:
        if min_run is None:
            min_run = conf.all_runs[min_run_idx]
        if max_run is None:
            max_run = conf.all_runs[max_run_idx]
        data_sets = [xr.load_dataset(conf.path_to_online_data / f'{var}/{timescale}/g025/{var}_{timescale}_{model}_{scen}_{run}_g025.nc')  for run in getattr(conf, set)[min_run_idx:max_run_idx]]

    return xr.concat(data_sets, "run")


def add_hist_dimension(ds, hist):
    history = []
    for i in range(0, hist):
        tmp = ds.shift(time=i)
        history.append(tmp)
        
    return xr.concat(history, "hist").assign_coords(hist=np.arange(hist))


def add_radius_dimension(ds, r):
	lat_translations = []
	for lat_trans in range(-r,r+1):
		lat_temp = ds.roll(lat=lat_trans)		#This is not a really correct way of handling edgecases but thanks to the dataholes at North and southpol, the only Risk is get some usless data, if one is working with large influece radius. 
		lon_translations = []
		for lon_trans in range(-r,r+1):
			lon_temp = lat_temp.roll(lon = lon_trans)
			lon_translations.append(lon_temp)
		lat_temp = xr.concat(lon_translations, "lon_translations").assign_coords(lon_translations=np.arange(-r,r+1))
		lat_translations.append(lat_temp)
	return xr.concat(lat_translations, "lat_translations").assign_coords(lat_translations=np.arange(-r,r+1))
    
