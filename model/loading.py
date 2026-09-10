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

    files = {"mean" :file_path / f'mean_paramerters{var}{scen}{name}{name_postfix}.nc',"variance": file_path / f'variance_paramerters{var}{scen}{name}{name_postfix}.nc'}

    helper_files = {"skew":file_path / f'skew_paramerters{var}{scen}{name}{name_postfix}.nc', 
    "local_maximas": file_path / f'local_maximas{var}{scen}{name}{name_postfix}.nc', 
    "chunk_mask": file_path / f'chunk_mask{var}{scen}{name}{name_postfix}.nc',
    "detail_mask": file_path / f'detail_mask{var}{scen}{name}{name_postfix}.nc'}

    for key, file in files.items():
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist.")
    for key, file in helper_files.items():
        if file.exists():
            files[key] = file

    return {key: xr.load_dataset(file) for key, file in files.items()}

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





def load_make_concat_set(var = "tas", scen = "historical", set = "all_runs", min_run_idx = 11, max_run_idx = 21, local = True, model = "MPI-ESM1-2-LR", timescale = "mon", min_run = None, max_run = None):
    if local:
        data_sets = [xr.load_dataset(conf.path_to_local_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc')  for run in getattr(conf, set)[min_run_idx:max_run_idx]]
    else:
        if min_run is None:
            min_run = conf.all_runs[min_run_idx]
        if max_run is None:
            max_run = conf.all_runs[max_run_idx]
        data_sets = [xr.load_dataset(conf.path_to_online_data / f'{var}/{timescale}/g025/{var}_{timescale}_{model}_{scen}_{run}_g025.nc')  for run in getattr(conf, set)[min_run_idx:max_run_idx]]

    return xr.concat(data_sets, "run")


    
