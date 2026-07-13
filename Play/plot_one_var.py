import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from Play import config as conf

xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)

def show_data_set(var = "tas", scen = "historical", test = False):
    if test :
        return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.test_run}_g025.nc')
    return xr.load_dataset(conf.path_to_data / f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{conf.run}_g025.nc')

def save_plot(fig, var, scen,name_prefix = "", name = "", name_postfix = ""):
    if name != "":
        file_path = conf.path_to_output / name
    else:
        file_path = conf.path_to_output

    if name_prefix != "":
        name_prefix = "_" +  name_prefix  
    if name != "":
        name = "_" + name
    if name_postfix != "":
        name_postfix = "_" + name_postfix  
        
    out = file_path / f'plot_{var}_{scen}{name}{name_postfix}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)

def create_cell_area_weights_array(ds, var):
    # Earth's average radius in meters
    R = 6.371e6

    da = getattr(ds, var)
    lon = da.lon
    lat = da.lat

    dlon = R * np.gradient(np.deg2rad(lon))
    dlat = R * np.gradient(np.deg2rad(lat)) * np.cos(np.deg2rad(lat))

    dlon_da = xr.DataArray(dlon, coords={"lon": lon}, dims=("lon",))
    dlat_da = xr.DataArray(dlat, coords={"lat": lat}, dims=("lat",))

    return dlat_da * dlon_da


def plot_mean_over_time(var = "tas", scen = "historical", name = "", name_postfix = ""):

    ds_var = show_data_set(var=var,scen = scen)
    if var == "mrsol": #seperat da Daten auf verschiedenen Tiefen vorhanden
        getattr(ds_var, var).mean(dim="time").plot(col = "depth", col_wrap = 2, x="lon");
        plt.title(var) #der scheint nohc nicht zu functionieren
        save_plot(plt.gcf(), var=var, scen=scen, name_prefix="timemean",name=name,name_postfix = name_postfix)
    else:
        getattr(ds_var, var).mean(dim="time").plot(x="lon");
        plt.title(var) #woher kommt das automatische height = 2.0[m]
        save_plot(plt.gcf(),var=var, scen=scen, name_prefix="timemean",name=name,name_postfix = name_postfix)
    

#TapTapTap hat immerhin nicht geklappt
def plot_weighted_global_mean(var = "tas", scen = "historical", from_year = None, to_year = None, smoothed = False, depth = 1, save = False, name = "", name_postfix = ""):
    ds = show_data_set(var, scen)

    if  var == "mrsol": 
        ds = ds.isel(depth=depth) #nur eine Tiefe auswählen
    
    if smoothed: #ev. später noch mesmertool einbauen
        ds = ds.resample(time="1YE").mean()

    ds = ds.sel(time=slice(from_year, to_year))

    weights = create_cell_area_weights_array(ds, var)
    getattr(ds, var).weighted(weights).mean(["lat", "lon"]).plot(x="time");

    plt.title(f"Global Weighted Mean of {var} over time_{name}")
    if save:
        save_plot(plt.gcf(), var, scen, "global_weighted_mean", name, name_postfix)


def plot_weighted_local_mean(var = "tas", scen = "historical", min_lat = None, max_lat = None, min_lon = None, max_lon = None, from_year = None, to_year = None, smoothed = False, depth = 1, save = False, name = "", name_postfix = ""):
    ds = show_data_set(var, scen)
    ds_original = ds.copy()

    if var == "mrsol":
        ds = ds.isel(depth=depth) #nur eine Tiefe auswählen
        ds_original = ds_original.isel(depth=depth)

    if smoothed: #ev. später noch mesmertool einbauen
        ds = ds.resample(time="1YE").mean()

    ds = ds.sel(time=slice(from_year, to_year))
    ds = ds.sel(lat=slice(max_lat, max_lat))
    ds = ds.sel(lon=slice(min_lon, max_lon))

    weights = create_cell_area_weights_array(ds, var)
    getattr(ds, var).weighted(weights).mean(["lat", "lon"]).plot(x="time");

    plt.title(f"Local Weighted Mean of {var} over time_{name}")
    if save:
        save_plot(plt.gcf(), var, scen, "local_weighted_mean", name, name_postfix)

    #KI for the looks
    # World map with selected region highlighted
    ds_map = getattr(ds_original, var).isel(time=0) 
    fig, ax = plt.subplots(figsize=(10, 4))
    ds_map.plot(ax=ax, x="lon", y="lat", cmap="viridis", add_colorbar=True)

    if None not in (min_lon, max_lon, min_lat, max_lat):
        width = max_lon - min_lon
        height = max_lat - min_lat
        rect = plt.Rectangle((min_lon, min_lat), width, height,
                             edgecolor="red", facecolor="none", linewidth=2)
        ax.add_patch(rect)

    ax.set_title(f"Selected region for {var}")
    plt.show()  

def timeline_plots(var = "tas", scen = "historical", time_step = "10YE", number_of_plots = 10, depth = 1, save = False, name = "", name_postfix = ""):
    ds_var = show_data_set(var=var, scen=scen)
    ds_var_resample = ds_var.resample(time=time_step).mean()
    ds_var_resample = ds_var_resample.isel(time=slice(-number_of_plots, None))
        
    #seperat da Daten auf verschiedenen Tiefen vorhanden    
    if  var == "mrsol": 
        ds_var_resample = ds_var_resample.isel(depth=depth) #nur eine Tiefe auswählen
     
    #Timeline Absolutwerte:
    getattr(ds_var_resample, var).plot(col="time", col_wrap = 4, x = "lon");
    plt.title("temparature timeline") #warum macht der nichts?
    plt.show()
    if save:
        save_plot(plt.gcf(),var= var,scen= scen,name_prefix= "timeline",name= name,name_postfix= name_postfix)

    #Timeline Differenzen:
    ds_var_resample_diff = ds_var_resample - ds_var_resample.isel(time=0) #Jahrzehnt 1850
    getattr(ds_var_resample_diff, var).plot(col="time", col_wrap = 4, x = "lon");  
    plt.title("temparature timeline")
    plt.show()
    if save:
        save_plot(plt.gcf(),var= var,scen= scen,name_prefix= "timeline",name= name,name_postfix= name_postfix)


def position_helper(min_lat = None, max_lat = None, min_lon = None, max_lon = None, min_lat_idx = None, max_lat_idx = None, min_lon_idx = None, max_lon_idx = None):
    ds = show_data_set("mrsol")
    ds = ds.isel(depth=0)

    ds_map = getattr(ds, "mrsol").isel(time=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ds_map.plot(ax=ax, x="lon", y="lat", cmap="viridis", add_colorbar=True)


    if None not in (min_lon_idx, max_lon_idx, min_lat_idx, max_lat_idx):
        min_lat = ds.lat[min_lat_idx]
        max_lat = ds.lat[max_lat_idx]
        min_lon = ds.lon[min_lon_idx]
        max_lon = ds.lon[max_lon_idx]
    
    if None not in (min_lon, max_lon, min_lat, max_lat):
        width = max_lon - min_lon
        height = max_lat - min_lat
        rect = plt.Rectangle((min_lon, min_lat), width, height,
                            edgecolor="red", facecolor="none", linewidth=2)
        ax.add_patch(rect)

        ax.set_title(f"Selected region")
    
        plt.show()  

    else:
        print ("o, your coordinates where not sufitient")


def show_my_position(min_lat = None, max_lat = None, min_lon = None, max_lon = None, min_lat_idx = None, max_lat_idx = None, min_lon_idx = None, max_lon_idx = None,  lat_idx = None, lon_idx = None):
    
    #wenn nur lat_idx gegen, wird es als anfang interpretiert
    if lat_idx != None:
        min_lat_idx = lat_idx
    if lon_idx != None:
        min_lon_idx = lon_idx


    if max_lat == None and min_lat != None: 
        max_lat = min_lat + 15
    if max_lon == None and min_lon != None: 
        max_lon = min_lon + 15
    if max_lat_idx == None and min_lat_idx != None: 
        max_lat_idx = min_lat_idx + 1
    if max_lon_idx == None and max_lon_idx != None: 
        max_lon_idx = max_lon_idx + 1
    position_helper(min_lat= min_lat,max_lat= max_lat ,min_lon= min_lon ,max_lon= max_lon ,min_lat_idx= min_lat_idx ,max_lat_idx= max_lat_idx ,min_lon_idx= min_lon_idx ,max_lon_idx= max_lon_idx)
    