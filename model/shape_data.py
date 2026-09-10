import numpy as np
import xarray as xr
from model import config as conf
from PIL import Image

def prune_group_ds_timespan(ds, 
                            start = None, end = None, time_step = "1ME", Month_idx = None):
    ds = ds.sel(time=slice(start, end)).resample(time=time_step).mean()
    if Month_idx != None:
        ds = ds.sel(time= ds.time.dt.month == Month_idx)
    return ds


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
