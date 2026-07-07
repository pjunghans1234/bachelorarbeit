import importlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


from sklearn.linear_model import LinearRegression
from Play import shape_data as shape
from Play import datatree 


def local_datatree_to_one_dim(dtree, output_ds, output_var = "mrsol",
                            lat_idx = 30, lon_idx = 10, depth = 0, r = 0,
                            Month_idx = None, hist = 1, 
                            prune = False, start ="1850-01-01", end = "1890-01-01", time_step = "1ME",
                            max_depth = 5, randomstate = 0 ):

    input_arr = shape.dt_to_features(dt= dtree,lat_idx= lat_idx,lon_idx= lon_idx,r= r,Month_idx= Month_idx,hist= hist,prune= prune,start =start,end= end,time_step= time_step)
    
    output_ds = output_ds.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")

    if output_var == "mrsol" :
        output_ds = output_ds.isel(depth = depth)

 
    output_ds = output_ds.isel(lat=lat_idx,lon=lon_idx)

    output_ds = output_ds.sel(time=input_arr.time)

    #return input_arr
    regr = LinearRegression()
    regr.fit(input_arr.values, output_ds.to_array().T.values.squeeze())
    
    return regr
    


def local_vars_to_one_dim(scen = "historical", variables = ["tas", "pr"], output_var = "mrsol",
                            lat_idx = 30, lon_idx = 10, depth = 0, r = 0,
                            start ="1850-01-01", end = "1890-01-01", time_step = "1ME",  Month_idx = None, hist = 1, 
                            max_depth = 5, randomstate = 0):
    start_hist = shape.start_with_hist(start= start,time_step= time_step,hist= hist)
    input_dt = shape.load_create_datatree(scenarios= [scen],variables= variables,start= start_hist,end= end,time_step= time_step)
    output_ds = shape.show_data_set(var= output_var,scen= scen)
    output_ds = shape.prune_group_ds_timespan(ds= output_ds,start= start,end= end,time_step=time_step, Month_idx = Month_idx)
    #return input_dt, output_ds
    return local_datatree_to_one_dim(dtree=input_dt,output_ds= output_ds,output_var= output_var,lat_idx= lat_idx,lon_idx= lon_idx,max_depth= max_depth,randomstate= randomstate,depth= depth,r= r,Month_idx= Month_idx,hist= hist)



def global_datatree_to_one_dim(dtree, output_ds, output_var = "mrsol",
                            min_lat_idx = 0, max_lat_idx=40,min_lon_idx = 0, max_lon_idx = 40, depth = 0, r = 0,
                            hist = 1, 
                            prune = False, start ="1850-01-01", end = "1890-01-01", time_step = "1ME",  Month_idx = None,time_bar = False,  
                            max_depth = 5, randomstate = 0 ):
    regr_mat = []

    for dlat_idx in range (min_lat_idx,max_lat_idx):
        new_row = []
        for dlon_idx in range(min_lon_idx,max_lon_idx):
            try:
                new_row.append(local_datatree_to_one_dim(dtree=dtree,output_ds= output_ds,output_var = output_var, lat_idx=dlat_idx, lon_idx=dlon_idx,max_depth=max_depth, randomstate=randomstate, r = r, depth = depth, hist = hist, prune = prune, start =start, end = end, time_step = time_step, Month_idx = Month_idx))
            except Exception :
                new_row.append(None)
        regr_mat.append(new_row)
        if time_bar :
            print(f"from {min_lat_idx} to {max_lat_idx}, dlat_idx {dlat_idx} is now done.")
        
    return regr_mat



def global_vars_to_one_dim(scen = "historical", variables = ["tas", "pr"], output_var = "mrsol",
                            min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0, max_lon_idx = 40, depth = 0,r = 0,
                            start ="1850-01-01", end = "1890-01-01", time_step = "1ME", Month_idx = None,  hist = 1,
                            time_bar = False,
                            max_depth = 5, randomstate = 0):
    start_hist = shape.start_with_hist(start=start,time_step= time_step,hist= hist)
    input_dt = shape.load_create_datatree(scenarios=[scen],variables= variables,start= start_hist,end= end,time_step= time_step)
    output_ds = shape.show_data_set(var=output_var,scen=scen)
    output_ds = shape.prune_group_ds_timespan(ds=output_ds,start=start,end=end,time_step=time_step,Month_idx= Month_idx)
    return global_datatree_to_one_dim(dtree=input_dt,output_ds= output_ds,output_var= output_var,max_depth= max_depth,randomstate= randomstate,min_lat_idx=  min_lat_idx,max_lat_idx= max_lat_idx,min_lon_idx= min_lon_idx,max_lon_idx= max_lon_idx,time_bar= time_bar, r = r,  depth = depth, hist = hist,  Month_idx = Month_idx, time_step=time_step)


'''

def local_datatree_to_one_dim(dtree, output_ds, output_var = "mrsol", lat_idx = 30, lon_idx = 0, max_depth = 5, randomstate = 0, depth = 0, r = 0):

    #stark an collapse angelehnt aber halt merge
    datasets = [subtree.to_dataset() for subtree in dtree.subtree if not subtree.is_empty]

    input_ds = xr.merge(datasets).isel(lat=slice(lat_idx-r,lat_idx+r+1),lon=slice(lon_idx-r,lon_idx+r+1))

    output_ds = output_ds.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")

    if output_var == "mrsol" :
        output_ds = output_ds.isel(depth = depth)


    input_arr = input_ds.to_array().transpose("time","variable","lat","lon").stack(features=("variable","lat","lon"))

 
    output_ds = output_ds.isel(lat=lat_idx,lon=lon_idx)

    regr = LinearRegression(max_depth=max_depth, random_state= randomstate)
    regr.fit(input_arr.values, output_ds.to_array().T.values.squeeze())
    
    return regr
    



def local_vars_to_one_dim(scen = "historical", variables = ["tas", "pr"], output_var = "mrsol", lat_idx = 30, lon_idx = 0, start ="1850-01-01", end = "1890-01-01", timestep = "1ME", max_depth = 5, randomstate = 0, depth = 0, Month_idx = None):
    input_dt = shape.load_create_datatree([scen], variables, start, end, timestep, Month_idx = Month_idx)
    output_ds = shape.show_data_set(output_var,scen)
    output_ds = shape.prune_group_ds_timespan(output_ds,start,end,timestep, Month_idx = Month_idx)
    return local_datatree_to_one_dim(input_dt, output_ds, output_var, lat_idx, lon_idx, max_depth, randomstate, depth)

def global_datatree_to_one_dim(dtree, output_ds, output_var = "mrsol", max_depth = 5, randomstate = 0,min_lat_idx = 0, max_lat_idx=40,min_lon_idx = 0, max_lon_idx = 40, time_bar = False, r = 0, depth = 0):
    regr_mat = []

    for dlat_idx in range (min_lat_idx,max_lat_idx):
        new_row = []
        for dlon_idx in range(min_lon_idx,max_lon_idx):
            try:
                new_row.append(local_datatree_to_one_dim(dtree, output_ds, output_var = output_var, lat=dlat_idx, lon=dlon_idx,max_depth=max_depth, randomstate=randomstate, r = r, depth = depth))
            except Exception :
                new_row.append(None)
        regr_mat.append(new_row)
        if time_bar :
            print(f"from {min_lat_idx} to {max_lat_idx}, dlat_idx {dlat_idx} is now done.")
        
    return regr_mat

def global_vars_to_one_dim(scen = "historical", variables = ["tas", "pr"], output_var = "mrsol",  start ="1850-01-01", end = "1890-01-01", timestep = "1ME", max_depth = 5, randomstate = 0, min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0, max_lon_idx = 40, time_bar = False, Month_idx = None, r = 0, depth = 0):
    input_dt = shape.load_create_datatree([scen], variables, start, end, timestep, Month_idx = Month_idx)
    output_ds = shape.show_data_set(output_var,scen)
    output_ds = shape.prune_group_ds_timespan(output_ds,start,end,timestep, Month_idx = Month_idx)
    return global_datatree_to_one_dim(input_dt, output_ds, output_var, max_depth, randomstate,  min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx, time_bar, r = r,  depth = depth)

'''
