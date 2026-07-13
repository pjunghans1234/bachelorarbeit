import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import calendar
import numpy as np
import math
import xarray as xr
import sklearn


from sklearn.linear_model import LinearRegression
from Play import shape_data as shape
from Play import plot_one_var
from Play import dt_functions 
from Play import LinReg_analise
from Play import LinReg_building
from Play import shape_data





def single_residuals_mass_scatter(scen = "historical", var = "mrsol",
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0], r = 0,
                    mon = 1, hist = 1,
                    location = "",
                    save = False):
    
    if scen == "historical":
        start = "1850-01-01"
        end = "1900-01-01"
    else : 
        start = "2000-01-01"
        end = "2100-01-01"

    if var == "mrsol":
    
        for depth in depth_arr:
            
            lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist)
            residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, number_of_runs = 10)


            fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True, figsize = (5,4*3))  #grid
                                        
            xr.plot.scatter(residuals, x="tas", y = "pr", hue = "mrsol_res", ax = axs[0,0], cmap = "viridis")      

            xr.plot.scatter(residuals, x="tas", y = "mrsol_res", hue = "pr", ax = axs[1,0])    

            xr.plot.scatter(residuals, x="pr", y = "mrsol_res", hue = "tas", ax = axs[2,0])

            if save:
                plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}/depth = {depth}", name_postfix = f"/scatter lat_idx = {lat_idx} lon_idx = {lon_idx}")
        
    else :
        lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx, r = r,start = start, end = end, Month_idx = mon, hist = hist)
        residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx, r = r,start = start, end = end, Month_idx = mon, hist = hist)


        fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True, figsize = (5,4*3))  #grid
                                    
        xr.plot.scatter(residuals, x="tas", y = "pr", hue = f"{var}_res", ax = axs[0,0],  cmap = "viridis")      

        xr.plot.scatter(residuals, x="tas", y = f"{var}_res", hue = "pr", ax = axs[1,0])    

        xr.plot.scatter(residuals, x="pr", y = f"{var}_res", hue = "tas", ax = axs[2,0])

        if save:
            plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}", name_postfix = f"/scatter lat_idx = {lat_idx} lon_idx = {lon_idx}")
        
                            


def plot_histogram():
    return "done"









def all_residuals_mass_scatter(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0,1,2,3,4], r_arr = [0],
                    months = [1,7], hist_arr = [1],
                    location = "",
                    save = False, all = False):
    
    for scen in scenarios:
        if scen == "historical":
            start = "1850-01-01"
            end = "1900-01-01"
        else : 
            start = "2000-01-01"
            end = "2100-01-01"
        for mon in months:
            for r in r_arr:
                for hist in hist_arr:
                    for var in variables:
                        single_residuals_mass_scatter(scen = scen, var = var, lat_idx = lat_idx, lon_idx = lon_idx , depth_arr = depth_arr, r = r, mon = mon, hist = hist, location = location, save = save)
                        if not all:
                            return "done"
                    