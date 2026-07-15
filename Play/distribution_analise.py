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




def single_total_histogramm(scen = "historical", var = "mrsol",
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0], r = 0,
                    mon = 1, hist = 1,
                    run_idx = 1, min_run_idx = 11, max_run_idx = 21):
    if scen == "historical":
        start = "1850-01-01"
        end = "1900-01-01"
    else : 
        start = "2000-01-01"
        end = "2100-01-01"



    for depth in depth_arr:
        lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, run_idx=run_idx)
        residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx=min_run_idx, max_run_idx=max_run_idx)

        fig , ax = plt.subplots(
        1,2,figsize = (12,6)
        )

        ax[0].hist(residuals[var].values.flatten(), bins = 50)
        ax[0].set_title("real_values")

        ax[1].hist(residuals[f"{var}_res"].values.flatten(), bins = 50)
        ax[1].set_title("residuals")

        fig.show()

    return (lin_regr, residuals, fig, ax) 


#quick and dirty
def single_residual_split_histogram(scen = "historical", var = "mrsol",
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0], r = 0,
                    mon = 1, hist = 1,
                    location = "",
                    save = False, run_idx = 1, min_run_idx = 11, max_run_idx = 21):
    
    if scen == "historical":
        start = "1850-01-01"
        end = "1900-01-01"
    else : 
        start = "2000-01-01"
        end = "2100-01-01"

    
    for depth in depth_arr:
        lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, run_idx=run_idx)
        residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx=min_run_idx, max_run_idx=max_run_idx)

        #plot
        da_tas = residuals["tas"]
        da_pr = residuals["pr"]
        da_res = residuals[f"{var}_res"]
        da_abs = residuals[var]

        tas_edges = np.quantile(da_tas.values[np.isfinite(da_tas)], np.linspace(0, 1, 5))
        pr_edges = np.quantile(da_pr.values[np.isfinite(da_pr)], np.linspace(0, 1, 5))

        fig_res, axes_res = plt.subplots(
            4,
            4,
            figsize=(12, 12),
            sharex=True,
            sharey=True
        )
        fig_abs, axes_abs = plt.subplots(
            4,
            4,
            figsize=(12, 12),
            sharex=True,
            sharey=True
        )


        for tas_idx in range(3,-1,-1):  #Rückwärts, das in fig die Temperatur nach oben zu nimmt
            for pr_idx in range(4):

                mask = (
                    (da_tas >= tas_edges[tas_idx]) &
                    (da_tas < tas_edges[tas_idx+1]) &
                    (da_pr >= pr_edges[pr_idx]) &
                    (da_pr < pr_edges[pr_idx+1])
                )

                res_values = da_res.where(mask).values.flatten()
                abs_values = da_abs.where(mask).values.flatten()
                res_values = res_values[np.isfinite(res_values)]
                abs_values = abs_values[np.isfinite(abs_values)]


                axes_res[tas_idx,pr_idx].hist(res_values, bins=30)
                axes_res[tas_idx, pr_idx].set_title(
                    f"tas:{tas_edges[tas_idx]:.4g} to {tas_edges[tas_idx+1]:.4g},\n pr:{pr_edges[pr_idx]:.4g} to {pr_edges[pr_idx + 1]:.4g}\n N={len(res_values)}"
            )
                axes_abs[tas_idx, pr_idx].hist(abs_values, bins=30)
                axes_abs[tas_idx, pr_idx].set_title(
                    f"tas:{tas_edges[tas_idx]:.4g} to {tas_edges[tas_idx+1]:.4g},\n pr:{pr_edges[pr_idx]:.4g} to {pr_edges[pr_idx + 1]:.4g}\n N={len(abs_values)}"
            )

        fig_res.suptitle("residuals_histogram")
        fig_res.tight_layout()
        fig_res.show()

        fig_abs.suptitle("absolute_histogram")
        fig_abs.tight_layout()
        fig_abs.show()
        if var == "mrsol":
            name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}/depth = {depth}"
        else:
            name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}"
        if save:
            plot_one_var.save_plot(fig=fig_res,var= var,scen = scen ,name_prefix = var,  name = name, name_postfix = f"/res_histogramm lat_idx = {lat_idx} lon_idx = {lon_idx}")
            plot_one_var.save_plot(fig=fig_abs,var= var,scen = scen ,name_prefix = var,  name = name, name_postfix = f"/abs_histogramm lat_idx = {lat_idx} lon_idx = {lon_idx}")
    
        
def all_residual_histograms(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0,1,2,3,4], r_arr = [0],
                    months = [1,7], hist_arr = [1],
                    location = "",
                    save = False, all = False, run_idx = 1, min_run_idx = 11, max_run_idx = 21):
    
    for scen in scenarios:
        for mon in months:
            for r in r_arr:
                for hist in hist_arr:
                    for var in variables:
                        if var == "mrsol":
                            single_residual_split_histogram(scen = scen, var = var, lat_idx = lat_idx, lon_idx = lon_idx , depth_arr = depth_arr, r = r, mon = mon, hist = hist, location = location, save = save, run_idx=run_idx,min_run_idx=min_run_idx, max_run_idx=max_run_idx)
                        else:
                            single_residual_split_histogram(scen = scen, var = var, lat_idx = lat_idx, lon_idx = lon_idx , depth_arr = [0], r = r, mon = mon, hist = hist, location = location, save = save, run_idx=run_idx,min_run_idx=min_run_idx, max_run_idx=max_run_idx)
                        
                        if not all:
                            return "done"
    
                        plt.close('all')         



def single_residuals_mass_scatter(scen = "historical", var = "mrsol",
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0], r = 0,
                    mon = 1, hist = 1,
                    location = "",
                    save = False, run_idx = 1, min_run_idx = 11, max_run_idx = 21):
    
    if scen == "historical":
        start = "1850-01-01"
        end = "1900-01-01"
    else : 
        start = "2000-01-01"
        end = "2100-01-01"

    if var == "mrsol":
    
        for depth in depth_arr:
            
            lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, run_idx=run_idx)
            residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist,min_run_idx=min_run_idx, max_run_idx=max_run_idx)


            fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True, figsize = (5,4*3))  #grid
                                        
            xr.plot.scatter(residuals, x="tas", y = "pr", hue = "mrsol_res", ax = axs[0,0], cmap = "viridis")      

            xr.plot.scatter(residuals, x="tas", y = "mrsol_res", hue = "pr", ax = axs[1,0])    

            xr.plot.scatter(residuals, x="pr", y = "mrsol_res", hue = "tas", ax = axs[2,0])

            if save:
                plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}/depth = {depth}", name_postfix = f"/mass_scatter lat_idx = {lat_idx} lon_idx = {lon_idx}")
        
    else :
        lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx, r = r,start = start, end = end, Month_idx = mon, hist = hist, run_idx=run_idx)
        residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx, r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx=min_run_idx, max_run_idx=max_run_idx)


        fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True, figsize = (5,4*3))  #grid
                                    
        xr.plot.scatter(residuals, x="tas", y = "pr", hue = f"{var}_res", ax = axs[0,0],  cmap = "viridis")      

        xr.plot.scatter(residuals, x="tas", y = f"{var}_res", hue = "pr", ax = axs[1,0])    

        xr.plot.scatter(residuals, x="pr", y = f"{var}_res", hue = "tas", ax = axs[2,0])

        if save:
            plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}", name_postfix = f"/mass_scatter lat_idx = {lat_idx} lon_idx = {lon_idx}")
        








def all_residuals_mass_scatter(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0,1,2,3,4], r_arr = [0],
                    months = [1,7], hist_arr = [1],
                    location = "",
                    save = False, all = False,  run_idx = 1, min_run_idx = 11, max_run_idx = 21):
    
    for scen in scenarios:
        for mon in months:
            for r in r_arr:
                for hist in hist_arr:
                    for var in variables:
                        single_residuals_mass_scatter(scen = scen, var = var, lat_idx = lat_idx, lon_idx = lon_idx , depth_arr = depth_arr, r = r, mon = mon, hist = hist, location = location, save = save, run_idx=run_idx, min_run_idx=min_run_idx, max_run_idx=max_run_idx)
                        if not all:
                            return "done"
                        plt.close('all')


