import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import calendar
import numpy as np
import math
import xarray as xr
import sklearn
from scipy.stats import (
    norm,
    gamma,
    beta,
    expon,
    weibull_min,
    lognorm,
    uniform,
    t,
    pareto
)


from sklearn.linear_model import LinearRegression
from Play import shape_data as shape
from Play import plot_one_var
from Play import dt_functions 
from Play import LinReg_analise
from Play import LinReg_building
from Play import shape_data
from Play import distribution_building




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
def single_residual_split_histogram(lin_regr = None, scen = "historical", var = "mrsol",
                    lat_idx = 16, lon_idx = 32 , depth = 0, r = 0,
                    start = None, end = None, mon = 1, hist = 1,
                    location = "",
                    save = False, 
                    run_idx = 1, min_run_idx = 11, max_run_idx = 21,
                    tas_quantiles = 4, pr_quantiles = 4):
    
    if scen == "historical" and start == None :
        start = "1850-01-01"
    if scen == "historical" and end == None : 
        end = "1900-01-01"

    else :
        if start == None :
            start = "2000-01-01"
        if end == None :
            end = "2100-01-01"

    
    
    if lin_regr == None:
        lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, run_idx=run_idx)
    residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx=min_run_idx, max_run_idx=max_run_idx)

    #plot
    da_tas = residuals["tas"]
    da_pr = residuals["pr"]
    da_res = residuals[f"{var}_res"]
    da_abs = residuals[var]

    tas_edges = np.quantile(da_tas.values[np.isfinite(da_tas)], np.linspace(0, 1, tas_quantiles+1))
    pr_edges = np.quantile(da_pr.values[np.isfinite(da_pr)], np.linspace(0, 1, pr_quantiles+1))

    fig_res, axes_res = plt.subplots(
        tas_quantiles,
        pr_quantiles,
        figsize=(12, 12),
        sharex=True,
        sharey=True
    )
    

    fig_abs, axes_abs = plt.subplots(
        tas_quantiles,
        pr_quantiles,
        figsize=(12, 12),
        sharex=True,
        sharey=True,

    )
    
    

    for tas_idx in range(tas_quantiles): 
        for pr_idx in range(pr_quantiles):

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

            
            axes_abs[tas_quantiles-tas_idx-1, pr_idx].hist(abs_values, bins=30, density = True)  #Rückwärts, das in fig die Temperatur nach oben zu nimmt
            axes_abs[tas_quantiles-tas_idx-1, pr_idx].set_title(
                f"tas:{tas_edges[tas_idx]:.4g} to {tas_edges[tas_idx+1]:.4g},\n pr:{pr_edges[pr_idx]:.4g} to {pr_edges[pr_idx + 1]:.4g}\n N={len(abs_values)}"
        )
            axes_res[tas_quantiles-tas_idx-1,pr_idx].hist(res_values, bins=30, density = True)
            axes_res[tas_quantiles-tas_idx-1, pr_idx].set_title(
                f"tas:{tas_edges[tas_idx]:.4g} to {tas_edges[tas_idx+1]:.4g},\n pr:{pr_edges[pr_idx]:.4g} to {pr_edges[pr_idx + 1]:.4g}\n N={len(res_values)}"
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
        plot_one_var.save_plot(fig=fig_abs,var= var,scen = scen ,name_prefix = var,  name = name, name_postfix = f"/abs_histogramm lat_idx = {lat_idx} lon_idx = {lon_idx}")
        plot_one_var.save_plot(fig=fig_res,var= var,scen = scen ,name_prefix = var,  name = name, name_postfix = f"/res_histogramm lat_idx = {lat_idx} lon_idx = {lon_idx}")
            
        
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
                            for depth in depth_arr:
                                    single_residual_split_histogram(scen = scen, var = var, lat_idx = lat_idx, lon_idx = lon_idx , depth = depth, r = r, mon = mon, hist = hist, location = location, save = save, run_idx=run_idx,min_run_idx=min_run_idx, max_run_idx=max_run_idx)
                                    if not all:
                                        return "done"
                        else:
                            single_residual_split_histogram(scen = scen, var = var, lat_idx = lat_idx, lon_idx = lon_idx, r = r, mon = mon, hist = hist, location = location, save = save, run_idx=run_idx,min_run_idx=min_run_idx, max_run_idx=max_run_idx)
                        
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


    



#quick and dirty  ToDo sauberere Version
def single_residual_split_histogram_vs_mixed_distribution(dist_pred, scen = "historical", output_var = "mrsol",
                                                        lat_idx = 16, lon_idx = 32 , depth = 0, r = 0,
                                                        start = None, end = None, mon = 1, hist = 1,
                                                        location = "",
                                                        save = False, min_run_idx = 11, max_run_idx = 21,
                                                        tas_quantiles = 4, pr_quantiles = 4):
    
    if scen == "historical" and start == None :
        start = "1850-01-01"
    if scen == "historical" and end == None : 
        end = "1900-01-01"

    else :
        if start == None :
            start = "2000-01-01"
        if end == None :
            end = "2100-01-01"

    
    
   
    lin_regr = dist_pred[0]
    var_regr = dist_pred[1]
    residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = output_var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx=min_run_idx, max_run_idx=max_run_idx)

    #plot
    da_tas = residuals["tas"]
    da_pr = residuals["pr"]
    da_res = residuals[f"{output_var}_res"]
    da_est = residuals[f"{output_var}_est"]
    da_abs = residuals[output_var]

    tas_edges = np.quantile(da_tas.values[np.isfinite(da_tas)], np.linspace(0, 1, tas_quantiles+1))
    pr_edges = np.quantile(da_pr.values[np.isfinite(da_pr)], np.linspace(0, 1, pr_quantiles+1))

    fig_res, axes_res = plt.subplots(
        tas_quantiles,
        pr_quantiles,
        figsize=(12, 12),
        sharex=True,
        sharey=True
    )
    axes_res[0, 0].set_xlim(da_res.min().item(),da_res.max().item())


    fig_abs, axes_abs = plt.subplots(
        tas_quantiles,
        pr_quantiles,
        figsize=(12, 12),
        sharex=True,
        sharey=True,

    )
    axes_abs[0, 0].set_xlim(da_abs.min().item(),da_abs.max().item())


    for tas_idx in range(tas_quantiles):  #Rückwärts, das in fig die Temperatur nach oben zu nimmt
        for pr_idx in range(pr_quantiles):

            mask = (
                (da_tas >= tas_edges[tas_idx]) &
                (da_tas < tas_edges[tas_idx+1]) &
                (da_pr >= pr_edges[pr_idx]) &
                (da_pr < pr_edges[pr_idx+1])
            )

            tas_values = da_tas.where(mask).values.flatten()
            pr_values = da_pr.where(mask).values.flatten()
            res_values = da_res.where(mask).values.flatten()
            est_values = da_est.where(mask).values.flatten()
            abs_values = da_abs.where(mask).values.flatten()

            tas_values = tas_values[np.isfinite(tas_values)]
            pr_values = pr_values[np.isfinite(pr_values)]
            res_values = res_values[np.isfinite(res_values)]
            est_values = est_values[np.isfinite(est_values)]
            abs_values = abs_values[np.isfinite(abs_values)]

            input_values = np.array([tas_values,pr_values]).T

            #Absolute hist
            axes_abs[tas_quantiles-tas_idx-1, pr_idx].hist(abs_values, bins=30, density = True)
            #distribution line
            x_min, x_max = axes_abs[tas_quantiles - tas_idx -1,pr_idx].get_xlim()
            x_pdf = np.linspace(x_min, x_max, 500)
            mixed_pdf = np.mean([norm.pdf(x_pdf,est_values[i], np.sqrt(var_regr.predict(input_values))[i]) for i in range(len(input_values))], axis = 0)   #lineare variance schätzung kann unter 0 gehen
            axes_abs[tas_quantiles - tas_idx-1,pr_idx].plot(x_pdf, mixed_pdf, 'r-', lw=2)
            

            axes_abs[tas_quantiles-tas_idx-1, pr_idx].set_title(
                f"tas:{tas_edges[tas_idx]:.4g} to {tas_edges[tas_idx+1]:.4g},\n pr:{pr_edges[pr_idx]:.4g} to {pr_edges[pr_idx + 1]:.4g}\n N={len(abs_values)}"
        )
            
            #residual hist
            axes_res[tas_quantiles-tas_idx-1,pr_idx].hist(res_values, bins=30, density = True)
            #distribution line
            x_min, x_max = axes_res[tas_quantiles - tas_idx -1,pr_idx].get_xlim()
            x_pdf = np.linspace(x_min, x_max, 500)
            mixed_pdf = np.mean([norm.pdf(x_pdf,0, np.sqrt(var_regr.predict(input_values))[i]) for i in range(len(input_values))], axis = 0)
            axes_res[tas_quantiles - tas_idx-1,pr_idx].plot(x_pdf, mixed_pdf, 'r-', lw=2)
            

            axes_res[tas_quantiles-tas_idx-1, pr_idx].set_title(
                f"tas:{tas_edges[tas_idx]:.4g} to {tas_edges[tas_idx+1]:.4g},\n pr:{pr_edges[pr_idx]:.4g} to {pr_edges[pr_idx + 1]:.4g}\n N={len(res_values)}"
        )

    fig_res.suptitle("residuals_histogram")
    fig_res.tight_layout()
    fig_res.show()

    fig_abs.suptitle("absolute_histogram")
    fig_abs.tight_layout()
    fig_abs.show()

    if output_var == "mrsol":
        folder = f"density/histogrammVSdistribution/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{output_var}/depth = {depth}"
    else:
       folder = f"density/histogrammVSdistribution/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{output_var}"
    if save:
        plot_one_var.save_plot(fig=fig_abs,var= output_var,scen = scen , folder = folder,  name_postfix = f"abs_histogramm_inkl_est_distribution_lat_idx = {lat_idx} lon_idx = {lon_idx}")
        plot_one_var.save_plot(fig=fig_res,var= output_var,scen = scen , folder=folder, name_postfix = f"res_histogramm_inkl_est_distribution_lat_idx = {lat_idx} lon_idx = {lon_idx}")
            
    
    

def all_histograms_vs_mixed_densities(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
                    lat_idx = 17, lon_idx = 33 , depth_arr = [0,1,2,3,4], r_arr = [0],
                    month_arr = [1,7], hist_arr = [1], start = None, end = None,
                    location = "",
                    save = False, all = False,
                    mu_min_run_idx = 1, mu_max_run_idx = 2,
                    var_min_run_idx = 2, var_max_run_idx = 11,
                    min_hist_run_idx = 11, max_hist_run_idx = 31 ):
    
    for scen in scenarios:
        for mon in month_arr:
            for r in r_arr:
                for hist in hist_arr:
                    for var in variables:
                        if var == "mrsol":
                            for depth in depth_arr:
                                predictor = distribution_building.distribution_predictor(scen = scen, input_vars = ["tas", "pr"], output_var = var,
                                        lat_idx = lat_idx, lon_idx = lon_idx , depth = depth, r = r,
                                        start =start, end = end,  Month_idx = mon, hist = hist, 
                                        mu_min_run_idx = mu_min_run_idx, mu_max_run_idx = mu_max_run_idx,
                                        var_min_run_idx = var_min_run_idx, var_max_run_idx = var_max_run_idx
                                        )
                                single_residual_split_histogram_vs_mixed_distribution(predictor, scen = scen, output_var = var,
                                        lat_idx = lat_idx, lon_idx = lon_idx , depth = depth, r = r,
                                        start =start, end = end, mon = mon, hist = hist,
                                        location = location,
                                        save = save, min_run_idx = min_hist_run_idx, max_run_idx = max_hist_run_idx,
                                        tas_quantiles = 4, pr_quantiles = 4)
                                plt.close("all")
                                if not all:
                                    return "done"
                        else:
                            predictor = distribution_building.distribution_predictor(scen = scen, input_vars = ["tas", "pr"], output_var = var,
                                        lat_idx = lat_idx, lon_idx = lon_idx, r = r,
                                        start =start, end = end,  Month_idx = mon, hist = hist, 
                                        mu_min_run_idx = mu_min_run_idx, mu_max_run_idx = mu_max_run_idx,
                                        var_min_run_idx = var_min_run_idx, var_max_run_idx = var_max_run_idx
                                        )
                            single_residual_split_histogram_vs_mixed_distribution(predictor, scen = scen, output_var = var,
                                        lat_idx = lat_idx, lon_idx = lon_idx, r = r,
                                        start =start, end = end, mon = mon, hist = hist,
                                        location = location,
                                        save = save, min_run_idx = min_hist_run_idx, max_run_idx = max_hist_run_idx,
                                        tas_quantiles = 4, pr_quantiles = 4)
                            plt.close("all") 
                        if not all:
                            return "done"
    
                        plt.close('all')



"""
def single_residual_split_histogram_vs_pred_distribution(dist_pred, scen = "historical", output_var = "mrsol",
                                                        lat_idx = 16, lon_idx = 32 , depth = 0, r = 0,
                                                        start = None, end = None, mon = 1, hist = 1,
                                                        location = "",
                                                        save = False, min_run_idx = 11, max_run_idx = 21,
                                                        tas_quantiles = 4, pr_quantiles = 4):

    hist_result = single_residual_split_histogram(lin_regr = dist_pred[0], scen = scen, var = output_var,
                    lat_idx = lat_idx, lon_idx = lon_idx , depth = depth, r = r,
                    start = start, end = end, mon = mon, hist = hist,
                    location = location,
                    save = False, min_run_idx = min_run_idx, max_run_idx = max_run_idx,
                    tas_quantiles=tas_quantiles, pr_quantiles=pr_quantiles)
    
    fig_abs = hist_result[0]
    fig_res = hist_result[1]
    axes_abs = hist_result[2]
    axes_res = hist_result[3]
    tas_edges = hist_result[4]
    pr_edges = hist_result[5]
    abs_values_mat = hist_result[6]
    res_values_mat = hist_result [7]
    
    
    print (abs_values_mat[0,0])
    
    for tas_idx in range(tas_quantiles):  #Rückwärts, das in fig die Temperatur nach oben zu nimmt
        for pr_idx in range(pr_quantiles):

            local_tas_mean = (tas_edges[tas_idx] + tas_edges[tas_idx + 1])/2
            local_pr_mean = (pr_edges[pr_idx] + pr_edges[pr_idx + 1])/2

            x_min, x_max = axes_abs[tas_quantiles - tas_idx -1,pr_idx].get_xlim()
            
            x_pdf = np.linspace(x_min, x_max, 500)
            norm_pdf = norm.pdf(x_pdf,dist_pred[0].predict([[local_tas_mean,local_pr_mean]])[0], np.sqrt(dist_pred[1].predict([[local_tas_mean,local_pr_mean]])[0]))
            
            axes_abs[tas_quantiles - tas_idx-1,pr_idx].plot(x_pdf, norm_pdf, 'r-', lw=2)
            fig_abs.show()


            x_min, x_max = axes_res[tas_quantiles - tas_idx -1,pr_idx].get_xlim()
            
            x_pdf = np.linspace(x_min, x_max, 500)
            norm_pdf = norm.pdf(x_pdf,0, np.sqrt(dist_pred[1].predict([[local_tas_mean,local_pr_mean]])[0]))
            
            axes_res[tas_quantiles - tas_idx-1,pr_idx].plot(x_pdf, norm_pdf, 'r-', lw=2)
            fig_res.show()


    return  (fig_abs, fig_res)

   """ 