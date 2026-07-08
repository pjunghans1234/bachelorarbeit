import calendar

import numpy as np
from numpy import save
import matplotlib.pyplot as plt


from Play import forest_building, plot_one_var 
from Play import forest_analise 

from Play import LinReg_building 
from Play import LinReg_analise 

def do_comparison(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
        min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0, max_lon_idx = 40, depth = [0,1,2,3,4],r_arr = [0,3],
        months = [1,7], hist = [1,9],
        location = "",
        max_depth = 5, randomstate = 0,
        save = False
    ):
    name = f"/comparison/{location}"

    for scen in scenarios:
        if scen == "historical":
            start = "1850-01-01"
            end = "1900-01-01"
        else : 
            start = "2000-01-01"
            end = "2100-01-01"
        for mon in months:
            for r in r_arr:
                for h in hist:
                    for var in variables:
                        if var == "mrsol":
                            for d in depth:
                                
                                forest_regr_mat = forest_building.global_vars_to_one_dim(scen = scen, output_var = "mrsol", min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)
                                lin_regr_mat = LinReg_building.global_vars_to_one_dim(scen = scen, output_var = "mrsol", min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)
                                forest_mse_mat =forest_analise.global_mse_over_time (forest_regr_mat,scen=scen,output_var="mrsol", min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                                lin_mse_mat =LinReg_analise.global_mse_over_time (lin_regr_mat,scen=scen,output_var="mrsol", min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                                
                                
                                forest_mse_mat =  np.array(forest_mse_mat, dtype=float)
                                lin_mse_mat =  np.array(lin_mse_mat, dtype=float)
                                forest_mse_mat[forest_mse_mat == None] = np.nan
                                lin_mse_mat[lin_mse_mat == None] = np.nan
                                
                                
                                max_mse = max(np.nanmax(forest_mse_mat), np.nanmax(lin_mse_mat))
                                
                                
                                fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True)  #grid
                                
                                im = axs[0,0].imshow(lin_mse_mat, origin= "lower",cmap="seismic", vmin = - max_mse, vmax = max_mse)       #für logscale siehe forest_analaysis, 
                                axs[0,0].set_title(f"linreg mse")
                               
                                im = axs[1,0].imshow(forest_mse_mat, origin= "lower", cmap="seismic", vmin = - max_mse, vmax = max_mse)       #für logscale siehe forest_analaysis, 
                                axs[1,0].set_title(f"forest mse")

                                im = axs[2,0].imshow(lin_mse_mat - forest_mse_mat, origin= "lower", cmap="seismic", vmin = - max_mse, vmax = max_mse)       #für logscale siehe forest_analaysis, 
                                axs[2,0].set_title(f"difference")

                                fig.colorbar(im, ax = axs)

                                

                                #if save:
                                 #   plot_one_var.save_plot(fig=fig,var= output_var,scen= scen ,name_prefix = output_var,  name = f"forest{name}",name_postfix = f"/features{name_postfix}")
                                return "done"
                                


                            else:
                                forest_regr_mat = forest_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                                lin_regr_mat = LinReg_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                                forest_mse_mat =forest_analise.global_mse_over_time (forest_regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                                
                                lin_mse_mat =LinReg_analise.global_mse_over_time (lin_regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                                    
