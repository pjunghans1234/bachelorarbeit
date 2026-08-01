import calendar
import numpy as np
import matplotlib.pyplot as plt

from Play import forest_building, plot_one_var 
from Play import forest_analise 

from Play import LinReg_building 
from Play import LinReg_analise 

from Play import plot_one_var

#Hilfsfunktion für die Plotreihe comparison, die scores und mse von linreg und forest vergleicht. (diese Funktion macht die mse Plots)
def do_mse_comparison(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
        min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0, max_lon_idx = 40, depth = [0,1,2,3,4],r_arr = [0,3],
        months = [1,7], hist = [1,3],
        location = "",
        max_depth = 5, randomstate = 0,
        save = False, all = False
    ):
    

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

                                #f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}


                                if save:
                                    plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"comparison/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}",name_postfix = f"mse_depth = {d}")
                                

                                if not all:
                                    return fig
                                


                        else:
                            forest_regr_mat = forest_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            lin_regr_mat = LinReg_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            forest_mse_mat =forest_analise.global_mse_over_time (forest_regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                            
                            lin_mse_mat =LinReg_analise.global_mse_over_time (lin_regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                            

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

                            
                            if save:
                                plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"comparison/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}",name_postfix = f"mse")
                            

                            if not all:
                                    return fig
                            
#Hilfsfunktion für die Plotreihe comparison, die scores und mse von linreg und forest vergleicht. (diese Funktion macht die Explained variance score Plots)
def do_score_comparison(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
        min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0, max_lon_idx = 40, depth = [0,1,2,3,4],r_arr = [0,3],
        months = [1,7], hist = [1,3],
        location = "",
        max_depth = 5, randomstate = 0,
        save = False, all = False
    ):
    

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
                                forest_score_mat =forest_analise.global_score_over_time (forest_regr_mat,scen=scen,output_var="mrsol", min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                                lin_score_mat =LinReg_analise.global_score_over_time (lin_regr_mat,scen=scen,output_var="mrsol", min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                               
                                
                                forest_score_mat =  np.array(forest_score_mat, dtype=float)
                                lin_score_mat =  np.array(lin_score_mat, dtype=float)
                                forest_score_mat[forest_score_mat == None] = np.nan
                                lin_score_mat[lin_score_mat == None] = np.nan
                                
                                
                                fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True)  #grid
                                
                                im = axs[0,0].imshow(lin_score_mat, origin= "lower", cmap="seismic", vmin = - 1, vmax = 1)       #für logscale siehe forest_analaysis, 
                                axs[0,0].set_title(f"linreg score")
                               
                                im = axs[1,0].imshow(forest_score_mat, origin= "lower",  cmap="seismic", vmin = - 1, vmax = 1)       #für logscale siehe forest_analaysis, 
                                axs[1,0].set_title(f"forest score")

                                im = axs[2,0].imshow(forest_score_mat-lin_score_mat , origin= "lower",  cmap="seismic",  vmin = - 1, vmax = 1)       #für logscale siehe forest_analaysis, 
                                axs[2,0].set_title(f"score forest - lin")

                                fig.colorbar(im, ax = axs)

                                

                                if save:
                                    plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"comparison/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}",name_postfix = f"score_depth = {d}")
                                

                                if not all:
                                    return fig


                        else:
                            forest_regr_mat = forest_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            lin_regr_mat = LinReg_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            forest_score_mat =forest_analise.global_score_over_time (forest_regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                            
                            lin_score_mat =LinReg_analise.global_score_over_time (lin_regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True)
                            

                            forest_score_mat =  np.array(forest_score_mat, dtype=float)
                            lin_score_mat =  np.array(lin_score_mat, dtype=float)
                            forest_score_mat[forest_score_mat == None] = np.nan
                            lin_score_mat[lin_score_mat == None] = np.nan
                            
                            
                            
                            
                            fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True)  #grid
                            
                            im = axs[0,0].imshow(lin_score_mat, origin= "lower", cmap="seismic", vmin = - 1, vmax = 1)       #für logscale siehe forest_analaysis, 
                            axs[0,0].set_title(f"linreg score")
                            
                            im = axs[1,0].imshow(forest_score_mat, origin= "lower", cmap="seismic",  vmin = - 1, vmax = 1)       #für logscale siehe forest_analaysis, 
                            axs[1,0].set_title(f"forest score")

                            im = axs[2,0].imshow(forest_score_mat-lin_score_mat , origin= "lower", cmap="seismic", vmin = - 1, vmax = 1)       #für logscale siehe forest_analaysis, 
                            axs[2,0].set_title(f"score forest - lin")

                            fig.colorbar(im, ax = axs)

                            
                            if save:
                                plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"comparison/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}",name_postfix = f"score")
                            

                            if not all:
                                    return fig
