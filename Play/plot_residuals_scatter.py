import matplotlib.pyplot as plt
import xarray as xr
import calendar
from Play import shape_data
from Play import LinReg_building
from Play import plot_one_var




#Hilfsfunktion für die Plotreihe dependence, mit dieser Funktion wird für linreg je ein scatter Plot ergänzt: dieser scattert jeweils zwei aus tas, pr, residual. der dritte Wert wird mit Farbe dargestellt. (residual ist der unterschied zwischen echt und schäzung der zielvariable) Mit dem Ziel verbliebene abhängigkeiten fest zu stellen
def residuals_scatter(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
                    lat_idx = 16, lon_idx = 32 , depth_arr = [0,1,2,3,4], r_arr = [0],
                    months = [1,7], hist_arr = [1],
                    location = "", run_idx = 1,
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
                        if var == "mrsol":
                            for depth in depth_arr:        

                                lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, run_idx = run_idx)
                                residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx,depth = depth, r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx = run_idx + 1, max_run_idx = run_idx + 2)


                                fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True, figsize = (5,4*3))  #grid
                                                            
                                xr.plot.scatter(residuals, x="tas", y = "pr", hue = "mrsol_res", ax = axs[0,0], cmap = "viridis")      

                                xr.plot.scatter(residuals, x="tas", y = "mrsol_res", hue = "pr", ax = axs[1,0])    

                                xr.plot.scatter(residuals, x="pr", y = "mrsol_res", hue = "tas", ax = axs[2,0])

                                if save:
                                    plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}/depth = {depth}", name_postfix = f"/scatter lat_idx = {lat_idx} lon_idx = {lon_idx}")
                                
                                if not all:
                                    return fig
                                
                        else :
                            lin_regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx, r = r,start = start, end = end, Month_idx = mon, hist = hist, run_idx = run_idx)
                            residuals = shape_data.create_residuals(lin_regr,scen = scen, output_var = var,lat_idx = lat_idx,lon_idx = lon_idx, r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx = run_idx + 1, max_run_idx = run_idx + 2)


                            fig, axs = plt.subplots(3, 1 ,squeeze=False, constrained_layout=True, figsize = (5,4*3))  #grid
                                                        
                            xr.plot.scatter(residuals, x="tas", y = "pr", hue = f"{var}_res", ax = axs[0,0],  cmap = "viridis")      

                            xr.plot.scatter(residuals, x="tas", y = f"{var}_res", hue = "pr", ax = axs[1,0])    

                            xr.plot.scatter(residuals, x="pr", y = f"{var}_res", hue = "tas", ax = axs[2,0])

                            if save:
                                plot_one_var.save_plot(fig=fig,var= var,scen = scen ,name_prefix = var,  name = f"LinReg/dependence/{location}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {hist}/{var}", name_postfix = f"/scatter lat_idx = {lat_idx} lon_idx = {lon_idx}")
                            
                            if not all:
                                return fig

    