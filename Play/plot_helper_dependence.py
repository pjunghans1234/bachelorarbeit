
import calendar

from Play import forest_building 
from Play import forest_analise 

from Play import LinReg_building 
from Play import LinReg_analise 

#Hilfsfunktion für die Plotreihe dependence, mit dieser Funktion werden für die forests je 3 pro konstelation 3 Plots gemacht: performance over time (lokale Zeitreihe Schäzung vs Effektiv wert), snapshots (regionale zeitsnapshots, die wiederum echt, geschäzt differenz und differenz zum Mittel zeigen), features, zeigt welche Werte  welchen einfluss Haben.
def do_forest(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
        min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0, max_lon_idx = 40,lat_idx = 19,lon_idx = 33, depth = [0,1,2,3,4],r_arr = [0,3],
        months = [1,7], hist = [1,9],
        location = "",
        max_depth = 5, randomstate = 0
    ):
    name = f"/dependence/{location}"

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
                                regr_mat = forest_building.global_vars_to_one_dim(scen = scen, output_var = "mrsol", min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                                regr = forest_building.local_vars_to_one_dim(scen = scen, output_var = "mrsol", lat_idx=lat_idx, lon_idx=lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                                forest_analise.plot_performance_against_mean(regr_mat,scen=scen,output_var="mrsol", min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}/depth = {d}")

                                forest_analise.plot_performance_over_time (regr,scen=scen,output_var="mrsol", lat_idx=lat_idx,lon_idx=lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,feature_plot=False, save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}/depth = {d}")

                                forest_analise.show_feature_importance(regr,scen=scen,output_var="mrsol",r=r,hist = h,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}/depth = {d}")
                        else:
                            regr_mat = forest_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            regr = forest_building.local_vars_to_one_dim(scen = scen, output_var = var, lat_idx=lat_idx, lon_idx=lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            forest_analise.plot_performance_against_mean(regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=depth,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}")

                            forest_analise.plot_performance_over_time (regr,scen=scen,output_var=var, lat_idx=lat_idx,lon_idx=lon_idx,depth=depth,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,feature_plot=False, save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}")

                            forest_analise.show_feature_importance(regr,scen=scen,output_var=var,r=r,hist = h,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}")

#Hilfsfunktion für die Plotreihe dependence, mit dieser Funktion werden für die lineare regression je 3 pro konstelation 3 Plots gemacht: performance over time (lokale Zeitreihe Schäzung vs Effektiv wert), snapshots (regionale zeitsnapshots, die wiederum echt, geschäzt differenz und differenz zum Mittel zeigen), coeffs, zeigt welche Werte  welchen einfluss Haben.
def do_LinReg(scenarios = ["historical", "ssp585"], variables = ["mrsol", "rsds","sfcWind","hurs"],
        min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0, max_lon_idx = 40,lat_idx = 19,lon_idx = 33, depth = [0,1,2,3,4],r_arr = [0,3],
        months = [1,7], hist = [1,9],
        location = "",
        max_depth = 5, randomstate = 0
    ):
    name = f"/dependence/{location}"

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
                                regr_mat = LinReg_building.global_vars_to_one_dim(scen = scen, output_var = "mrsol", min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                                regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = "mrsol", lat_idx=lat_idx, lon_idx=lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                                LinReg_analise.plot_performance_against_mean(regr_mat,scen=scen,output_var="mrsol", min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}/depth = {d}")

                                LinReg_analise.plot_performance_over_time (regr,scen=scen,output_var="mrsol", lat_idx=lat_idx,lon_idx=lon_idx,depth=d,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,feature_plot=False, save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}/depth = {d}")

                                LinReg_analise.show_coeffmatrix(regr,scen=scen,output_var="mrsol",r=r,hist = h,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}/depth = {d}")
                        else:
                            regr_mat = LinReg_building.global_vars_to_one_dim(scen = scen, output_var = var, min_lat_idx = min_lat_idx, max_lat_idx = max_lat_idx, min_lon_idx = min_lon_idx, max_lon_idx = max_lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            regr = LinReg_building.local_vars_to_one_dim(scen = scen, output_var = var, lat_idx=lat_idx, lon_idx=lon_idx, depth = d,r = r, start = start, end = end,  Month_idx = mon,  hist = h, max_depth = max_depth, randomstate = randomstate)

                            LinReg_analise.plot_performance_against_mean(regr_mat,scen=scen,output_var=var, min_lat_idx=min_lat_idx,max_lat_idx=max_lat_idx,min_lon_idx=min_lon_idx,max_lon_idx=max_lon_idx,depth=depth,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}")

                            LinReg_analise.plot_performance_over_time (regr,scen=scen,output_var=var, lat_idx=lat_idx,lon_idx=lon_idx,depth=depth,r=r,start=start,end=end,Month_idx = mon,hist = h, test=True,feature_plot=False, save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}")

                            LinReg_analise.show_coeffmatrix(regr,scen=scen,output_var=var,r=r,hist = h,save= True, name=f"{name}/{scen}/{calendar.month_name[mon]}/r = {r}/hist = {h}/{var}")


    