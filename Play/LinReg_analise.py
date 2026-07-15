import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import math
import xarray as xr
import sklearn


from sklearn.linear_model import LinearRegression
from Play import shape_data as shape
from Play import plot_one_var
from Play import dt_functions 

#ToDo
def show_forest_regressor(regr, input, Tree = 0 ):
    print("prediction:", regr.predict([input]))
    print(f"Tree {Tree} prediction:", regr.estimators_[Tree].predict([input]))
    print("Tree 1 prediction:", regr.estimators_[1].predict([input]))
    print("Tree 2 prediction:", regr.estimators_[2].predict([input]))
    print(f"Tree {Tree} decision path:", regr.estimators_[Tree].decision_path([input]).toarray())
    print("Tree 1 decision path:", regr.estimators_[1].decision_path([input]).toarray())
    print("Tree 2 decision path:", regr.estimators_[2].decision_path([input]).toarray())
    print(f"Tree {Tree} feature:", regr.estimators_[Tree].tree_.feature)
    print(f"Tree {Tree} threshold:", regr.estimators_[Tree].tree_.threshold)
    print(f"Tree {Tree} children left:", regr.estimators_[Tree].tree_.children_left)
    print(f"Tree {Tree} children right:", regr.estimators_[Tree].tree_.children_right)


#ToDo
def show_coeffmatrix(regr,r = 0, hist = 1,
                            clip = True,  threshhold = 0.001, 
                            save = False,scen="historical", output_var = "mrsol", name = "", name_postfix = ""):
    

    max_coef = max(max(abs(regr.coef_)),0.01)        
    arr = regr.coef_.reshape(2,hist,-1,2*r+1) 

        
    ncols = min(3, hist)                  
    nrows = math.ceil(hist / ncols)       


    fig, axs = plt.subplots(2 * nrows, ncols ,squeeze=False,figsize = (5*ncols,4*2*nrows))  #grid
    for i in range(0,hist):
        
        row = i // ncols
        col = i % ncols

        im = axs[row,col].imshow(arr[0][i], origin= "lower", vmin = - max_coef, vmax = max_coef) #     für logscale siehe forest, hier weggelassen, da auch reevante negative werte  => logvariate   abs(arr[0][i] ....,norm = colors.LogNorm(vmin=threshhold,vmax=max_coef, clip=clip)
        axs[row,col].set_title(f"tas hist {i}")

        im = axs[nrows + row,col].imshow(arr[1][i], origin= "lower", vmin = - max_coef, vmax = max_coef)   
        axs[nrows + row,col].set_title(f"pr hist {i}")

    fig.colorbar(im, ax = axs)
    

    if save:
        plot_one_var.save_plot(fig = fig,var= output_var,scen= scen,name_prefix = output_var, name = f"LinReg{name}",name_postfix = f"/coeffs_{name_postfix}")
    
    fig.show()
    plt.close("all")
    print("IMP")  
          


#ToDo
def show_coefficient_regrmatrix(regr_mat, 
                            save = False, scen = "historical", output_var = "", name = "",name_postfix = "") :
    example = shape.first_not_none_element(regr_mat)
    coeffs = []
    for feature in range(0,example.feature_importances_.size) :
        coeffs_current = []
        for col in regr_mat :
            new_col = []
            for regr in col:
                if(regr) :
                    new_col.append(regr.coeff_[feature])
                else:
                    new_col.append(0)
            coeffs_current.append(new_col)
        coeffs.append(coeffs_current)
    data = np.array(coeffs)

    for feature in range(0,example.feature_importances_.size) :
        plt.imshow(data[feature], origin= "lower")
        plt.colorbar()
        if save:
            plot_one_var.save_plot(fig=plt,var= output_var,scen= scen ,name_prefix = output_var, name = f"LinReg{name}",name_postfix = f"/ceoff{name_postfix}")
        plt.show()
        



def plot_performance_over_time (regr, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol",
                            lat_idx = 30, lon_idx = 0, depth = 0, r = 0,
                            start ="1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None, hist = 1,
                            run_idx = 1, 
                            feature_plot = True,
                            save = False, name = ""):

    #real outcome
    output = shape.show_data_set(var = output_var, scen = scen, run_idx = run_idx)
    output = shape.prune_group_ds_timespan(output,start= start,end= end,time_step = time_step, Month_idx = Month_idx)
    output = output.isel(lat = lat_idx, lon = lon_idx)

    output = output.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")
    if output_var == "mrsol":
        output= output.isel(depth = depth)

    #input
    start_hist = shape.start_with_hist(start=start,time_step= time_step,hist= hist)
    input_dt = shape.load_create_datatree(scenarios = [scen],variables = input_vars, start = start_hist,end = end,time_step= time_step, run_idx = run_idx)
    features = shape.dt_to_features(dt=input_dt,lat_idx= lat_idx,lon_idx= lon_idx,r= r,Month_idx= Month_idx,hist= hist,start= start,end= end,time_step= time_step)
    
    output = output.sel(time=features.time)

    output_estimated = xr.zeros_like(output)
                
    output_estimated[output_var][:] = regr.predict(features.values)

    
    square_error = (output_estimated - output) ** 2
    mse = square_error.mean("time")[f"{output_var}"].item()

    output_estimated = output_estimated.rename({output_var: f"{output_var}_est"})

    xr.merge([output,output_estimated]).to_array().plot.line(hue="variable", x= "time")


    if save:
        plot_one_var.save_plot(fig=plt,var= output_var,scen= scen, name_prefix = output_var, name = f"LinReg{name}",name_postfix = f"/over_time lat_idx {lat_idx} lon_idx {lon_idx} mse {mse:.3f} overtime")
    plt.show()
    
    
    print(f" mse {mse:.3f}")
    if feature_plot:
        show_coeffmatrix(regr=regr, r=r, hist=hist)
        

    plt.close("all")
    print("OT")

def local_mse_over_time (regr, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol",
                            lat_idx = 30, lon_idx = 0, depth = 0, r = 0,
                            start ="1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None, hist = 1,
                            run_idx = 1 ):

    #real outcome
    output = shape.show_data_set(var = output_var, scen = scen, run_idx = run_idx)
    output = shape.prune_group_ds_timespan(output,start= start,end= end,time_step = time_step, Month_idx = Month_idx)
    output = output.isel(lat = lat_idx, lon = lon_idx)

    output = output.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")
    if output_var == "mrsol":
        output= output.isel(depth = depth)

    #input
    start_hist = shape.start_with_hist(start=start,time_step= time_step,hist= hist)
    input_dt = shape.load_create_datatree(scenarios = [scen],variables = input_vars, start = start_hist,end = end,time_step= time_step, run_idx=run_idx)
    features = shape.dt_to_features(dt=input_dt,lat_idx= lat_idx,lon_idx= lon_idx,r= r,Month_idx= Month_idx,hist= hist,start= start,end= end,time_step= time_step)
    
    output = output.sel(time=features.time)

    output_estimated = xr.zeros_like(output)
                
    output_estimated[output_var][:] = regr.predict(features.values)

    
    square_error = (output_estimated - output) ** 2
    return square_error.mean("time")[f"{output_var}"].item()
    
def global_mse_over_time (regr_mat, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol",
                            min_lat_idx = 30 , max_lat_idx = 31, min_lon_idx = 0, max_lon_idx = 1, depth = 0, r = 0,
                            start ="1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None, hist = 1,
                            run_idx = 1 ):
    
    mse_mat = [[None for _ in range(min_lon_idx, max_lon_idx)] for _ in range(min_lat_idx, max_lat_idx)]

    if len(regr_mat) != (max_lat_idx - min_lat_idx) or len(regr_mat[0]) != (max_lon_idx - min_lon_idx):
        raise ValueError("Dimensions of regr_mat do not match the specified latitude and longitude ranges.")

    
    for res_lat in range(max_lat_idx - min_lat_idx):
        for res_col in range(max_lon_idx - min_lon_idx):
            try:
                mse_mat[res_lat][res_col] = local_mse_over_time (regr_mat[res_lat][res_col],scen=scen,input_vars=input_vars,output_var=output_var, 
                lat_idx=min_lat_idx+res_lat,lon_idx=min_lon_idx+res_col,depth=depth,r=r,
                start=start,end=end,time_step= time_step, Month_idx = Month_idx,hist = hist,
                run_idx = run_idx)
            except:
                mse_mat[res_lat][res_col] = None
                
    return mse_mat

def local_score_over_time (regr, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol",
                            lat_idx = 30, lon_idx = 0, depth = 0, r = 0,
                            start ="1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None, hist = 1,
                            run_idx = 1 ):

    #real outcome
    output = shape.show_data_set(var = output_var, scen = scen, run_idx = run_idx)
    output = shape.prune_group_ds_timespan(output,start= start,end= end,time_step = time_step, Month_idx = Month_idx)
    output = output.isel(lat = lat_idx, lon = lon_idx)

    output = output.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")
    if output_var == "mrsol":
        output= output.isel(depth = depth)

    #input
    start_hist = shape.start_with_hist(start=start,time_step= time_step,hist= hist)
    input_dt = shape.load_create_datatree(scenarios = [scen],variables = input_vars, start = start_hist,end = end,time_step= time_step, run_idx = run_idx)
    features = shape.dt_to_features(dt=input_dt,lat_idx= lat_idx,lon_idx= lon_idx,r= r,Month_idx= Month_idx,hist= hist,start= start,end= end,time_step= time_step)
    
    output = output.sel(time=features.time)

    output_estimated = xr.zeros_like(output)
                
    output_estimated[output_var][:] = regr.predict(features.values)
    score = sklearn.metrics.explained_variance_score(output.to_array().T, output_estimated.to_array().T)

    return score

def global_score_over_time (regr_mat, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol",
                            min_lat_idx = 30 , max_lat_idx = 31, min_lon_idx = 0, max_lon_idx = 1, depth = 0, r = 0,
                            start ="1850-01-01", end = "1900-01-01", time_step = "1ME", Month_idx = None, hist = 1,
                            run_idx = 1, show_error = False ):
    
    score_mat = [[None for _ in range(min_lon_idx, max_lon_idx)] for _ in range(min_lat_idx, max_lat_idx)]

    if len(regr_mat) != (max_lat_idx - min_lat_idx) or len(regr_mat[0]) != (max_lon_idx - min_lon_idx):
        raise ValueError("Dimensions of regr_mat do not match the specified latitude and longitude ranges.")

    
    for res_lat in range(max_lat_idx - min_lat_idx):
        for res_col in range(max_lon_idx - min_lon_idx):
            try:
                score_mat[res_lat][res_col] = local_score_over_time (regr_mat[res_lat][res_col],scen=scen,input_vars=input_vars,output_var=output_var, 
                lat_idx=min_lat_idx+res_lat,lon_idx=min_lon_idx+res_col,depth=depth,r=r,
                start=start,end=end,time_step= time_step, Month_idx = Month_idx,hist = hist,
                run_idx = run_idx)
            except:
                if show_error:
                    score_mat[res_lat][res_col] = local_score_over_time (regr_mat[res_lat][res_col],scen=scen,input_vars=input_vars,output_var=output_var, 
                    lat_idx=min_lat_idx+res_lat,lon_idx=min_lon_idx+res_col,depth=depth,r=r,
                    start=start,end=end,time_step= time_step, Month_idx = Month_idx,hist = hist,
                    run_idx = run_idx)
                score_mat[res_lat][res_col] = None
                
    return score_mat

def plot_performance_against_mean(regr_mat, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol",
                            min_lat_idx = 0, max_lat_idx=40,min_lon_idx = 0, max_lon_idx = 40, depth = 0, r = 0,
                            start ="1895-01-01", end = "1897-01-01", time_step = "1ME", Month_idx = None, hist = 1,
                            max_plots = 12, 
                            run_idx = 1,  
                            absolute = True,
                            save = False, name = "" ):
    
    #real outcome
    output = shape.show_data_set(var = output_var, scen = scen, run_idx = run_idx)
    output = shape.prune_group_ds_timespan(ds=output,start= start,end= end,time_step = time_step, Month_idx = Month_idx)
    output = output.isel(lat = slice(min_lat_idx,max_lat_idx), lon = slice(min_lon_idx,max_lon_idx))

    output = output.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")
    if output_var == "mrsol":
        output= output.isel(depth = depth)
    
    
    #input
    start_hist = shape.start_with_hist(start=start,time_step= time_step,hist= hist)
    input_dt = shape.load_create_datatree(scenarios = [scen],variables = input_vars, start = start_hist,end = end,time_step= time_step, run_idx = run_idx)
    
    features = shape.dt_to_features(dt=input_dt,lat_idx = min_lat_idx,lon_idx = min_lon_idx,r= r,Month_idx= Month_idx,hist= hist,start= start,end= end,time_step= time_step)
                
    output = output.sel(time=features.time)

    output_mean = output.mean("time")

    #Plot begrenzung
    n_plots = min(max_plots, output[output_var].sizes["time"])
    # gleichmäßig verteilte Indizes
    idx = np.linspace(0, output[output_var].sizes["time"] - 1, n_plots, dtype=int)

    

    #estimated output
    output_estimated = xr.zeros_like(output)
    for dlat_idx in range(min_lat_idx,max_lat_idx):
        for dlon_idx in range(min_lon_idx,max_lon_idx):
            if regr_mat[dlat_idx-min_lat_idx][dlon_idx-min_lon_idx] is not None:
                features = shape.dt_to_features(dt=input_dt,lat_idx= dlat_idx,lon_idx= dlon_idx,r= r,Month_idx= Month_idx,hist= hist,start= start,end= end,time_step= time_step)
                output_estimated[output_var][:,dlat_idx-min_lat_idx,dlon_idx-min_lon_idx] = regr_mat[dlat_idx-min_lat_idx][dlon_idx-min_lon_idx].predict(features.values)
    
    output = output.isel(time=idx)

    output_diff = output - output_mean

    output_estimated_diff = output - output_estimated

    output_estimated = output_estimated.isel(time=idx)





    #ablsolute Plots
    if absolute : 
        output_estimated[output_var].plot(col="time", col_wrap = 4, x = "lon");
        if save:
            plot_one_var.save_plot(fig=plt,var= output_var,scen= scen, name_prefix = output_var, name = f"LinReg{name}",name_postfix = "/snapshots estimated_absolute")
        plt.suptitle(f"{name} estimated_absolute" , fontsize = 16,x = 0.43,  y = 1.03)
        plt.figure().tight_layout()
        plt.show()
        print("AM")

        output[output_var].plot(col="time", col_wrap = 4, x = "lon"); 
        if save:
            plot_one_var.save_plot(fig=plt,var= output_var,scen= scen, name_prefix = output_var, name = f"LinReg{name}",name_postfix = "/snapshots real_absolute")
        plt.suptitle(f"{name} real_absolute" , fontsize = 16,x = 0.43,  y = 1.03)
        plt.figure().tight_layout()
        plt.show()
        print("AM")

    #relative Plots
    output_estimated_diff[output_var].plot(col="time", col_wrap = 4, x = "lon")
    if save:
        plot_one_var.save_plot(fig=plt,var= output_var,scen= scen, name_prefix = output_var, name = f"LinReg{name}",name_postfix = "/snapshots estimated_diff")
    plt.suptitle(f"{name} estimated_diff" , fontsize = 16,x = 0.43,  y = 1.03)
    plt.figure().tight_layout()
    plt.show()
    print("AM")

    output_diff[output_var].plot(col="time", col_wrap = 4, x = "lon"); 
    if save:
        plot_one_var.save_plot(fig=plt,var= output_var,scen= scen,name_prefix = output_var, name = f"LinReg{name}",name_postfix = "/snapshots mean_diff")
    plt.suptitle(f"{name} diff_to_mean" , fontsize = 16,x = 0.43,  y = 1.03)
    plt.figure().tight_layout()
    plt.show()
    plt.close("all")
    print("AM")






'''

def plot_performance_against_mean(regr_mat, scen = "historical", input_vars = ["tas","pr"], output_var = "mrsol", depth = 0, start ="1895-01-01", end = "1897-01-01", time_step = "1ME", absolute = True, name = "", Month_idx = None, min_lat_idx = 0, max_lat_idx=40,min_lon_idx = 0, max_lon_idx = 40, test = False, max_plots = 12, save = False, r = 0, hist = 0):
    
    #real outcome
    output = shape.show_data_set(var = output_var, scen = scen, test = test)
    output = shape.prune_group_ds_timespan(output, start, end,time_step = time_step, Month_idx = Month_idx)
    output = output.isel(lat = slice(min_lat_idx,max_lat_idx), lon = slice(min_lon_idx,max_lon_idx))

    output = output.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")
    if output_var == "mrsol":
        output= output.isel(depth = depth)
    
    output_mean = output.mean("time")

    #Plot begrenzung
    n_plots = min(max_plots, output[output_var].sizes["time"])
    # gleichmäßig verteilte Indizes
    idx = np.linspace(0, output[output_var].sizes["time"] - 1, n_plots, dtype=int)

    output = output.isel(time=idx)

    output_diff = output - output_mean

    #input
    input_datasets = []
    for var in input_vars:
        ds = shape.show_data_set(var = var, scen = scen, test = test)
        ds = shape.prune_group_ds_timespan(ds, start, end, time_step=time_step, Month_idx = Month_idx)
        ds = ds.drop_vars(["height", "time_bnds", "file_qf","depth_bnds"], errors="ignore")
        ds = ds.isel(lat = slice(min_lat_idx-r,max_lat_idx+r), lon = slice(min_lon_idx-r,max_lon_idx+r))
        ds = ds.isel(time=idx)
        input_datasets.append(ds)
    input  = xr.merge(input_datasets)


    #estimated output
    output_estimated = xr.zeros_like(output)
    for dlat_idx in range(0,output_estimated.sizes["lat"]):
        for dlon_idx in range(0,output_estimated.sizes["lon"]):
            if regr_mat[dlat_idx][dlon_idx] is not None:
                features = input.isel(lat=slice(dlat_idx,dlat_idx+2*r+1),lon=slice(dlon_idx,dlon_idx+2*r+1))
                features = features.to_array().transpose("time","variable","lat","lon").stack(features=("variable","lat","lon"))
                output_estimated[output_var][:,dlat_idx,dlon_idx] = regr_mat[dlat_idx][dlon_idx].predict(features.values)
    output_estimated_diff = output - output_estimated





    #ablsolute Plots
    if absolute : 
        output_estimated[output_var].plot(col="time", col_wrap = 4, x = "lon");
        if save:
            plot_one_var.save_plot(plt, output_var, scen, name_prefix = output_var, name = f"forest{name}",name_postfix = "estimated_absolute")
        plt.suptitle(f"{name} estimated_absolute" , fontsize = 16,x = 0.43,  y = 1.03)
        plt.figure().tight_layout()
        #plt.show()

        output[output_var].plot(col="time", col_wrap = 4, x = "lon"); 
        if save:
            plot_one_var.save_plot(plt, output_var, scen, name_prefix = output_var, name = f"forest{name}",name_postfix = "real_absolute")
        plt.suptitle(f"{name} real_absolute" , fontsize = 16,x = 0.43,  y = 1.03)
        plt.figure().tight_layout()
        #plt.show()

    #relative Plots
    output_estimated_diff[output_var].plot(col="time", col_wrap = 4, x = "lon")
    if save:
        plot_one_var.save_plot(plt, output_var, scen, name_prefix = output_var, name = f"forest{name}",name_postfix = "estimated_diff")
    plt.suptitle(f"{name} estimated_diff" , fontsize = 16,x = 0.43,  y = 1.03)
    plt.figure().tight_layout()
    #plt.show()

    output_diff[output_var].plot(col="time", col_wrap = 4, x = "lon"); 
    if save:
        plot_one_var.save_plot(plt, output_var, scen ,name_prefix = output_var, name = f"forest{name}",name_postfix = "mean_diff")
    plt.suptitle(f"{name} diff_to_mean" , fontsize = 16,x = 0.43,  y = 1.03)
    plt.figure().tight_layout()
    #plt.show()


'''
