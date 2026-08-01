
import numpy as np
import xarray as xr

import xskillscore as xs

#noch ungebraucht
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
from Play import shape_data


#Berechnet crps score der Distributions von dist_pred auf den runs min_run_idx bis max_run_idx. Für jeden zu schätzenden Datenpunkt, berechnet sie den score zwischen gezogenen Samples aus der predicteten Verteilung und dem echten Datenpunkt. Siehe https://xskillscore.readthedocs.io/en/stable/api/xskillscore.crps_ensemble.html 
def crps_ensemble_score(dist_pred, scen = "historical", output_var = "mrsol",
                        lat_idx = 16, lon_idx = 32 , depth = 0, r = 0,
                        start = None, end = None, mon = 1, hist = 1,
                        min_run_idx = 11, max_run_idx = 21,
                        Transform = None):

    if scen == "historical" and start == None :
        start = "1850-01-01"
    if scen == "historical" and end == None : 
        end = "1900-01-01"
    
    else :
        if start == None :
            start = "2000-01-01"
        if end == None :
            end = "2100-01-01"

    data = shape_data.create_residuals(dist_pred[0],scen = scen,output_var = output_var,lat_idx= lat_idx,lon_idx = lon_idx,depth=depth,  r = r,start = start, end = end, Month_idx = mon, hist = hist, min_run_idx=min_run_idx, max_run_idx=max_run_idx,Transform=Transform)
    res = data.mrsol_res.values.flatten()

    tas = data.tas.values.flatten()
    tas = tas[np.isfinite(res)]
    pr = data.pr.values.flatten()
    pr = pr[np.isfinite(res)]
    mask = np.isfinite(data.mrsol_res.values)
    data["sigma"] = xr.full_like(data["mrsol_res"], np.nan)
    data["sigma"].values[mask] = np.sqrt(np.maximum(dist_pred[1].predict(np.array([tas,pr]).T),1e-6))



    dist_samples = np.random.normal(
        loc=data.mrsol_est.values[..., None],
        scale=data.sigma.values[..., None],
        size=data.mrsol_est.shape + (1000,)
    )


    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=data.mrsol.dims + ("member",),
        coords={
            **data.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    

    if Transform == "Logit":
        space_score = xs.crps_ensemble(data.mrsol,dist_samples_da)
        
        data_bt = xr.full_like(data, np.nan)
        dist_samples_da_bt = xr.full_like(dist_samples_da, np.nan)
        data_bt["mrsol"] = 100 / (1 + np.exp(data.mrsol))
        dist_samples_da_bt = 100 / (1 + np.exp(dist_samples_da))

        score = xs.crps_ensemble(data_bt.mrsol,dist_samples_da_bt)

        return (score,space_score)

    if Transform == "Log":
            space_score = xs.crps_ensemble(data.mrsol,dist_samples_da)
            
            data_bt = xr.full_like(data, np.nan)
            dist_samples_da_bt = xr.full_like(dist_samples_da, np.nan)
            data_bt["mrsol"] = np.exp(data.mrsol)
            dist_samples_da_bt = np.exp(dist_samples_da)
    
            score = xs.crps_ensemble(data_bt.mrsol,dist_samples_da_bt)
    
            return (score,space_score)
    

    else :
        score = xs.crps_ensemble(data.mrsol,dist_samples_da)
        return (score, score)

#Wendet crps_ensemble_score (siehe oben) auf alle gewünschten ortskoordinaten an
def crps_ensemble_score_mat (dist_pred_mat, scen = "historical", output_var = "mrsol",
                        min_lat_idx = 0, max_lat_idx = 40, min_lon_idx = 0 , max_lon_idx = 40 , depth = 0, r = 0,
                        start = None, end = None, mon = 1, hist = 1,
                        min_run_idx = 11, max_run_idx = 21,
                        Transform = None): 
    score_mat = np.full((40,40),np.nan)
    space_score_mat = np.full((40,40),np.nan)
        
    for lat_idx in range(min_lat_idx,max_lat_idx):
        for lon_idx in range(min_lon_idx, max_lon_idx):
            if isinstance(dist_pred_mat[lat_idx,lon_idx], tuple):
                print(f"lat = {lat_idx}, lon = {lon_idx}")
                print(dist_pred_mat[lat_idx,lon_idx]) 
                score = crps_ensemble_score(dist_pred_mat[lat_idx,lon_idx], scen = scen, output_var = output_var,
                        lat_idx = lat_idx, lon_idx = lon_idx , depth = depth, r = r,
                        start = start, end = end, mon = mon, hist = hist,
                        min_run_idx = min_run_idx, max_run_idx = max_run_idx,
                        Transform = Transform)
                score_mat [lat_idx, lon_idx] = score[0]
                
                space_score_mat[lat_idx,lon_idx] = score[1]
                                    
                
    return (score_mat, space_score_mat) 