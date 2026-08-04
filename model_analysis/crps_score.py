import numpy as np
import xarray as xr

import xskillscore as xs

from model import transform


#Berechnet crps score der dist_prediction verglichen mit target_ds. Für jeden zu schätzenden Datenpunkt, berechnet sie den score zwischen gezogenen Samples aus der predicteten Verteilung und dem echten Datenpunkt. Siehe https://xskillscore.readthedocs.io/en/stable/api/xskillscore.crps_ensemble.html 
def crps_ensemble_score(mean_prediction_ds,var_prediction_ds, target_ds, transformation = None, ensemble_size = 100):

    dist_samples = np.random.normal(
        loc=mean_prediction_ds.prediction.values[..., None],
        scale=np.sqrt(var_prediction_ds.prediction.values)[..., None],
        size=mean_prediction_ds.prediction.shape + (ensemble_size,)
    )

    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=target_ds.mrsol.dims + ("member",), #um allgemeiner in der variable zu werden, kann man auch mean_prediction.prediction.dims verwenden
        coords={
            **target_ds.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    if transformation == "Logit":
        space_score = xs.crps_ensemble(target_ds.mrsol,dist_samples_da,member_dim="member",dim=[])


        #invert the tranformation
        target_ds_bt = transform.Logit_Transform_ds_inv(target_ds)

        dist_samples_da_bt = transform.Logit_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_da_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
        
        space_score = space_score.rename_vars({"prediction": "crps_score"})
        return (score,space_score)


    if transformation == "Log":
        space_score = xs.crps_ensemble(target_ds.mrsol,dist_samples_da,member_dim="member",dim=[])

        #invert the tranformation
        target_ds_bt = transform.Log_Transform_ds_inv(target_ds)

        dist_samples_da_bt = transform.Log_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_da_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
                
        space_score = space_score.rename_vars({"prediction": "crps_score"})
                

        return (score,space_score)
    else :
        score = xs.crps_ensemble(target_ds.mrsol,dist_samples_da,member_dim="member",dim=[])

        
        score = score.rename_vars({"prediction": "crps_score"})
        return (score, score)

def crpss():
    return "Todo"