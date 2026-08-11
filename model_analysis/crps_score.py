import numpy as np
import xarray as xr

import xskillscore as xs

from model import transform


#Berechnet crps score der dist_prediction verglichen mit target_ds. Für jeden zu schätzenden Datenpunkt, berechnet sie den score zwischen gezogenen Samples aus der predicteten Verteilung und dem echten Datenpunkt. Siehe https://xskillscore.readthedocs.io/en/stable/api/xskillscore.crps_ensemble.html 
def crps_ensemble_score(mean_prediction_ds,var_prediction_ds, target_ds_t, transformation = None, ensemble_size = 100):

    dist_samples = np.random.normal(
        loc=mean_prediction_ds.prediction.values[..., None],
        scale=np.sqrt(var_prediction_ds.prediction.values)[..., None],
        size=mean_prediction_ds.prediction.shape + (ensemble_size,)
    )

    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=target_ds_t.mrsol.dims + ("member",), #um allgemeiner in der variable zu werden, kann man auch mean_prediction.prediction.dims verwenden
        coords={
            **target_ds_t.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    dist_samples_ds = dist_samples_da.to_dataset(name = "prediction")

    if transformation == "Logit":
        space_score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])


        #invert the tranformation
        target_ds_bt = transform.Logit_Transform_ds_inv(target_ds_t)

        dist_samples_ds_bt = transform.Logit_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_ds_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
        
        space_score = space_score.rename_vars({"prediction": "crps_score"})
        return (score,space_score)


    if transformation == "Log":
        space_score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])

        #invert the tranformation
        target_ds_bt = transform.Log_Transform_ds_inv(target_ds_t)

        dist_samples_ds_bt = transform.Log_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_ds_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
                
        space_score = space_score.rename_vars({"prediction": "crps_score"})
                

        return (score,space_score)
    else :
        score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])

        
        score = score.rename_vars({"prediction": "crps_score"})
        return (score, score)


#Berechnet crps score der dist_prediction verglichen mit target_ds. Für jeden zu schätzenden Datenpunkt, berechnet sie den score zwischen gezogenen Samples aus der predicteten Verteilung und dem echten Datenpunkt. Siehe https://xskillscore.readthedocs.io/en/stable/api/xskillscore.crps_ensemble.html 
def crps_gamma_ensemble_score(shape_prediction_ds,scale_prediction_ds, target_ds_t, transformation = None, ensemble_size = 100):

    dist_samples = np.random.gamma(
        shape=shape_prediction_ds.prediction.values[..., None],
        scale=scale_prediction_ds.prediction.values[..., None],
        size=shape_prediction_ds.prediction.shape + (ensemble_size,)
    )

    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=target_ds_t.mrsol.dims + ("member",), #um allgemeiner in der variable zu werden, kann man auch mean_prediction.prediction.dims verwenden
        coords={
            **target_ds_t.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    dist_samples_ds = dist_samples_da.to_dataset(name = "prediction")

    if transformation == "Logit":
        space_score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])


        #invert the tranformation
        target_ds_bt = transform.Logit_Transform_ds_inv(target_ds_t)

        dist_samples_ds_bt = transform.Logit_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_ds_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
        
        space_score = space_score.rename_vars({"prediction": "crps_score"})
        return (score,space_score)


    if transformation == "Log":
        space_score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])

        #invert the tranformation
        target_ds_bt = transform.Log_Transform_ds_inv(target_ds_t)

        dist_samples_ds_bt = transform.Log_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_ds_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
                
        space_score = space_score.rename_vars({"prediction": "crps_score"})
                

        return (score,space_score)
    else :
        score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])

        
        score = score.rename_vars({"prediction": "crps_score"})
        return (score, score)


def crps_beta_ensemble_score(aplha_prediction_ds,beta_prediction_ds, target_ds_t, transformation = None, mrsol_maximas = None, ensemble_size = 100):

    dist_samples = np.random.beta(
        a=aplha_prediction_ds.prediction.values[..., None],
        b=beta_prediction_ds.prediction.values[..., None],
        size=aplha_prediction_ds.prediction.shape + (ensemble_size,)
    )

    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=target_ds_t.mrsol.dims + ("member",), #um allgemeiner in der variable zu werden, kann man auch mean_prediction.prediction.dims verwenden
        coords={
            **target_ds_t.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    dist_samples_ds = dist_samples_da.to_dataset(name = "prediction")

    if transformation == "rel":
        space_score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])

        if mrsol_maximas is None:
            raise ValueError(
                f"for a backtransformed score you have to give your mrsol_maximas dataset for reference"
                    )
        #invert the tranformation
        target_ds_bt = target_ds_t.copy(deep=True) 
        target_ds_bt["mrsol"] = target_ds_t["mrsol"] * mrsol_maximas["mrsol"]

        dist_samples_ds_bt = dist_samples_ds.copy(deep=True) 
        dist_samples_ds_bt["prediction"] = dist_samples_ds["prediction"] * mrsol_maximas["mrsol"]
        
        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_ds_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
        
        space_score = space_score.rename_vars({"prediction": "crps_score"})
        return (score,space_score)


    if transformation == "Logit":
        space_score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])


        #invert the tranformation
        target_ds_bt = transform.Logit_Transform_ds_inv(target_ds_t)

        dist_samples_ds_bt = transform.Logit_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_ds_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
        
        space_score = space_score.rename_vars({"prediction": "crps_score"})
        return (score,space_score)


    if transformation == "Log":
        space_score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])

        #invert the tranformation
        target_ds_bt = transform.Log_Transform_ds_inv(target_ds_t)

        dist_samples_ds_bt = transform.Log_Transform_ds_inv(dist_samples_da, var = "prediction")

        score = xs.crps_ensemble(target_ds_bt.mrsol,dist_samples_ds_bt,member_dim="member",dim=[])

        score = score.rename_vars({"prediction": "crps_score"})
                
        space_score = space_score.rename_vars({"prediction": "crps_score"})
                

        return (score,space_score)
    else :
        score = xs.crps_ensemble(target_ds_t.mrsol,dist_samples_ds,member_dim="member",dim=[])

        
        score = score.rename_vars({"prediction": "crps_score"})
        return (score, score)

#Important: The target data must be provided in the same transformation space in which the CRPS score is computed.
def add_crpss_against_mean_ensemble_score(crps_score , target_ds ):

    #Maybe it would be useful to put in a warning if the data looks like a transformation was vorgotten?
    
    mrsol_mean_da = target_ds.mrsol.mean("time")

    crps = np.abs(target_ds.mrsol - mrsol_mean_da)
    crps.attrs.clear()

    const_mean_crps_ds = crps.to_dataset(name="const_mean_crps")

    total_crps = crps_score.crps_score.mean("time")

    total_const_mean_crps = const_mean_crps_ds.const_mean_crps.mean("time")

    crpss_score = 1 - total_crps / total_const_mean_crps

    all_scores = crps_score.copy(deep = True)

    all_scores["const_mean_crps"] = const_mean_crps_ds.const_mean_crps

    all_scores["crpss_against_mean"] = crpss_score

    return all_scores

def combine_two_crps_and_the_crpss(crps_score_1, crps_score_2, name_1 = "model_1", name_2 = "model_2"):

    combined_crps = xr.concat([crps_score_1, crps_score_2], dim="model")
    combined_crps.attrs.clear()

    crpss_1_vs_2 = 1 - crps_score_1.mean("time") / crps_score_2.mean("time")
    crpss_2_vs_1 = 1 - crps_score_2.mean("time") / crps_score_1.mean("time")

    all_scores = xr.Dataset(
        {
            name_1 : crps_score_1,
            name_2 : crps_score_2,
            f"crpss_{name_1}_vs_{name_2}": crpss_1_vs_2,
            f"crpss_{name_2}_vs_{name_1}": crpss_2_vs_1
        }
    )

    all_scores.attrs.clear()
    all_scores.attrs["description"] = f"Combined CRPS and CRPSS scores for {name_1} and {name_2}, where CRPSS_1_vs_2 is computed as 1 - CRPS_model_1 / CRPS_model_2. So if CRPSS > 0, model_1 is better than model_2, if CRPSS < 0, model_2 is better than model_1."
    return all_scores
