import numpy as np
import xarray as xr
from scipy.stats import skewnorm

import xskillscore as xs

from model import transform

def multiplication_with_maximas(dist_samples_ds, maximas_da):
    if not set(maximas_da.dims).issubset(dist_samples_ds.dims):
        raise ValueError(
            f"dimensions of maximas_da {maximas_da.dims} are not a subset of dimensions of dist_samples_ds {dist_samples_ds.dims}"
        )
    _, maximas_da = xr.align(
    dist_samples_ds,
    maximas_da,
    join="exact"
    )
    maximas_da = maximas_da.broadcast_like(dist_samples_ds)
    return dist_samples_ds * maximas_da
    

def crps_norm_score(mean_prediction_ds,var_prediction_ds, target_ds, transformation = None,maximas_da = None, ensemble_size = 100):

    var_prediction_ds = var_prediction_ds.clip(min = 0)
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

    dist_samples_ds = dist_samples_da.to_dataset(name = "prediction")

    if transformation == "Logit":

        dist_samples_ds = transform.Logit_Transform_ds_inv(dist_samples_ds, var = "prediction")

        dist_samples_ds = multiplication_with_maximas(dist_samples_ds, maximas_da)

        
    elif transformation == "Log":
        
        dist_samples_ds = transform.Log_Transform_ds_inv(dist_samples_ds, var = "prediction")

    elif transformation is not None:
        raise ValueError(f"transformation {transformation} is not supported. Please choose from 'Logit', 'Log' or None.")
    
    score = xs.crps_ensemble(target_ds.mrsol,dist_samples_ds,member_dim="member",dim=[])

    score = score.rename_vars({"prediction": "crps_score"})
    return score

def convert_skew_parameters(mean, var, skew):
    delta_sign = np.sign(skew)
    const = (2-np.pi/2)**(2/3)
    abs_skew_pot = np.abs(skew)**(2/3)
    
    delta = delta_sign * np.sqrt((np.pi/2)*abs_skew_pot/(abs_skew_pot+const))

    a = delta/np.sqrt(1-delta**2)
    scale = np.sqrt(var/(1-2*delta**2/np.pi))
    loc = mean - scale * delta *np.sqrt(2/np.pi)

    return a, scale, loc 


def crps_skew_score(mean_prediction_ds,var_prediction_ds,skew_prediction_ds, target_ds, transformation = None, maximas_da = None, ensemble_size = 100):

    skew_prediction_ds["prediction"] = skew_prediction_ds.prediction.clip(min = -0.995 ).clip(max = 0.995)
    var_prediction_ds["prediction"] = var_prediction_ds.prediction.clip(min = 1e-32)

    a,scale,loc = convert_skew_parameters(mean_prediction_ds.prediction.values,var_prediction_ds.prediction.values,skew_prediction_ds.prediction.values)

    dist_samples = skewnorm.rvs(a=a[...,None], loc= loc[...,None], scale = scale[...,None] , size=mean_prediction_ds.prediction.shape + (ensemble_size,) ) 

    
    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=target_ds.mrsol.dims + ("member",), #um allgemeiner in der variable zu werden, kann man auch mean_prediction.prediction.dims verwenden
        coords={
            **target_ds.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    dist_samples_ds = dist_samples_da.to_dataset(name = "prediction")

    if transformation == "Logit":

        dist_samples_ds = transform.Logit_Transform_ds_inv(dist_samples_ds, var = "prediction")

        dist_samples_ds = multiplication_with_maximas(dist_samples_ds, maximas_da)

    elif transformation == "Log":
        
        dist_samples_ds = transform.Log_Transform_ds_inv(dist_samples_ds, var = "prediction")

    elif transformation is not None:
        raise ValueError(f"transformation {transformation} is not supported. Please choose from 'Logit', 'Log' or None.")
    score = xs.crps_ensemble(target_ds.mrsol,dist_samples_ds,member_dim="member",dim=[])

    score = score.rename_vars({"prediction": "crps_score"})
    return score


def crps_gamma_score(mean_prediction_ds,var_prediction_ds, target_ds, transformation = None,maximas_da = None, ensemble_size = 100):
    

    shape_prediction_ds = mean_prediction_ds**2/var_prediction_ds
    scale_prediction_ds = var_prediction_ds/mean_prediction_ds


    dist_samples = np.random.gamma(
        shape=shape_prediction_ds.prediction.values[..., None],
        scale=scale_prediction_ds.prediction.values[..., None],
        size=shape_prediction_ds.prediction.shape + (ensemble_size,)
    )

    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=target_ds.mrsol.dims + ("member",), #um allgemeiner in der variable zu werden, kann man auch mean_prediction.prediction.dims verwenden
        coords={
            **target_ds.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    dist_samples_ds = dist_samples_da.to_dataset(name = "prediction")

    if transformation == "Logit":
        
        dist_samples_ds = transform.Logit_Transform_ds_inv(dist_samples_da, var = "prediction")

        dist_samples_ds = multiplication_with_maximas(dist_samples_ds, maximas_da)


    elif transformation == "Log":
        
        dist_samples_ds = transform.Log_Transform_ds_inv(dist_samples_da, var = "prediction")

    elif transformation is not None:
        raise ValueError(f"transformation {transformation} is not supported. Please choose from 'Logit', 'Log' or None.")

    score = xs.crps_ensemble(target_ds.mrsol,dist_samples_ds,member_dim="member",dim=[])

    score = score.rename_vars({"prediction": "crps_score"})
    return score


def crps_beta_score(mean_prediction_ds,var_prediction_ds, target_ds, transformation = None, maximas_da = None, ensemble_size = 100):

    mean = mean_prediction_ds.copy(deep=True)
    var = var_prediction_ds.copy(deep=True).clip(min=1e-32)

#todo precision parameter in guter Quelle finden
    precision = (mean*(1-mean))/var-1

    alpha_prediction_ds = (mean*precision).clip(min=1e-32)
    beta_prediction_ds = ((1-mean)*precision).clip(min=1e-32)

    dist_samples = np.random.beta(
        a=alpha_prediction_ds.prediction.values[..., None],
        b=beta_prediction_ds.prediction.values[..., None],
        size=alpha_prediction_ds.prediction.shape + (ensemble_size,)
    )

    dist_samples_da = xr.DataArray(
        dist_samples,
        dims=target_ds.mrsol.dims + ("member",), #um allgemeiner in der variable zu werden, kann man auch mean_prediction.prediction.dims verwenden
        coords={
            **target_ds.mrsol.coords,
            "member": np.arange(dist_samples.shape[-1])
        }
    )

    dist_samples_ds = dist_samples_da.to_dataset(name = "prediction")

    if transformation == "rel":

        if maximas_da is None:
            raise ValueError(
                f"for a backtransformed score you have to give your mrsol_maximas dataset for reference"
                    )
        dist_samples_ds = multiplication_with_maximas(dist_samples_ds, maximas_da) 
        
        
    elif transformation == "Logit":
    
        dist_samples_ds = transform.Logit_Transform_ds_inv(dist_samples_ds, var = "prediction")

        dist_samples_ds = multiplication_with_maximas(dist_samples_ds, maximas_da)  
        
    elif transformation == "Log":
        
        dist_samples_ds = transform.Log_Transform_ds_inv(dist_samples_ds, var = "prediction")

    elif transformation is not None:
        raise ValueError(f"transformation {transformation} is not supported. Please choose from 'Logit', 'Log' or None.")
    
    score = xs.crps_ensemble(target_ds.mrsol,dist_samples_ds,member_dim="member",dim=[])

    score = score.rename_vars({"prediction": "crps_score"})
        
    return score

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

###############################################################################################################
# old and longer variants of the crps score functions, which are now replaced by the more general ones above. 



#Berechnet crps score der dist_prediction verglichen mit target_ds. Für jeden zu schätzenden Datenpunkt, berechnet sie den score zwischen gezogenen Samples aus der predicteten Verteilung und dem echten Datenpunkt. Siehe https://xskillscore.readthedocs.io/en/stable/api/xskillscore.crps_ensemble.html 
def crps_ensemble_score(mean_prediction_ds,var_prediction_ds, target_ds_t, transformation = None,maximas_ds = None, ensemble_size = 100):

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

        #probably check that target and maximas have the same shape
        target_ds_bt = target_ds_bt * maximas_ds

        dist_samples_ds_bt = transform.Logit_Transform_ds_inv(dist_samples_da, var = "prediction")

        dist_samples_ds_bt = dist_samples_ds_bt * maximas_ds

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

#todo
def crps_skew_ensemble_score(mean_prediction_ds,var_prediction_ds, target_ds_t, transformation = None, ensemble_size = 100):

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


def crps_beta_ensemble_score(aplha_prediction_ds,beta_prediction_ds, target_ds_t, bug_target=None, transformation = None, mrsol_maximas = None, ensemble_size = 100):

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
        if bug_target is not None:
            target_ds_bt = bug_target
        else:
            target_ds_bt = target_ds_t.copy(deep=True) 
            target_ds_bt["mrsol"] = target_ds_t["mrsol"] * mrsol_maximas["mrsol"]
        
        dist_samples_ds_bt = multiplication_with_maximas(dist_samples_ds, mrsol_maximas)  

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
