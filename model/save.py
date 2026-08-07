from model import config as conf



# Speichert einen Plot mit der gewünschten Speicher strucktur
def save_params(mean_parameter,varience_parameter ,var = "", scen = "",folder = "", name_prefix = "", name = "", name_postfix = ""):
    if folder != "":
        file_path = conf.path_to_parametere_output / folder
    else:
        file_path = conf.path_to_parametere_output

    if var != "":
        var = "_" + var
    if scen != "":
        scen = "_" + scen
    if name_prefix != "":
        name_prefix = "_" +  name_prefix  
    if name != "":
        name = "_" + name
    if name_postfix != "":
        name_postfix = "_" + name_postfix
      

    out_mean = file_path / f'mean_paramerters{var}{scen}{name}{name_postfix}.nc'
    out_mean.parent.mkdir(parents=True, exist_ok=True)

    out_variance = file_path / f'variance_paramerters{var}{scen}{name}{name_postfix}.nc'
    out_variance.parent.mkdir(parents=True, exist_ok=True)

    mean_parameter.to_netcdf(out_mean)
    varience_parameter.to_netcdf(out_variance)

    



# Speichert einen Plot mit der gewünschten Speicher strucktur
def save_plot(fig, var = "", scen = "",folder = "", name_prefix = "", name = "", name_postfix = ""):
    if folder != "":
        file_path = conf.path_to_plot_output / folder
    else:
        file_path = conf.path_to_plot_output

    if var != "":
        var = "_" + var
    if scen != "":
        scen = "_" + scen
    if name_prefix != "":
        name_prefix = "_" +  name_prefix  
    if name != "":
        name = "_" + name
    if name_postfix != "":
        name_postfix = "_" + name_postfix
      
        
    out = file_path / f'plot{var}{scen}{name}{name_postfix}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)



def save_crps_scores(crps_score,var = "", scen = "",folder = "", name_prefix = "", name = "", name_postfix = ""):

    if folder != "":
        file_path = conf.path_to_scores_output / folder
    else:
        file_path = conf.path_to_scores_output

    if var != "":
        var = "_" + var
    if scen != "":
        scen = "_" + scen
    if name_prefix != "":
        name_prefix = "_" +  name_prefix  
    if name != "":
        name = "_" + name
    if name_postfix != "":
        name_postfix = "_" + name_postfix
      

    out = file_path / f'crps_scores{var}{scen}{name}{name_postfix}.nc'
    out.parent.mkdir(parents=True, exist_ok=True)

    crps_score.to_netcdf(out)