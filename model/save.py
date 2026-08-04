from model import config as conf



# Speichert einen Plot mit der gewünschten Speicher strucktur
def save_params(prameter_tuple, var = "", scen = "",folder = "", name_prefix = "", name = "", name_postfix = ""):
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
      
    ####ToDo sinnvolle speichervariante finden    
    out = file_path / f'paramerters{var}{scen}{name}{name_postfix}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    prameter_tuple.savefig(out)



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