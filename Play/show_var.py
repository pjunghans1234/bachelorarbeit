
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import show_var_confic as conf

xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)

#xmode minimal
#matplotlib inline
#config InlineBackend.figure_format='retina'


#Some manual tests
#------------------------------------------------------------------------------------------------------------------------
ds_var = xr.load_dataset( conf.path_to_data / conf.standart_path)


getattr(ds_var, conf.var).mean(dim="time").plot(x="lon");
plt.title("temperatur 2m over ground") #woher kommt das automatische height = 2.0[m]
plt.show()
plt.savefig(Path(conf.path_to_output / f'plot_{conf.var}_{conf.scen}'))

#---------------------------------------------------------------------------------------------------------------------
