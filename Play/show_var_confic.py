#Sehr altes File nicht weiter verwendet 

from pathlib import Path

path_to_data = Path('/home/pjunghans/BachelorArbeit/Play_data_MPI-ESM1-2-LR_r10_r11/cmip6-ng-coarse-grid')

path_to_output = Path('/home/pjunghans/BachelorArbeit/Play_output')



# verwendete Daten
# ----------------------------------------------------------------------------------------------------
scen = 'ssp585'

#Datenset 1
var = 'hurs'
run = 'r10i1p1f1'
standart_path = Path(f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc')


#optionales Datenset 2
var_2 = 'pr'
run2 = 'r10i1p1f1'
standart_path_2 = Path(f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc')

#optionales Datenset 3
var_3 = 'pr'
run_3 = 'r10i1p1f1'
standart_path_3 = Path(f'{var}/mon/g025/{var}_mon_MPI-ESM1-2-LR_{scen}_{run}_g025.nc')
# ----------------------------------------------------------------------------------------------------


# Test / Plottyp
# ----------------------------------------------------------------------------------------------------
# vermutlich doch nicht richtige struktur:)

