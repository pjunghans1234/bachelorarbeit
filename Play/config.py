import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

path_to_data = Path('/home/pjunghans/BachelorArbeit/Play_data_MPI-ESM1-2-LR/cmip6-ng-coarse-grid')

path_to_output = Path('/home/pjunghans/BachelorArbeit/Play_output')

run = 'r10i1p1f1'

test_run = 'r11i1p1f1'

hold_out_run = 'r12i1p1f1'

ten_run_set = ["r20i1p1f1","r21i1p1f1","r22i1p1f1","r23i1p1f1","r24i1p1f1","r25i1p1f1","r26i1p1f1","r27i1p1f1","r28i1p1f1","r29i1p1f1"]