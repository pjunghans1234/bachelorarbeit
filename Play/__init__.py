# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
The mesmer package provides tools to train the MESMER emulator, create emulations, and
analyze the results.
"""

#from importlib.metadata import version as _get_version

from Play import (
    
    dt_functions,
    forest_analise,
    forest_building,
    plot_helper_compare_forest_linreg,
    plot_one_var,
    manuel_forest_tests,
    shape_data,
    plot_helper_dependence,
    LinReg_building,
    LinReg_analise, 
    plot_residuals_scatter
)
#from mesmer._core import _data as data
#from mesmer._core.options import get_options, set_options

# "legacy" modules
__all__ = [
    "plot_one_var",
    "forest_building"
    "manuel_forest_tests",
    "forest_analise",
    "shape_data",
    "dt_functions",
    "plot_helper_dependence",
    "plot_helper_compare_forest_linreg",
    "LinReg_building",
    "LinReg_analise",
    "plot_residuals_scatter"
]



""" 
try:
   __version__ = _get_version("mesmer-emulator")
except Exception:  # pragma: no cover
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

    """