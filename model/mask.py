import xarray as xr


#Aus Mesmer kopiert
def _where_if_coords(obj, cond, coords):

    # xarray applies where to all data_vars - even if they do not have the corresponding
    # dimensions - we don't want that https://github.com/pydata/xarray/issues/7027

    def _where(da):
        if all(coord in da.coords for coord in coords):
            return da.where(cond)
        return da

    if isinstance(obj, xr.Dataset):
        return obj.map(_where, keep_attrs=True)

    return obj.where(cond)

#Maskiert Orte aus, die nicht zu allen Zeiten gültige Werte habe, i.e. Werte zu nah an null.
def mask_nonpositivs (data_set, variable = "mrsol", threshold = 1e-4,  x_coords: str = "lon", y_coords: str = "lat"):

    mask_bool = (data_set[variable] > threshold).all(dim = "time")

    
    return _where_if_coords(data_set,mask_bool,[y_coords, x_coords]), mask_bool


#Maskiert Orte aus, die nicht zu allen Zeiten gültige Werte habe, i.e. Werte zu nah an null. Auch für andere Target dim die nicht height ist adaptierbar
def mask_nonpositiv_height_chunks (data_set, variable = "mrsol", threshold = 1e-4,  x_coords: str = "lon", y_coords: str = "lat"):

    mask_bool = (data_set[variable] > threshold).all(dim = "time").any("depth")

    data_set = _where_if_coords(data_set,mask_bool,[y_coords, x_coords])

    return data_set.clip(min=threshold), mask_bool


def mask_mask (data_set,mask_bool, x_coords: str = "lon", y_coords: str = "lat"):

    return _where_if_coords(data_set,mask_bool,[y_coords, x_coords])