"""GDAL-based raster inspection utilities."""
from __future__ import annotations

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


def synthetic_geotiff(path: str, width: int = 256, height: int = 256) -> None:
    """Write a tiny synthetic single-band GeoTIFF to *path* for testing."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([-180.0, 360.0 / width, 0, 90.0, 0, -180.0 / height])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    data = np.random.rand(height, width).astype(np.float32) * 100
    band.WriteArray(data)
    band.SetNoDataValue(-9999)
    ds.FlushCache()
    ds = None


def inspect_raster(path: str) -> dict:
    """Return metadata + band statistics for a GeoTIFF."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f"Cannot open: {path}")

    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = srs.GetAuthorityCode(None)

    result = {
        "driver": ds.GetDriver().ShortName,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "bands": ds.RasterCount,
        "epsg": epsg,
        "resolution_x": abs(gt[1]),
        "resolution_y": abs(gt[5]),
        "bbox": {
            "west": gt[0],
            "north": gt[3],
            "east": gt[0] + gt[1] * ds.RasterXSize,
            "south": gt[3] + gt[5] * ds.RasterYSize,
        },
        "bands_stats": [],
    }

    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        stats = band.ComputeStatistics(False)
        result["bands_stats"].append({
            "band": i,
            "min": stats[0],
            "max": stats[1],
            "mean": stats[2],
            "stddev": stats[3],
        })

    ds = None
    return result


def gpu_normalize(path: str) -> dict:
    """Normalize band 1 to [0,1] using CuPy if available; fall back to NumPy."""
    try:
        import cupy as cp
        xp = cp
        backend = "cupy"
    except ImportError:
        xp = np
        backend = "numpy"

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    arr = xp.asarray(data, dtype=xp.float32)
    arr_min, arr_max = float(xp.nanmin(arr)), float(xp.nanmax(arr))
    if arr_max > arr_min:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = xp.zeros_like(arr)

    return {
        "backend": backend,
        "original_min": arr_min,
        "original_max": arr_max,
        "normalized_mean": float(xp.nanmean(norm)),
    }
