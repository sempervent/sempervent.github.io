import os
import tempfile
from fastapi import FastAPI, HTTPException
from app.raster import synthetic_geotiff, inspect_raster, gpu_normalize

app = FastAPI(title="Geospatial Slim Demo")

_SYNTHETIC = os.path.join(tempfile.gettempdir(), "synthetic.tif")
synthetic_geotiff(_SYNTHETIC)


@app.get("/health")
def health():
    from osgeo import gdal
    return {"status": "ok", "gdal_version": gdal.VersionInfo()}


@app.get("/inspect")
def inspect(path: str = _SYNTHETIC):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return inspect_raster(path)


@app.get("/normalize")
def normalize(path: str = _SYNTHETIC):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return gpu_normalize(path)
