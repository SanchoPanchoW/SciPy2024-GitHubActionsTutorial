#! /usr/bin/env python

import xarray as xr
import rasterio as rio
import rioxarray
import numpy as np
import os
from autoRIFT import autoRIFT
from scipy.interpolate import interpn
import pystac
import pystac_client
import stackstac
from dask.distributed import Client
import geopandas as gpd
from shapely.geometry import shape
import dask
import warnings
import argparse

# Silence some warnings from stackstac and autoRIFT
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def download_s2(img1_product_name, img2_product_name, bbox):
    '''
    Download a pair of Sentinel-2 images acquired on given dates over a given bounding box.
    '''
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        query=[f's2:product_uri={img1_product_name}']
    )
    img1_items = search.item_collection()
    img1_full = stackstac.stack(img1_items)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        query=[f's2:product_uri={img2_product_name}']
    )
    img2_items = search.item_collection()
    img2_full = stackstac.stack(img2_items)

    aoi = gpd.GeoDataFrame({'geometry': [shape(bbox)]})
    img1_clipped = img1_full.rio.clip_box(*aoi.total_bounds, crs=4326)
    img2_clipped = img2_full.rio.clip_box(*aoi.total_bounds, crs=4326)

    img1_ds = img1_clipped.to_dataset(dim='band')
    img2_ds = img2_clipped.to_dataset(dim='band')

    return img1_ds, img2_ds

def run_autoRIFT(img1, img2, skip_x=3, skip_y=3, min_x_chip=16, max_x_chip=64,
                 preproc_filter_width=3, mpflag=4, search_limit_x=30, search_limit_y=30):
    '''
    Configure and run autoRIFT feature tracking with Sentinel-2 data for large mountain glaciers.
    '''
    obj = autoRIFT()
    obj.MultiThread = mpflag

    obj.I1 = img1
    obj.I2 = img2

    obj.SkipSampleX = skip_x
    obj.SkipSampleY = skip_y

    obj.ChipSizeMinX = min_x_chip
    obj.ChipSizeMaxX = max_x_chip
    obj.ChipSize0X = min_x_chip
    obj.OverSampleRatio = {obj.ChipSize0X: 16, obj.ChipSize0X * 2: 32, obj.ChipSize0X * 4: 64}

    # Generate grid
    m, n = obj.I1.shape
    xGrid = np.arange(obj.SkipSampleX, n - obj.SkipSampleX, obj.SkipSampleX)
    yGrid = np.arange(obj.SkipSampleY, m - obj.SkipSampleY, obj.SkipSampleY)

    # Clip indices to ensure they are within valid bounds
    xGrid_clipped = np.clip(xGrid - 1, 0, n - 1)
    yGrid_clipped = np.clip(yGrid - 1, 0, m - 1)

    # Create 2D grids for x and y
    nd = len(xGrid)
    md = len(yGrid)
    obj.xGrid = np.int32(np.dot(np.ones((md, 1)), np.reshape(xGrid, (1, nd))))
    obj.yGrid = np.int32(np.dot(np.reshape(yGrid, (md, 1)), np.ones((1, nd))))

    # Create no-data mask
    noDataMask = np.invert(
        np.logical_and(
            obj.I1[:, xGrid_clipped][yGrid_clipped, :] > 0,
            obj.I2[:, xGrid_clipped][yGrid_clipped, :] > 0,
        )
    )

    # Set search limits and offsets
    obj.SearchLimitX = np.full_like(obj.xGrid, search_limit_x)
    obj.SearchLimitY = np.full_like(obj.xGrid, search_limit_y)
    obj.SearchLimitX *= np.logical_not(noDataMask)
    obj.SearchLimitY *= np.logical_not(noDataMask)
    obj.Dx0 = np.zeros_like(obj.SearchLimitX)
    obj.Dy0 = np.zeros_like(obj.SearchLimitY)

    print("Preprocessing images...")
    obj.WallisFilterWidth = preproc_filter_width
    obj.preprocess_filt_lap()
    obj.uniform_data_type()

    print("Starting autoRIFT...")
    obj.runAutorift()
    print("autoRIFT complete.")

    obj.Dx_m = obj.Dx * 10
    obj.Dy_m = obj.Dy * 10

    return obj

def prep_outputs(obj, img1_ds, img2_ds):
    '''
    Interpolate pixel offsets to original dimensions and calculate total horizontal velocity.
    '''
    x_coords = obj.xGrid[0, :]
    y_coords = obj.yGrid[:, 0]

    x_coords_new, y_coords_new = np.meshgrid(
        np.arange(obj.I2.shape[1]),
        np.arange(obj.I2.shape[0])
    )

    Dx_full = interpn((y_coords, x_coords), obj.Dx, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_full = interpn((y_coords, x_coords), obj.Dy, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dx_m_full = interpn((y_coords, x_coords), obj.Dx_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_m_full = interpn((y_coords, x_coords), obj.Dy_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)

    img1_ds = img1_ds.assign({'Dx': (['y', 'x'], Dx_full),
                              'Dy': (['y', 'x'], Dy_full),
                              'Dx_m': (['y', 'x'], Dx_m_full),
                              'Dy_m': (['y', 'x'], Dy_m_full)})

    img1_ds['veloc_x'] = (img1_ds.Dx_m / (img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days) * 365.25
    img1_ds['veloc_y'] = (img1_ds.Dy_m / (img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days) * 365.25
    img1_ds['veloc_horizontal'] = np.sqrt(img1_ds['veloc_x'] ** 2 + img1_ds['veloc_y'] ** 2)

    return img1_ds

def get_parser():
    parser = argparse.ArgumentParser(description="Run autoRIFT to find pixel offsets for two Sentinel-2 images")
    parser.add_argument("img1_product_name", type=str, help="Product name of first Sentinel-2 image")
    parser.add_argument("img2_product_name", type=str, help="Product name of second Sentinel-2 image")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [11.386104222980578, 78.80418486060194],
                [11.386104222980578, 78.57706771985502],
                [13.269392947720405, 78.57706771985502],
                [13.269392947720405, 78.80418486060194],
                [11.386104222980578, 78.80418486060194]
            ]
        ],
    }

    img1_ds, img2_ds = download_s2(args.img1_product_name, args.img2_product_name, bbox)
    img1 = img1_ds.nir.squeeze().values
    img2 = img2_ds.nir.squeeze().values

    search_limit_x = search_limit_y = round(((((img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days) * 100) / 365.25).item())
    obj = run_autoRIFT(img1, img2, search_limit_x=search_limit_x, search_limit_y=search_limit_y)
    ds = prep_outputs(obj, img1_ds, img2_ds)

    ds.veloc_horizontal.rio.to_raster(f'S2_{args.img1_product_name[11:19]}_{args.img2_product_name[11:19]}_horizontal_velocity.tif')

if __name__ == "__main__":
    main()
