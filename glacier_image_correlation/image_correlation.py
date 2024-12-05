#!/usr/bin/env python

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
import geopandas as gpd
from shapely.geometry import shape
import warnings
import argparse

# Silence some warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def download_s2(img1_product_name, img2_product_name, bbox):
    """
    Download Sentinel-2 images for the given product names and bounding box
    """
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)

    # Search for the first image
    search1 = catalog.search(
        collections=["sentinel-2-l2a"],
        query={"s2:product_uri": img1_product_name},
    )
    img1_items = search1.item_collection()
    img1_full = stackstac.stack(img1_items)

    # Search for the second image
    search2 = catalog.search(
        collections=["sentinel-2-l2a"],
        query={"s2:product_uri": img2_product_name},
    )
    img2_items = search2.item_collection()
    img2_full = stackstac.stack(img2_items)

    # Clip images to the bounding box
    aoi = gpd.GeoDataFrame({'geometry': [shape(bbox)]})
    img1_clipped = img1_full.rio.clip_box(*aoi.total_bounds, crs=4326)
    img2_clipped = img2_full.rio.clip_box(*aoi.total_bounds, crs=4326)

    img1_ds = img1_clipped.to_dataset(dim="band")
    img2_ds = img2_clipped.to_dataset(dim="band")

    return img1_ds, img2_ds


def run_autoRIFT(img1, img2, skip_x=3, skip_y=3, min_x_chip=16, max_x_chip=64,
                 preproc_filter_width=3, mpflag=4, search_limit_x=30, search_limit_y=30):
    """
    Run autoRIFT for Sentinel-2 data
    """
    obj = autoRIFT()
    obj.MultiThread = mpflag
    obj.I1 = img1
    obj.I2 = img2

    obj.SkipSampleX = skip_x
    obj.SkipSampleY = skip_y

    m, n = obj.I1.shape
    xGrid = np.arange(obj.SkipSampleX + 10, n, obj.SkipSampleX)
    yGrid = np.arange(obj.SkipSampleY + 10, m, obj.SkipSampleY)

    # Clip indices to avoid out-of-bound errors
    xGrid = np.clip(xGrid, 0, n - 1)
    yGrid = np.clip(yGrid, 0, m - 1)

    # No-data mask
    xGrid_clipped = np.clip(xGrid - 1, 0, n - 1)
    yGrid_clipped = np.clip(yGrid - 1, 0, m - 1)
    noDataMask = np.invert(
        np.logical_and(
            obj.I1[:, xGrid_clipped][yGrid_clipped, :] > 0,
            obj.I2[:, xGrid_clipped][yGrid_clipped, :] > 0,
        )
    )

    obj.SearchLimitX = np.full_like(obj.xGrid, search_limit_x)
    obj.SearchLimitY = np.full_like(obj.xGrid, search_limit_y)

    obj.SearchLimitX *= np.logical_not(noDataMask)
    obj.SearchLimitY *= np.logical_not(noDataMask)
    obj.Dx0 = obj.Dx0 * np.logical_not(noDataMask)
    obj.Dy0 = obj.Dy0 * np.logical_not(noDataMask)
    obj.Dx0[noDataMask] = 0
    obj.Dy0[noDataMask] = 0
    obj.NoDataMask = noDataMask

    print("Preprocessing images")
    obj.WallisFilterWidth = preproc_filter_width
    obj.preprocess_filt_lap()
    obj.uniform_data_type()

    print("Starting autoRIFT")
    obj.runAutorift()
    print("autoRIFT complete")

    obj.Dx_m = obj.Dx * 10
    obj.Dy_m = obj.Dy * 10

    return obj


def prep_outputs(obj, img1_ds, img2_ds):
    """
    Prepare the output dataset with velocity components and horizontal velocity
    """
    x_coords = obj.xGrid[0, :]
    y_coords = obj.yGrid[:, 0]

    x_coords_new, y_coords_new = np.meshgrid(
        np.arange(obj.I2.shape[1]),
        np.arange(obj.I2.shape[0]),
    )

    Dx_full = interpn((y_coords, x_coords), obj.Dx, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_full = interpn((y_coords, x_coords), obj.Dy, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dx_m_full = interpn((y_coords, x_coords), obj.Dx_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_m_full = interpn((y_coords, x_coords), obj.Dy_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)

    img1_ds = img1_ds.assign({
        "Dx": (["y", "x"], Dx_full),
        "Dy": (["y", "x"], Dy_full),
        "Dx_m": (["y", "x"], Dx_m_full),
        "Dy_m": (["y", "x"], Dy_m_full),
    })

    time_diff = (img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days
    img1_ds["veloc_x"] = (img1_ds.Dx_m / time_diff) * 365.25
    img1_ds["veloc_y"] = (img1_ds.Dy_m / time_diff) * 365.25
    img1_ds["veloc_horizontal"] = np.sqrt(img1_ds["veloc_x"]**2 + img1_ds["veloc_y"]**2)

    return img1_ds


def get_parser():
    parser = argparse.ArgumentParser(description="Run autoRIFT on Sentinel-2 images")
    parser.add_argument("img1_product_name", type=str, help="First Sentinel-2 image product name")
    parser.add_argument("img2_product_name", type=str, help="Second Sentinel-2 image product name")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    bbox = {
        "type": "Polygon",
        "coordinates": [[
            [11.3861, 78.8042],
            [11.3861, 78.5771],
            [13.2694, 78.5771],
            [13.2694, 78.8042],
            [11.3861, 78.8042],
        ]],
    }

    img1_ds, img2_ds = download_s2(args.img1_product_name, args.img2_product_name, bbox)

    img1 = img1_ds.nir.squeeze().values
    img2 = img2_ds.nir.squeeze().values

    time_diff_days = (img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days.item()
    search_limit_x = search_limit_y = round((time_diff_days * 100) / 365.25)

    obj = run_autoRIFT(img1, img2, search_limit_x=search_limit_x, search_limit_y=search_limit_y)
    ds = prep_outputs(obj, img1_ds, img2_ds)

    output_file = f"S2_{args.img1_product_name[11:19]}_{args.img2_product_name[11:19]}_horizontal_velocity.tif"
    ds.veloc_horizontal.rio.to_raster(output_file)


if __name__ == "__main__":
    main()
