import xarray as xr
import os
import pystac
import pystac_client
import stackstac
from dask.distributed import Client
import dask
import json
import pandas as pd
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Search for Sentinel-2 images")
    parser.add_argument("cloud_cover", type=str, help="percent cloud cover allowed in images (0-100)")
    parser.add_argument("start_month", type=str, help="first month of year to search for images")
    parser.add_argument("stop_month", type=str, help="last month of year to search for images")
    parser.add_argument("npairs", type=str, help="number of pairs per image")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Hardcode bbox for now
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [11.386104222980578,78.80418486060194],
                [11.386104222980578,78.57706771985502],
                [13.269392947720405,78.57706771985502],
                [14.67046173123066, 78.28521422287416],
                [13.269392947720405,78.80418486060194],
                [11.386104222980578,78.80418486060194]
            ]
        ]
    }
    
    # Use the API from Element84 to query the data
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=bbox,
        query={"eo:cloud_cover": {"lt": float(args.cloud_cover)}}
    )
    
    # Check how many items were returned
    items = search.item_collection()
    print(f"Returned {len(items)} Items")
    
    # Create xarray dataset without loading data
    # Specify a common CRS (e.g., EPSG:32633)
    try:
        sentinel2_stack = stackstac.stack(items, epsg=32633)
    except ValueError as e:
        print(f"Error during stacking: {e}")
        # Debugging CRS conflicts
        crs_set = {asset["proj:epsg"] for item in items for asset in item.assets.values()}
        print(f"CRSs in assets: {crs_set}")
        return

    # Filter to specified month range
    sentinel2_stack_snowoff = sentinel2_stack.where(
        (sentinel2_stack.time.dt.month >= int(args.start_month)) & 
        (sentinel2_stack.time.dt.month <= int(args.stop_month)), 
        drop=True
    )
    
    # Select the first image of each month
    period_index = pd.PeriodIndex(sentinel2_stack_snowoff['time'].values, freq='M')
    sentinel2_stack_snowoff.coords['year_month'] = ('time', period_index)
    first_image_indices = sentinel2_stack_snowoff.groupby('year_month').apply(lambda x: x.isel(time=0))
    
    product_names = first_image_indices['s2:product_uri'].values.tolist()
    print('\n'.join(product_names))
    
    # Create Matrix Job Mapping (JSON Array)
    pairs = []
    for r in range(len(product_names) - int(args.npairs)):
        for s in range(1, int(args.npairs) + 1):
            img1_product_name = product_names[r]
            img2_product_name = product_names[r + s]
            shortname = f'{img1_product_name[11:19]}_{img2_product_name[11:19]}'
            pairs.append({'img1_product_name': img1_product_name, 'img2_product_name': img2_product_name, 'name': shortname})
    
    matrixJSON = f'{{"include":{json.dumps(pairs)}}}'
    print(f'Number of image pairs: {len(pairs)}')
    
    # Output the results to the environment's GitHub output file
    github_output = os.environ.get('GITHUB_OUTPUT', 'output.txt')
    with open(github_output, 'a') as f:
        print(f'IMAGE_DATES={product_names}', file=f)
        print(f'MATRIX_PARAMS_COMBINATIONS={matrixJSON}', file=f)

if __name__ == "__main__":
    main()
