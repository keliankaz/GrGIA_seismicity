from sklearn.neighbors import BallTree
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import MultiLineString
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge
from typing import Callable

EARTH_RADIUS = 6371  # km


def densify_geometry(line_geometry, step, crs_in=None, crs_out=None):
    # crs: epsg code of a coordinate reference system you want your line to be georeferenced with
    # step: add a vertice every step in whatever unit your coordinate reference system use.

    length_m = line_geometry.length  # get the length

    xy = []  # to store new tuples of coordinates

    for distance_along_old_line in np.arange(0, int(length_m), step):
        point = line_geometry.interpolate(
            distance_along_old_line
        )  # interpolate a point every step along the old line
        xp, yp = point.x, point.y  # extract the coordinates

        xy.append((xp, yp))  # and store them in xy list

    new_line = LineString(
        xy
    )  # Here, we finally create a new line with densified points.

    if (
        crs_in != None
    ):  #  If you want to georeference your new geometry, uses crs to do the job.
        new_line_geo = gpd.geoseries.GeoSeries(new_line, crs=crs_in)

    if crs_out != None:
        new_line_geo = new_line_geo.to_crs(crs_out)
        return new_line_geo

    else:
        return new_line


def radius_search(points, queries, radius) -> list[list]:
    """Searches for points within a radius of a query lat lon point and returns a list of corresponding indices."""
    tree = BallTree(np.deg2rad(points), metric="haversine")
    return tree.query_radius(
        np.deg2rad(queries), r=radius / EARTH_RADIUS, return_distance=False
    )


def k_nearest_search(points, queries, k=1):
    """Searches for the indices of the k nearest points to a query lat lon point and returns a list of corresponding indices."""
    tree = BallTree(np.deg2rad(points), metric="haversine")
    return tree.query(np.deg2rad(queries), k=k)[1]


def get_geometry_neighbors(
    catalog: pd.DataFrame,
    boundary: LineString,
    fun: Callable = radius_search,
    kwarg: dict = {"radius": 100},
    return_unique: bool = False,
):
    """Returns a list of indices of earthquakes that are neighboring the the specified linestring geometry."""

    queries = np.array([boundary.xy[1], boundary.xy[0]]).T
    points = np.array([catalog.lat.to_numpy(), catalog.lon.to_numpy()]).T

    indices = fun(points, queries, **kwarg)

    if return_unique:
        indices = np.unique(np.concatenate(indices))

    return indices
