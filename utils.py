# %%
from __future__ import annotations
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
import cartopy
import cartopy.crs as ccrs
from sklearn.cluster import KMeans
from obspy.clients.fdsn import Client
from typing import Optional, Literal, Union, Tuple
from pathlib import Path
import warnings
import os
import copy 

EARTH_RADIUS_KM = 6371

def get_xyz_from_lonlat(
    lon: np.ndarray, lat: np.ndarray, depth_km: Optional[np.ndarray] = None
) -> np.ndarray:
    """Converts longitude, latitude, and depth to x, y, and z Cartesian
    coordinates.

    Args:
        lon: The longitude, in degrees.
        lat: The latitude, in degrees.
        depth_km: The depth, in kilometers.

    Returns:
        The Cartesian coordinates (x, y, z), in kilometers.
    """
    # Check the shapes of the input arrays
    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape")

    assert -180 <= lon.all() <= 180, "Longitude must be between -180 and 180"
    assert -90 <= lat.all() <= 90, "Latitude must be between -90 and 90"
    assert depth_km is None or depth_km.all() >= 0, "Depth must be positive"

    # Assign zero depth if not provided:
    if depth_km is None:
        depth_km = np.zeros_like(lat)

    # Convert to radians
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180

    # Calculate the distance from the center of the earth using the depth
    # and the radius of the earth (6371 km)
    r = EARTH_RADIUS_KM - depth_km

    # Calculate the x, y, z coordinates
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.array([x, y, z]).T

class Catalog:
    def __init__(
        self,
        catalog: pd.DataFrame,
        mag_completeness: Optional[float] = None,
        units: Optional[dict] = None,
    ):
        self.raw_catalog = (
            catalog.copy()
        )  # Save a copy of the raw catalog in case of regret

        _catalog = catalog.copy()
        self.catalog: pd.DataFrame = _catalog
        self.__mag_completeness = mag_completeness
        self.__mag_completeness_method = None

        self.units = {k: None for k in self.catalog.keys()}
        if units is not None:
            assert set(units.keys()).issubset(
                set(self.catalog.keys())
            ), "Invalid keys in units"
            self.units.update(units)

        self.__update__()

    def __update__(self):
        self.catalog = self.catalog.sort_values(by="time")

        # Save catalog attributes to self
        self.start_time = self.catalog["time"].min()
        self.end_time = self.catalog["time"].max()
        self.duration = self.end_time - self.start_time

        if "lat" in self.catalog.keys() and "lon" in self.catalog.keys():
            self.latitude_range = (self.catalog["lat"].min(), self.catalog["lat"].max())
            self.longitude_range = (
                self.catalog["lon"].min(),
                self.catalog["lon"].max(),
            )

        assert "time" in self.catalog.keys() is not None, "No time column"
        assert "mag" in self.catalog.keys() is not None, "No magnitude column"

        # check whether the catalog has locations (which is preferred)
        for key in ["lat", "lon", "depth"]:
            if key not in self.catalog.keys():
                warnings.warn(
                    f"Catalog does not have {key} column, this may cause errors."
                )

    @property
    def mag_completeness(
        self,
        magnitude_key: str = "mag",
        method: Literal["minimum", "maximum curvature"] = "minimum",
        filter_catalog: bool = True,
    ):
        if (
            self.__mag_completeness is None
            or self.__mag_completeness_method is not method
        ) and magnitude_key in self.catalog.keys():
            f = {
                "minimum": lambda M: min(M),
                "maximum curvature": lambda M: np.histogram(M)[1][
                    np.argmax(np.histogram(M)[0])
                ]
                + 0.2,
            }
            self.__mag_completeness = f[method](self.catalog[magnitude_key])
            self.__mag_completeness_method = method

            if filter_catalog:
                self.catalog = self.catalog[self.catalog.mag >= self.__mag_completeness]

        return self.__mag_completeness

    @mag_completeness.setter
    def mag_completeness(self, value):
        self.catalog = self.catalog[self.catalog.mag >= value]
        self.__mag_completeness = value

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, index: int) -> pd.Series:
        return self.catalog.iloc[index]

    def __getslice__(
        self,
        start: int,
        stop: int,
        step: Optional[int] = None,
    ) -> Catalog:
        new = copy.deepcopy(self)
        new.catalog = self.catalog[start:stop:step]
        new.__update__()

        return new

    def __iter__(self):
        return self.catalog.iterrows()

    def __add__(self, other) -> Catalog:
        combined_catalog = pd.concat(
            [self.catalog, other.catalog], ignore_index=True, sort=False
        )
        new = copy.deepcopy(self)
        new.catalog = combined_catalog
        new.__update__()

        return new

    def __radd__(self, other):
        return self.__add__(other)

    def slice_by(
        self,
        col_name: str,
        start=None,
        stop=None,
    ) -> Catalog:
        if start is None:
            start = self.catalog[col_name].min()
        if stop is None:
            stop = self.catalog[col_name].max()

        assert start <= stop
        in_range = (self.catalog[col_name] >= start) & (self.catalog[col_name] <= stop)

        new = copy.deepcopy(self)
        new.catalog = self.catalog.loc[in_range]
        new.__update__()

        return new

    def get_time_slice(self, start_time, end_time):
        return self.slice_by("time", start_time, end_time)

    def get_space_slice(self, latitude_range, longitude_range):
        return self.slice_by("lat", *latitude_range).slice_by("lon", *longitude_range)

    def get_polygon_slice(
        self,
        polygonal_boundary: np.ndarray,
    ) -> Catalog:
        polygon = shapely.geometry.Polygon(polygonal_boundary)

        new = copy.deepcopy(self)
        new.catalog = new.catalog[
            new.catalog.apply(
                lambda row: polygon.contains(shapely.geometry.Point(row.lon, row.lat)),
                axis=1,
            )
        ]

        return new

    def filter_duplicates(
        self,
        buffer_radius_km: float = 100,
        buffer_time_days: float = 30,
        stategy: Literal[
            "keep first", "keep last", "keep largest", "referece"
        ] = "keep first",
        ref_preference=None,
    ) -> Catalog:
        """returns a new catalog with duplicate events removed

        Checks for duplicates within `buffer_radius_km` and `buffer_time_seconds` of each other.

        """

        self.catalog.reset_index(drop=True, inplace=True)
        indices = self.intersection(
            self,
            buffer_radius_km=buffer_radius_km,
            buffer_time_days=buffer_time_days,
            return_indices=True,
        )[1]

        if ref_preference is not None:
            ranking = {i_ref: i for i, i_ref in enumerate(ref_preference)}

        indices_to_drop = []
        for i, neighbors in enumerate(indices):
            # Expects that all events in the catalog will have at least themselves as a neighbor
            if (
                len(neighbors) > 1
                and len(self.catalog.iloc[neighbors].ref.unique()) > 1
            ):  # don't drop if all events are the same reference
                if stategy == "keep first":
                    keep = self.catalog.iloc[neighbors].time.argmin()

                if stategy == "keep last":
                    keep = self.catalog.iloc[neighbors].time.argmax()

                if stategy == "keep largest":
                    keep = self.catalog.iloc[neighbors].mag.argmax()

                if stategy == "reference":
                    if ref_preference is None:
                        raise ValueError("Must specify reference event")
                    references = self.catalog.ref.iloc[neighbors]
                    rank = np.array([ranking[ref] for ref in references])
                    keep = rank.argmin()
                indices_to_drop.append(np.delete(neighbors, keep))

        new = copy.deepcopy(self)
        if len(indices_to_drop) > 0:
            indices_to_drop = np.unique(np.concatenate(indices_to_drop))
            new.catalog = new.catalog.drop(indices_to_drop)

        return new

    def intersection(self, other: Union[Catalog, list], buffer_radius_km: float = 50.0) -> Catalog:
        """returns a new catalog with the events within `buffer_radius_km` of the events in `other`.
        
        Other can either be a another Catalog or a list with with lat and lon"""

        indices = np.unique(
            np.concatenate(self.get_intersecting_indices(other, buffer_radius_km))
        )

        return Catalog(self.catalog.iloc[indices])

    def get_intersecting_indices(
        self, other: Union[Catalog, list], buffer_radius_km: float = 50.0
    ) -> np.ndarray:
        """gets the indices of events in `self` that are within `buffer_radius_km` from other.

        The ouput therefore has dimensions [len(other),k] where k is the number of neibors for each event.

        For instance:

        ```
        [self[indices] for indices in self.get_neighboring_indices(other)]
        ```

        Returns a list of events for each neighborhood of events in other."""
        
        if isinstance(other,Catalog):
            R = [other.catalog.lat.values, other.catalog.lon.values]
        elif isinstance(other,list):
            R = other
        
        tree = BallTree(
            np.deg2rad([self.catalog.lat.values, self.catalog.lon.values]).T,
            metric="haversine",
        )

        indices = tree.query_radius(
            np.deg2rad(R).T,
            r=buffer_radius_km / EARTH_RADIUS_KM,
            return_distance=False,
        )
        
        return indices

    def get_neighboring_indices(
        self, other: Union[Catalog, list], buffer_radius_km: float = 50.0
    ) -> np.ndarray:
        """gets the indices of events in `other` that are within `buffer_radius_km` from self.

        The ouput therefore has dimensions [len(self),k] where k is the number of neibors for each event.

        For instance:

        ```
        [other[indices] for indices in self.get_neighboring_indices(other)]
        ```

        Returns a list of events for each neighborhood of events in self."""
        
        if isinstance(other,Catalog):
            R = [other.catalog.lat.values, other.catalog.lon.values]
        elif isinstance(other,list):
            R = other

        tree = BallTree(
            np.deg2rad(R).T,
            metric="haversine",
        )

        return tree.query_radius(
            np.deg2rad(self.catalog[["lat", "lon"]]).values.T,
            r=buffer_radius_km / EARTH_RADIUS_KM,
            return_distance=False,
        )

    def get_clusters(
        self,
        column: Union[list, str],
        number_of_clusters: int,
    ) -> list[Catalog]:
        if isinstance(column, str):
            assert column in self.catalog.columns
            X = np.atleast_2d(self.catalog[column].values).T
        elif isinstance(column, list):
            for col in column:
                assert col in self.catalog.columns
            X = self.catalog[column].values
        kmeans = KMeans(
            n_clusters=number_of_clusters,
        ).fit(X)

        subcatalogs = []
        for i in range(number_of_clusters):
            new = copy.deepcopy(self)
            new.catalog = self.catalog.loc[kmeans.labels_ == i]
            new.__update__()
            subcatalogs.append(new)

        idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
        subcatalogs = [subcatalogs[i] for i in idx]

        return subcatalogs

    def plot_time_series(
        self, column: str = "mag", type="scatter", ax=None
    ) -> plt.axes.Axes:
        """
        Plots a time series of a given column in a dataframe.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if column == "mag" and self.mag_completeness is not None:
            bottom = self.mag_completeness - 0.05
        else:
            bottom = 0

        if type == "scatter":
            markers, stems, _ = ax.stem(
                self.catalog["time"],
                self.catalog[column],
                markerfmt=".",
                bottom=bottom,
            )
            plt.setp(stems, linewidth=0.5, alpha=0.5)
            plt.setp(markers, markersize=0.5, alpha=0.5)

        elif type == "hist":
            ax.hist(self.catalog["time"], bins=500)
            ax.set_yscale("log")

        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        axb = ax.twinx()
        sns.ecdfplot(self.catalog["time"], c="C1", stat="count", ax=axb)

        return ax

    def plot_space_time_series(
        self,
        p1: list[float, float] = None,  # lon, lat
        p2: list[float, float] = None,  # lon, lat
        column: str = "mag",
        k_largest_events: Optional[int] = None,
        plot_histogram: bool = True,
        kwargs: dict = None,
        ax: Optional[plt.axes.Axes] = None,
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if p1 is None and p2 is None:
            p1 = np.array([self.longitude_range[0], self.latitude_range[0]])
            p2 = np.array([self.longitude_range[1], self.latitude_range[1]])

        default_kwargs = {
            "alpha": 0.5,
            "color": "C0",
        }

        if kwargs is None:
            kwargs = {}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        p1, p2, x = [
            get_xyz_from_lonlat(np.atleast_2d(ll)[:, 0], np.atleast_2d(ll)[:, 1])
            for ll in [p1, p2, self.catalog[["lon", "lat"]].values]
        ]

        distance_along_section = np.matmul((x - p1), (p2 - p1).T) / np.linalg.norm(
            p2 - p1
        )

        marker_size = getattr(self.catalog, column) if isinstance(column, str) else 1

        ax.scatter(
            self.catalog.time,
            distance_along_section,
            **kwargs,
            s=marker_size,
        )

        if k_largest_events is not None:
            index_of_largest_events = np.argsort(self.catalog[column].values)[-k_largest_events:]
            ax.scatter(
                self.catalog.time.values[index_of_largest_events],
                distance_along_section[index_of_largest_events],
                **dict(kwargs, marker="*", s=60),
            )

        if plot_histogram is True:
            # horizonta histogram of distance along section on the right side of the plot pointing left
            axb = ax.twiny()
            axb.hist(
                distance_along_section,
                orientation="horizontal",
                density=True,
                alpha=0.3,
            )

            axb.set(
                xlim=np.array(axb.get_xlim()[::-1]) * 10,
                xticks=[],
            )

        return ax

    def plot_depth_cross_section(
        self,
        p1: list[float, float] = None,
        p2: list[float, float] = None,
        width_km: float = None,
        column: str = "mag",
        k_largest_events: Optional[int] = None,
        plot_histogram: bool = True,
        kwargs: dict = None,
        ax: Optional[plt.axes.Axes] = None,
    ) -> plt.axes.Axes:
        for column_name in ["lon", "lat", "depth", column]:
            assert (
                column_name in self.catalog.columns
            ), f"column {column_name} not in catalog"  # TODO: make this assertion as part of the catalog class itself?

        if ax is None:
            fig, ax = plt.subplots()

        if p1 is None and p2 is None:
            p1 = np.array([self.longitude_range[0], self.latitude_range[0]])
            p2 = np.array([self.longitude_range[1], self.latitude_range[1]])

        default_kwargs = {
            "alpha": 0.5,
            "c": "C0",
            "s": getattr(self.catalog, column) if column in self.catalog.keys() else 1,
        }

        if kwargs is None:
            kwargs = {}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        p1, p2, x = [
            get_xyz_from_lonlat(np.atleast_2d(ll)[:, 0], np.atleast_2d(ll)[:, 1])
            for ll in [p1, p2, self.catalog[["lon", "lat"]].values]
        ]

        distance_along_section = np.squeeze(
            np.matmul((x - p1), (p2 - p1).T) / np.linalg.norm(p2 - p1)
        )

        depth = self.catalog.depth.values

        if width_km is not None:
            distance_orthogonal_to_section = np.sqrt(
                np.linalg.norm(x - p1, axis=1) ** 2 - distance_along_section**2
            )
            index = distance_orthogonal_to_section < width_km
            if np.sum(index) == 0:
                warnings.warn(
                    "No data in the specified area, consider increasing width_km or checking if lat lon are correct"
                )
            distance_along_section = distance_along_section[index]
            depth = depth[index]
            if column in self.catalog.keys():
                default_kwargs["s"] = default_kwargs["s"][index]
            mag = self.catalog.mag[index].values

        sh = ax.scatter(
            distance_along_section,
            depth,
            **kwargs,
        )

        if k_largest_events is not None:
            index_of_largest_events = np.argsort(mag)[-k_largest_events:]
            ax.scatter(
                distance_along_section[index_of_largest_events],
                depth[index_of_largest_events],
                **dict(kwargs, marker="*", s=60),
            )

        ax.set(
            ylabel="Depth (km)",
            xlabel="Distance along section (km)",
            ylim=(np.max(depth), 0),
        )

        if plot_histogram is True:
            # horizonta histogram of distance along section on the right side of the plot pointing left
            axb = ax.twiny()
            axb.hist(
                depth,
                orientation="horizontal",
                density=True,
                alpha=0.3,
                color=sh.get_facecolor(),
            )

            axb.set(
                xlim=np.array(axb.get_xlim()[::-1]) * 10,
                xticks=[],
            )

        return ax

    def plot_base_map(
        self,
        extent: Optional[np.ndarray] = None,
        ax=None,
    ) -> plt.axes.Axes:
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

        usemap_proj = ccrs.PlateCarree()
        # set appropriate extents: (lon_min, lon_max, lat_min, lat_max)
        if extent is None:
            buffer = 1
            if self.longitude_range is None or self.latitude_range is None:
                extent = (
                    np.array(
                        [
                            self.catalog["lon"].min(),
                            self.catalog["lon"].max(),
                            self.catalog["lat"].min(),
                            self.catalog["lat"].max(),
                        ]
                    )
                    + np.array([-1, 1, -1, 1]) * buffer
                )

            else:
                extent = (
                    np.array(self.longitude_range + self.latitude_range)
                    + np.array([-1, 1, -1, 1]) * buffer
                )

        if extent[0] < -180:
            extent[0] = -179.99
        if extent[1] > 180:
            extent[1] = 179.99

        if extent[2] < -90:
            extent[2] = 90
        if extent[3] > 90:
            extent[3] = 90

        ax.set_extent(
            extent,
            crs=ccrs.PlateCarree(),
        )

        ax.add_feature(cartopy.feature.LAND, color="lightgray")
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")

        # plot grid lines
        ax.gridlines(draw_labels=True, crs=usemap_proj, color="gray", linewidth=0.3)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        return ax

    def plot_map(
        self,
        column: str = "mag",
        scatter_kwarg: Optional[dict] = None,
        k_largest_events: Optional[int] = None,
        extent: Optional[np.ndarray] = None,
        ax=None,
    ) -> plt.axes.Axes:
        ax = self.plot_base_map(extent=extent, ax=ax)

        if scatter_kwarg is None:
            scatter_kwarg = {}
        default_scatter_kawrg = {
            "s": self.catalog[column],
            "c": "lightgray",
            "marker": "o",
            "edgecolors": "brown",
            "transform": ccrs.PlateCarree(),
        }
        default_scatter_kawrg.update(scatter_kwarg)

        ax.scatter(
            self.catalog["lon"],
            self.catalog["lat"],
            **default_scatter_kawrg,
        )

        if k_largest_events is not None:
            index_of_largest_events = np.argsort(self.catalog[column].to_numpy())[-k_largest_events:]
            ax.scatter(
                self.catalog["lon"].values[index_of_largest_events],
                self.catalog["lat"].values[index_of_largest_events],
                **dict(scatter_kwarg, marker="*", s=60),
            )

        return ax

    def plot_hist(
        self, columm: str = "mag", log_scale: bool = True, ax=None
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.catalog[columm], log=log_scale)
        ax.set_xlabel(columm)

        return ax

    def plot_scaling(
        self, column: str = "duration", log_scale: bool = True, ax=None
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        assert self.catalog[column] is not None, "No duration column"
        ax.scatter(self.catalog["mag"], self.catalog[column])
        ax.set(
            xlabel="Magnitude",
            ylabel=column,
        )
        if log_scale:
            ax.set_yscale("log")

        return ax

    def plot_summary(
        self, kwarg={"time series": None, "map": None, "hist": None}, ax=None
    ) -> Tuple[plt.axes.Axes, plt.axes.Axes, plt.axes.Axes, plt.axes.Axes]:
        if ax is None:
            fig = plt.figure(figsize=(6.5, 7))
            gs = fig.add_gridspec(4, 3)
            ax1 = fig.add_subplot(gs[0:2, 0:2], projection=ccrs.PlateCarree())
            ax2 = fig.add_subplot(gs[0:2, 2])
            ax3 = fig.add_subplot(gs[2, :])
            ax4 = fig.add_subplot(gs[3, :])
        else:
            print("Bold decision")
            ax1, ax2, ax3, ax4 = ax

        self.plot_map(ax=ax1)
        self.plot_hist(ax=ax2)
        self.plot_time_series(ax=ax3)
        self.plot_space_time_series(ax=ax4)

        plt.tight_layout()

        return (ax1, ax2, ax3, ax4)

    def read_catalog(self, filename):
        """
        Reads a catalog from a file and returns a dataframe.
        """
        raise NotImplementedError
    
    def to_csv(self,filename: str, columns: list = ["time", "mag", "lat", "lon"]):
        """Writes catalog as csv 

        Args:
            filename: path and filename to save the csv file
            columns: columns of the catalog data that should be saved to the csv file 
        """
        self.catalog.to_csv(filename, columns=columns, index=False)



class EarthquakeCatalog(Catalog):
    def __init__(
        self,
        filename: str = None,
        kwargs: dict = None,
        use_other_catalog: bool = False,
        other_catalog: Catalog = None,
    ) -> Catalog:
        if kwargs is None:
            kwargs = {}

        if use_other_catalog and other_catalog is not None:
            metadata = {
                "starttime": other_catalog.start_time,
                "endtime": other_catalog.end_time,
                "latitude_range": other_catalog.latitude_range,
                "longitude_range": other_catalog.longitude_range,
            }
            metadata.update(kwargs)
        elif not use_other_catalog:
            metadata = kwargs
        else:
            raise ValueError("No other catalog provided")

        _catalog = self.get_and_save_catalog(filename, **metadata)
        self.catalog = self._add_time_column(_catalog, "time")

        super().__init__(self.catalog)

    @staticmethod
    def _add_time_column(df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(pd.to_datetime(df["time"], unit="d"))
        return df

    @staticmethod
    def get_and_save_catalog(
        filename: Union[str, Path] = "_temp_local_catalog.csv",
        starttime: str = "2019-01-01",
        endtime: str = "2020-01-01",
        latitude_range: list[float] = [-90, 90],
        longitude_range: list[float] = [-180, 180],
        minimum_magnitude: float = 4.5,
        use_local_client: bool = False,
        default_client_name: str = "IRIS",
        reload: bool = True,
    ) -> pd.DataFrame:
        """
        Gets earthquake catalog for the specified region and minimum event
        magnitude and writes the catalog to a file.

        By default, events are retrieved from the NEIC PDE catalog for recent
        events and then the ISC catalog when it becomes available. These default
        results include only that catalog's "primary origin" and
        "primary magnitude" for each event.
        """

        if longitude_range[1] > 180:
            longitude_range[1] = 180
            warnings.warn("Longitude range exceeds 180 degrees. Setting to 180.")

        if longitude_range[0] < -180:
            longitude_range[0] = -180
            warnings.warn("Longitude range exceeds -180 degrees. Setting to -180.")

        if latitude_range[1] > 90:
            latitude_range[1] = 90
            warnings.warn("Latitude range exceeds 90 degrees. Setting to 90.")

        if latitude_range[0] < -90:
            latitude_range[0] = -90
            warnings.warn("Latitude range exceeds -90 degrees. Setting to -90.")

        def is_within(lat_range_querry, lon_range_querry, lat_range, lon_range):
            """
            Checks if a point is within a latitude and longitude range.
            """
            return (
                (lat_range[0] <= lat_range_querry[0] <= lat_range[1])
                and (lon_range[0] <= lon_range_querry[0] <= lon_range[1])
                and (lat_range[0] <= lat_range_querry[1] <= lat_range[1])
                and (lon_range[0] <= lon_range_querry[1] <= lon_range[1])
            )

        local_client_coverage = {
            "GEONET": [[-49.18, -32.28], [163.52, 179.99]],
        }

        # Note that using local client supersedes the any specified default_client_name
        if use_local_client:
            ## use local clients if lat and long are withing the coverage of the local catalogs
            index = []
            for i, key in enumerate(local_client_coverage.keys()):
                if is_within(
                    latitude_range, longitude_range, *local_client_coverage[key]
                ):
                    index.append(i)
                else:
                    index.append(i)

            if len(index) > 1:
                raise ValueError("Multiple local clients found")
            elif len(index) == 1:
                if default_client_name is not None:
                    warnings.warn("Using local client instead of default client")
                client_name = list(local_client_coverage.keys())[index[0]]
            else:
                client_name = default_client_name
        else:
            client_name = default_client_name

        querry = dict(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=minimum_magnitude,
            minlatitude=latitude_range[0],
            maxlatitude=latitude_range[1],
            minlongitude=longitude_range[0],
            maxlongitude=longitude_range[1],
        )

        if not (
            reload is False
            and os.path.exists(filename)
            and np.load(
                os.path.splitext(filename)[0] + "_metadata.npy", allow_pickle=True
            ).item()
            == querry
        ):
            warnings.warn(f"Reloading {filename}")

            # Use obspy api to ge  events from the IRIS earthquake client
            client = Client(client_name)
            cat = client.get_events(**querry)

            # Write the earthquakes to a file
            f = open(filename, "w")
            f.write("time,lat,lon,depth,mag\n")
            for event in cat:
                loc = event.preferred_origin()
                lat = loc.latitude
                lon = loc.longitude
                dep = loc.depth
                time = loc.time.matplotlib_date
                mag = event.preferred_magnitude().mag
                f.write("{},{},{},{},{}\n".format(time, lat, lon, dep, mag))
            f.close()

            # Save querry to metadatafile
            np.save(os.path.splitext(filename)[0] + "_metadata.npy", querry)
        else:
            warnings.warn(f"Using existing {filename}")

        df = pd.read_csv(filename, na_values="None")

        # remove rows with NaN values, reset index and provide a warning is any rows were removed
        if df.isna().values.any():
            warnings.warn(
                f"{sum(sum(df.isna().values))} NaN values found in catalog. Removing rows with NaN values."
            )
            df = df.dropna()
            df = df.reset_index(drop=True)

        df.depth = df.depth / 1000  # convert depth from m to km

        return df
