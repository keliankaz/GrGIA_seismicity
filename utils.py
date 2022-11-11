import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature


from obspy.clients.fdsn import (
    Client,
)  # obspy is a well maintained toolbox for processing seismological data
from obspy import UTCDateTime  # https://docs.obspy.org/
import typing

class Catalog:
    def __init__(self, catalog: pd.DataFrame):
        _catalog = catalog.copy()
        _catalog = _catalog.sort_values(by="time")
        self.catalog = _catalog

        # Save catalog attributes to self
        self.start_time = self.catalog["time"].min()
        self.end_time = self.catalog["time"].max()
        self.duration = self.end_time - self.start_time

        self.latitude_range = (
            self.catalog["lat"].min(),
            self.catalog["lat"].max()
        )

        self.longitude_range = (
            self.catalog["lon"].min(),
            self.catalog["lon"].max()
        )

        assert self.catalog["time"] is not None, "No time column"
        assert self.catalog["mag"] is not None, "No magnitude column"

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, index):
        return self.catalog[index]

    def __getslice__(self, start, stop, step=1):
        return self.catalog[start:stop:step]
    
    def __iter__(self):
        return self.catalog.iterrows()

    def __add__(self, other):
        combined_catalog = pd.concat([self.catalog, other.catalog],ignore_index=True,sort=False)
        return Catalog(combined_catalog)

    def plot_time_series(self, column: str = "mag", ax=None):
        """
        Plots a time series of a given column in a dataframe.
        """
        if ax is None: 
            fig,ax = plt.subplots()

        ax.stem(self.catalog["time"], self.catalog[column])
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        axb = ax.twinx()
        axb.plot(self.catalog["time"], range(len(self.catalog["time"])), "r")

    def plot_map(self, columm: str = "mag", ax=None):

        if ax is None: 
            fig,ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

        usemap_proj = ccrs.PlateCarree()   
        # set appropriate extents: (lon_min, lon_max, lat_min, lat_max)
        buffer = 1
        ax.set_extent(
            np.array(self.longitude_range + self.latitude_range)+np.array([-1,1,-1,1])*buffer, 
            crs=ccrs.PlateCarree()
        )

        ax.add_feature(cfeature.LAND, color='lightgray')
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # plot grid lines
        ax.gridlines(draw_labels=True, crs=usemap_proj, color='gray', linewidth=0.3)

        ax.scatter(
            self.catalog["lon"],
            self.catalog["lat"],
            s=self.catalog[columm],
            c='lightgray',
            marker='o',
            edgecolors='brown',
            transform=ccrs.PlateCarree(),
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    def plot_hist(self, columm: str = "mag", log_scale: bool = True, ax=None):
        if ax is None: 
            fig,ax = plt.subplots()
        ax.hist(self.catalog[columm], log=log_scale)
        ax.set_xlabel(columm)

    def plot_scaling(self, column: str = "duration", log_scale: bool = True, ax=None):
        if ax is None: 
            ax = plt.subplots()
        assert self.catalog[column] is not None, "No duration column"
        ax.scatter(self.catalog["mag"], self.catalog[column])
        ax.xlabel("Magnitude")
        ax.ylabel(column)
        if log_scale:
            ax.yscale("log")

    @staticmethod
    def read_catalog(self, filename):
        """
        Reads a catalog from a file and returns a dataframe.
        """
        raise NotImplementedError

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
                "starttime":  other_catalog.start_time,
                "endtime": other_catalog.end_time,
                "latitude_range": other_catalog.latitude_range,
                "longitude_range": other_catalog.longitude_range,
            }
            metadata.update(kwargs)
        elif not use_other_catalog:
            metadata = kwargs
        else :
            raise ValueError("No other catalog provided")
    
        _catalog = self.get_and_save_catalog(filename,**metadata)
        self.catalog = self._add_time_column(_catalog, "time")

        super().__init__(self.catalog)

    @staticmethod
    def _add_time_column(df, column):
        """
        Adds a column to a dataframe with the time in days since the beginning of the year.
        """
        df[column] = pd.to_datetime(pd.to_datetime(df['time'],unit='d'))
        return df

    @staticmethod
    def get_and_save_catalog(
        filename: str = "_temp_local_catalog.csv",
        starttime: str = "2019-01-01",
        endtime: str = "2020-01-01",
        latitude_range: list[float, float] = [-90, 90],
        longitude_range: list[float, float] = [-180, 180],
        minimum_magnitude: float = 4.5,
    ) -> pd.DataFrame:
        """
        Gets earthquake catalog for the specified region and minimum event
        magnitude and writes the catalog to a file.

        By default, events are retrieved from the NEIC PDE catalog for recent
        events and then the ISC catalog when it becomes available. These default
        results include only that catalog's "primary origin" and
        "primary magnitude" for each event.
        """

        # Use obspy api to ge  events from the IRIS earthquake client
        client = Client("IRIS")
        cat = client.get_events(
            starttime=starttime,
            endtime=endtime,
            magnitudetype="MW",
            minmagnitude=minimum_magnitude,
            minlatitude=latitude_range[0],
            maxlatitude=latitude_range[1],
            minlongitude=longitude_range[0],
            maxlongitude=longitude_range[1],
        ) 

        # Write the earthquakes to a file
        f = open(filename, "w")
        f.write("EVENT_ID,time,lat,lon,dep,mag\n")
        for event in cat:
            longID = event.resource_id.id
            ID = longID.split("eventid=", 1)[1]
            loc = event.preferred_origin()
            lat = loc.latitude
            lon = loc.longitude
            dep = loc.depth
            time = loc.time.matplotlib_date
            mag = event.preferred_magnitude().mag
            f.write("{}, {}, {}, {}, {}, {}\n".format(ID, time, lat, lon, dep, mag))
        f.close()
        df = pd.read_csv(filename)

        return df
