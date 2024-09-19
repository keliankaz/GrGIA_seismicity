# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import scipy.io
import geopandas as gpd
import cartopy
import shapely
import shapely.geometry
from sklearn.neighbors import BallTree
from utils import EarthquakeCatalog, Catalog
from geo_utils import densify_geometry
from typing import Literal, Optional


class SpaceTimeGrid:
    def __init__(self, fields: dict, coordinates: dict, times: np.ndarray):
        self.fields = fields
        self.coordinates = coordinates
        self._base_coordinate = next(iter(coordinates.values()))
        self._base_coordinate_name = next(iter(coordinates.keys()))
        self.times = times

        self.validate_args()

    def validate_args(self):
        return NotImplementedError

    def __add__(self, other):
        assert self.coordinates == other.coordinates
        assert np.all(self.times == other.times)

        new = copy.deepcopy(self)

        for k1 in new.fields.keys():
            new.fields[k1] += other.fields[k1]

        return new

    def time_stack(self, field):
        return self.fields[field].mean(axis=0)

    def space_stack(self, field):
        return self.fields[field].mean(axis=0)

    def plot_grid(self, field, coordinate=None, ax=None, imshowkwargs=None):

        if ax is None:
            fig, ax = plt.subplots()

        kwargs = dict(
            origin="lower",
            extent=[
                min(self.times),
                max(self.times),
                min(
                    self._base_coordinate
                    if not coordinate
                    else self.coordinates[coordinate]
                ),
                max(
                    self._base_coordinate
                    if not coordinate
                    else self.coordinates[coordinate]
                ),
            ],
        )

        if imshowkwargs is not None:
            kwargs.update(imshowkwargs)

        ax.imshow(self.fields[field], **kwargs)

        ax.set(
            xlabel="Time",
            ylabel=self._base_coordinate_name if not coordinate else coordinate,
        )

        return ax


class PlateBoundaryRateGrid(SpaceTimeGrid):
    def __init__(
        self,
        earthquakes: Catalog,
        NUMBER_OF_PERIODS=28,  # number of periods to split the catalog into
        WINDOW_SIZE=100,  # km
        NUMBER_OF_EVENTS=200,
        starttime=None,
        endtime=None,
    ):

        # date_range = pd.date_range(
        #     start=datetime.datetime(GrGIA_strain_metadata["starttime"], 1, 1),
        #     end=datetime.datetime(GrGIA_strain_metadata["endtime"], 1, 1),
        #     periods=NUMBER_OF_PERIODS,
        # )

        return NotImplementedError


class PlateBoundary:

    def __init__(
        self,
        filename=None,
        bounding_box=[-180, -90, 180, 90],
        boundary_names=None,
        mapping_crs=cartopy.crs.PlateCarree(),
    ):
        plate_boundaries = gpd.read_file(filename)
        self._raw_data = plate_boundaries
        self.bounds = bounding_box
        self.mapping_crs = mapping_crs

        # get the north selected plate boundary:
        if boundary_names:
            plate_boundaries = plate_boundaries.iloc[
                np.logical_or.reduce(
                    [plate_boundaries.Name == boundary for boundary in boundary_names]
                )
            ]

        plate_boundaries = plate_boundaries.clip(bounding_box).explode(index_parts=True)

        self.geometries = plate_boundaries

    def plot_basemap(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": self.mapping_crs})

        ax.coastlines()

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        return ax

    def plot(self, ax=None):
        ax = self.plot_basemap(ax=ax)
        self.geometries.geometry.to_crs(self.mapping_crs).plot(ax=ax, color="maroon")

        return ax

    def get_earthquake_catalog(
        self,
        earthquakes=None,
        filename="global_M4.csv",
        query={
            "minimum_magnitude": 4,
            "starttime": "1960-01-01",
            "endtime": "2022-01-01",
        },
        buffer_km=100,
    ):
        """Gets the plate boundary catalog within a distance buffer_km form the surface trace of the plate boundary."""
        if earthquakes is None:
            earthquakes = EarthquakeCatalog(
                filename=filename,
                kwargs=query,
            )

        lonlat = self.geometries.get_coordinates().values

        return earthquakes.intersection(
            [lonlat[:, 1], lonlat[:, 0]], buffer_radius_km=buffer_km
        )


class MidAtlanticRidge(PlateBoundary):

    def __init__(
        self,
        filename,  # shapefile
        stepsize=50000,
        bounding_box=[-180, 55, 10, 90],
        boundary_names=["NA-EU", "EU-NA"],
        geographic_crs=cartopy.crs.NorthPolarStereo(),
        mapping_crs=cartopy.crs.NorthPolarStereo(),
        exclude_iceland=True,
        iceland_bounds=[(-25, 52), (-25, 68), (-13, 68), (-13, 52), (-25, 52)],
    ):
        super().__init__(
            filename,
            bounding_box=bounding_box,
            boundary_names=boundary_names,
            mapping_crs=cartopy.crs.NorthPolarStereo(),
        )

        if exclude_iceland:
            # remove iceland from the plate boundary
            iceland_bounds = gpd.GeoSeries([shapely.geometry.Polygon(iceland_bounds)])
            iceland_bounds_df = gpd.GeoDataFrame(
                {"geometry": iceland_bounds}, crs="EPSG:4326"
            )
            self.geometries = gpd.overlay(
                self.geometries, iceland_bounds_df, how="difference"
            ).explode(index_parts=True)

        self.mapping_crs = mapping_crs
        self.geographic_crs = geographic_crs  # this is used for converting to meters
        self.merged_geometry_meters = self.densify(step_size=stepsize)
        self.boundary_names = boundary_names

    def densify(self, step_size):

        # Entering Coordinate reference system hell: proceed with caution
        crs_proj4 = (
            self.geographic_crs.proj4_init
        )  # turns projection into string with projection description

        crs_in = crs_proj4
        crs_out = "EPSG:4326"

        # currently in "EPSG:4326"

        # convert to new crs that is in meters, this will enable the densifying step.
        boundary_in_meters = self.geometries.to_crs(crs_in)

        # now in crs_proj4

        # merge the boundary into a single line
        merged_boundary_in_meters = shapely.ops.linemerge(
            shapely.geometry.MultiLineString(boundary_in_meters.geometry.values)
        )

        # densify the boundary
        interp_merged_boundary = densify_geometry(
            merged_boundary_in_meters,
            step_size,
            crs_in=crs_in,
            crs_out=crs_out,
        ).geometry.values[0]

        # back into "EPSG:4326"

        return interp_merged_boundary


class GIA:
    def __init__(
        self,
        filename,
        data_key,
        longterm_filename,
        data_config,
        strain_positive_convention,
        strain_units,
    ):

        self.filename = filename
        self.longterm_filename = longterm_filename
        [setattr(self, k, v) for k, v in data_config.items()]
        self.strain_positive_convention = strain_positive_convention
        self.strain_units = strain_units

        # negative sign because the sign convention in the modelling
        # specified expansion as positive, but we want **contraction as positive**
        strain = (
            -1 if strain_positive_convention == "compression" else 1
        ) * scipy.io.loadmat(filename)[data_key]

        # idk man, life is hard (latitudes were specified in decreasing order)
        strain = np.flip(strain, 1)

        if longterm_filename:
            # negative sign because the sign convention in the modelling
            # specified expansion as compression, but we want **contraction as positive**
            lt = (
                -1 if strain_positive_convention == "compression" else 1
            ) * scipy.io.loadmat(longterm_filename)[data_key]

            # idk man, life is hard (latitudes were specified in decreasing order)
            lt = np.flip(lt, 1)

            strain += lt

        assert [
            data_config[k]
            for k in ["number_of_times", "number_of_latitudes", "number_of_longitudes"]
        ] + [3, 3] == list(
            strain.shape
        ), "strain data dimensions do not match metadata specification"

        strain_years = np.linspace(
            *[data_config[k] for k in ["starttime", "endtime", "number_of_times"]],
        )

        # Careful about tthe edges of the grid. I am following what seems to be done in the lat.mat and lon.mat files.
        strain_latitudes = np.linspace(
            *data_config["latitude_range"],
            data_config["number_of_latitudes"] + 1,
        )[1:]

        strain_longitudes = np.linspace(
            *data_config["longitude_range"],
            data_config["number_of_longitudes"] + 1,
        )[:-1]

        self.strain = strain
        self.time = strain_years
        self.lat = strain_latitudes
        self.lon = strain_longitudes

    def query(
        self,
        times: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        """Query strain data using nearest neighbor search"""

        time_grid, lat_grid, lon_grid = np.meshgrid(
            self.time, self.lat, self.lon, indexing="ij"
        )

        tree = BallTree(
            np.column_stack(
                [time_grid.flatten(), lat_grid.flatten(), lon_grid.flatten()]
            )
        )

        querried_indices = tree.query(
            np.column_stack(
                [times, latitudes, longitudes],
            ),
            k=1,
        )[1]

        return self.strain.reshape(-1, *self.strain.shape[-2:])[
            querried_indices, :, :
        ].squeeze()

    @staticmethod
    def get_normal_strain(epsilon, p1, p2):
        horizontal_strain = epsilon[:-1, :-1]
        rotation90 = np.array(
            [
                [0, -1],
                [1, 0],
            ]
        )
        delta_r = p2 - p1
        unit_normal = rotation90 @ delta_r / np.sqrt(np.sum(delta_r * delta_r))
        return (horizontal_strain @ unit_normal).T @ (unit_normal)


class GIAthumbnail(GIA):

    def __init__(
        self,
        strain_base_filename: str,
        coords_filename: str,
        long_term_strain_filename: Optional[str] = None, 
        data_config: dict = dict(
            number_of_latitudes=409,
            number_of_longitudes=149,
            number_of_years = 27,
            starttime=1990,
        ),
        strain_positive_convention: Literal["Compression", "Extension"] = "Compression",
    ):

        self.strain_base_filename = strain_base_filename
        self.coords_filename = coords_filename
        self.long_term_strain_filename = long_term_strain_filename
        [setattr(self, k, v) for k, v in data_config.items()]
        self.strain_positive_convention = strain_positive_convention

        strain = []
        print("Parsing raw data...")
        for i_year_count in np.arange(self.number_of_years)+1:
            print(1990+i_year_count)
            filename = self.strain_base_filename + (f'__0{i_year_count}' if i_year_count<10 else f'__{i_year_count}')
            strain.append(self.read_strain_file(filename))
    
        self.strain = [si - sip1 for si, sip1 in zip(strain[:-1], strain[1:])] # return to cummulative strain
            
        if self.long_term_strain_filename is not None:
            long_term_strain = self.read_strain_file(filename)
            self.strain = [s + long_term_strain for s in self.strain]    

        self.coordinates = (
            pd.read_csv(self.coords_filename, delim_whitespace=True)
            .values[-(self.number_of_latitudes * self.number_of_longitudes):,:]
            .reshape(
                (self.number_of_latitudes, self.number_of_longitudes, 3)
            )  # reshape into [space 1, space 2, [lat, lon, depth_m]]
        )
        
        self.time = np.arange(self.number_of_years-1) + self.starttime 


    def read_strain_file(self, filename) -> np.ndarray:
        
        # negative sign because the sign convention in the modelling
        # specified expansion as positive, but we want **contraction as positive**
        raw_strain = pd.read_csv(filename, delim_whitespace=True, names=["Exx","Exy","Exz","Eyy","Eyz","Ezz"])

        exx = raw_strain["Exx"].values
        exy = raw_strain["Exy"].values
        exz = raw_strain["Exz"].values
        eyy = raw_strain["Eyy"].values
        eyz = raw_strain["Eyz"].values
        ezz = raw_strain["Ezz"].values

        strain_tensor_flat = np.column_stack(
            [exx, exy, exz, exy, eyy, eyz, exz, eyz, ezz]
        )
        
        flat_strain_depth_slice = strain_tensor_flat[-self.number_of_latitudes * self.number_of_longitudes:,:]

        strain = 1e9 * (-1 if self.strain_positive_convention == "Compression" else 1) * (
            flat_strain_depth_slice.reshape(  # use only the shallowest layer
                (self.number_of_latitudes, self.number_of_longitudes, 3, 3)
            )  # reshape into [space 1, space 2, strain_tensor]
        )
        
        return strain
         

    def query(
        self,
        times: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        """Query strain data using nearest neighbor search"""

        tree = BallTree(
            np.column_stack(
                [
                    self.coordinates[:, :, 1].flatten(),
                    self.coordinates[:, :, 0].flatten(),
                ]
            )
        )

        querried_indices = tree.query(
            np.column_stack(
                [latitudes, longitudes],
            ),
            k=1,
        )[1]
        
        times_indices = np.clip(
            np.int16(np.round(times)-self.starttime),
            0, self.number_of_years-1-1 # -1 because of zero indexing and -1 for dt
        )
        
        return np.array([
            self.strain[time_index].reshape(-1, *self.strain[time_index].shape[-2:])[
                querried_index, :, :
            ].squeeze() 
            for time_index, querried_index in zip(times_indices, querried_indices)
        ])


if __name__ == '__main__':
    strain_filename = 'data/August24/strain_26'
    coords_filename = 'data/August24/local_coords.txt'
    gia = GIAthumbnail(strain_filename,coords_filename)
# %%

