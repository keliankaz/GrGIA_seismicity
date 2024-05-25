import numpy as np
import matplotlib.pyplot as plt
import copy
import geopandas as gpd
import cartopy
import shapely
import shapely.geometry
from utils import EarthquakeCatalog
from geo_utils import densify_geometry


class SpaceTimeGrid:
    def __init__(
        self,
        fields:dict,
        coordinates:dict,
        times:np.ndarray
    ):
        self.fields = fields
        self.coordinates = coordinates 
        self._base_coordinate = next(iter(coordinates.values()))
        self._base_coordinate_name = next(iter(coordinates.keys()))
        self.times = times
        
        self.validate_args()
        
    def validate_args(self):
        return NotImplementedError
    
    def __add__(self,other):
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
        
    def plot_grid(
        self,
        field,
        coordinate=None,
        ax=None,
        imshowkwargs=None
    ):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        kwargs = dict(
            origin='lower',
            extent=[
                min(self.times),
                max(self.times), 
                min(self._base_coordinate if not coordinate else self.coordinates[coordinate]), 
                max(self._base_coordinate if not coordinate else self.coordinates[coordinate]), 
            ]
        )
        
        if imshowkwargs is not None:
            kwargs.update(imshowkwargs)
        
        ax.imshow(self.fields[field], **kwargs)
        
        ax.set(
            xlabel='Time',
            ylabel= self._base_coordinate_name if not coordinate else coordinate,
        )
            
        return ax
    
class PlateBoundarySPGrid(SpaceTimeGrid):
    def __init__(
        self,
        fields:dict,
        coordinates:dict,
        times:np.ndarray
    ):
        super().__init__(fields,coordinates,times)
        
class PlateBoundary:
    
    def __init__(
        self,
        filename=None,
        bounding_box=[-180,-90,180,90],
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
                    [
                        plate_boundaries.Name == boundary
                        for boundary in boundary_names
                    ]
                )
            ]
            
        plate_boundaries = (
            plate_boundaries
            .clip(bounding_box)
            .explode(index_parts=True)
        )
        
        self.geometries = plate_boundaries
        
        
    
    def plot_basemap(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection":  self.mapping_crs})

        ax.coastlines()

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        return ax
    
    def plot(self, ax=None):
        ax = self.plot_basemap(ax=ax)
        self.geometries.geometry.to_crs(self.mapping_crs).plot(ax=ax,color='maroon')    
        
        return ax
        
    def get_earthquake_catalog(
        self,
        filename = "global_M4.csv",
        query = {
            "minimum_magnitude":4,
            "starttime": "1960-01-01",
            "endtime": "2022-01-01",
        },
        buffer_km = 100,  
    ):
        """Gets the plate boundary catalog within a distance buffer_km form the surface trace of the plate boundary."""
        earthquakes = EarthquakeCatalog(
            filename=filename,
            kwargs=query,
        )
        
        lonlat =  self.geometries.get_coordinates().values
        
        return earthquakes.intersection([lonlat[:,1],lonlat[:,0]], buffer_radius_km=buffer_km) 
                      
class MidAtlanticRidge(PlateBoundary):
    
    def __init__(
        self,
        filename, # shapefile
        stepsize = 50000,
        bounding_box = [-180, 55, 10, 90],
        boundary_names=["NA-EU", "EU-NA"],
        geographic_crs = cartopy.crs.NorthPolarStereo(),
        mapping_crs = cartopy.crs.NorthPolarStereo(),
        exclude_iceland = True,
        iceland_bounds = [(-25, 52), (-25, 68), (-13, 68), (-13, 52), (-25, 52)],
    ):
        super().__init__(
            filename,
            bounding_box=bounding_box,
            boundary_names=boundary_names,
            mapping_crs = cartopy.crs.NorthPolarStereo(),
        )
        
        if exclude_iceland:
            # remove iceland from the plate boundary
            iceland_bounds =gpd.GeoSeries([shapely.geometry.Polygon(iceland_bounds)])
            iceland_bounds_df = gpd.GeoDataFrame({'geometry': iceland_bounds}, crs="EPSG:4326")
            self.geometries = gpd.overlay(self.geometries, iceland_bounds_df, how='difference').explode(index_parts=True)

        self.mapping_crs = mapping_crs
        self.geographic_crs = geographic_crs # this is used for converting to meters
        self.merged_geometry_meters = self.densify(step_size=stepsize)
        self.boundary_names = boundary_names

    def densify(self, step_size):
        
        # Entering Coordinate reference system hell: proceed with caution
        crs_proj4 = self.geographic_crs.proj4_init # turns projection into string with projection description 
        
        crs_in = crs_proj4
        crs_out = "EPSG:4326"

        # currently in "EPSG:4326"
        
        # convert to new crs that is in meters, this will enable the densifying step.
        boundary_in_meters = self.geometries.to_crs(crs_in)

        # now in crs_proj4

        # merge the boundary into a single line
        merged_boundary_in_meters = shapely.ops.linemerge(
            shapely.geometry.MultiLineString(boundary_in_meters.geometry.values))

        # densify the boundary
        interp_merged_boundary = densify_geometry(
            merged_boundary_in_meters,
            step_size,
            crs_in=crs_in,
            crs_out=crs_out,
        ).geometry.values[0]
        
        # back into "EPSG:4326"
        
        return interp_merged_boundary
        
