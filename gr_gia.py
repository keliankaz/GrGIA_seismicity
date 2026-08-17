#%%
import numpy as np
import pandas as pd
import datetime
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from geo_utils import (
    radius_search,
    k_nearest_search,
    get_geometry_neighbors,
)
from stat_utils import bootstrap_statistic
from data import MidAtlanticRidge, GIA

#%%
font = {"size": 9}
mpl.rc("font", **font)
mpl.rc("figure", dpi=300)
color_map = mpl.cm.get_cmap('RdYlBu_r')

#%%
AVAILABLE_STRAIN_RATE_FILES = {
    "july25_thick": "Jul25/yearly_strain_rate_tensors_l90ump5lm3_smoothice_S.mat",
    "july25_long_term": "Jul25/presentday_strain_rate_tensors_l90ump5lm3_ICE6G_S_fixbeta.mat",
}

DEFAULT_MAP_CRS = ccrs.NorthPolarStereo()

def reproduce_figures(
    plate_boundary_metadata: dict,
    region_metadata: dict,
    earthquake_catalog_metadata: dict,
    GrGIA_strain_metadata: dict,
    save_path: str = "figures/",
    figure_suffix: str = "",
    crs: ccrs.NorthPolarStereo = DEFAULT_MAP_CRS,
):

    # plate boundary data:
    MAR = MidAtlanticRidge(
        **plate_boundary_metadata,
        **region_metadata,
    )

    # earthquake data (ISC):
    earthquakes = MAR.get_earthquake_catalog(**earthquake_catalog_metadata["ISC"])
    
    earthquakes = earthquakes.get_time_slice(
        datetime.datetime(GrGIA_strain_metadata['data_config']["starttime"],1,1),
        datetime.datetime(GrGIA_strain_metadata['data_config']["endtime"]+1,1,1)
    )
    
    earthquakes.mag_completeness = earthquake_catalog_metadata["mag_completeness"]

    # GIA model output:
    gia = GIA(**GrGIA_strain_metadata)

    date_range = pd.date_range(
        start=datetime.datetime(GrGIA_strain_metadata['data_config']["starttime"], 1, 1),
        end=datetime.datetime(GrGIA_strain_metadata['data_config']["endtime"], 1, 1),
        periods=GrGIA_strain_metadata['data_config']["number_of_times"]+1, # + 1 because we want to include the last year for seismicity analysis
    )

    # strain -> earthquake
    earthquake_time_dt = pd.to_datetime(earthquakes.catalog.time).dt

    strain = gia.query(
        earthquake_time_dt.year + earthquake_time_dt.dayofyear / 365 - 1,
        earthquakes.catalog.lat.to_numpy(),
        earthquakes.catalog.lon.to_numpy(),
    )


    earthquakes.catalog["gia_strain_tensor"] = [
        strain[i, :, :] for i in range(len(earthquakes))
    ]

    # boundary geometry -> earthquake:
    indices = k_nearest_search(
        np.array([MAR.merged_geometry_meters.xy[1], MAR.merged_geometry_meters.xy[0]]).T,
        earthquakes.catalog[["lat", "lon"]].values,
    )
    earthquakes.catalog["distance_along_boundary"] = (
        indices * plate_boundary_metadata["stepsize"]
    )

    segment = k_nearest_search(
        np.array([MAR.merged_geometry_meters.xy[1], MAR.merged_geometry_meters.xy[0]]).T,
        earthquakes.catalog[["lat", "lon"]].values,
        k=2,
    )
    earthquakes.catalog["plate_boundary_segment"] = [
        segment[i, :] for i in range(segment.shape[0])
    ]

    def unravelled_grid(t, lat, lon):
        time_grid, lat_grid = [q.ravel() for q in np.meshgrid(t, lat)]
        _, lon_grid = [q.ravel() for q in np.meshgrid(t, lon)]
        return np.column_stack([time_grid, lat_grid, lon_grid])


    plate_boundary_unravelled_grid = unravelled_grid(
        (date_range.year + (date_range.dayofyear - 1) / 365).values[1:],  # time,
        MAR.merged_geometry_meters.xy[1],  # latitudes of the plate boundary
        MAR.merged_geometry_meters.xy[0],  # longitudes of the plate boundary
    )

    strain_at_plate_boundary = gia.query(
        plate_boundary_unravelled_grid[:, 0],
        plate_boundary_unravelled_grid[:, 1],
        plate_boundary_unravelled_grid[:, 2],
    )

    strain_at_plate_boundary_grid = np.reshape(
        strain_at_plate_boundary,
        (len(MAR.merged_geometry_meters.xy[0]), len(date_range) - 1, 3, 3),
    )

    normal_strain_grid = []
    for i in np.arange(strain_at_plate_boundary_grid.shape[0] - 1):
        normal_strain_row = []
        for j in np.arange(strain_at_plate_boundary_grid.shape[1]):
            normal_strain_row.append(
                gia.get_normal_strain_latlon(
                    epsilon=strain_at_plate_boundary_grid[i, j, :, :],
                    latitude_1=MAR.merged_geometry_meters.xy[1][i],
                    longitude_1=MAR.merged_geometry_meters.xy[0][i],
                    latitude_2=MAR.merged_geometry_meters.xy[1][i + 1],
                    longitude_2=MAR.merged_geometry_meters.xy[0][i + 1],
                ).squeeze()
            )
        normal_strain_grid.append(normal_strain_row)
    normal_strain_grid = np.array(normal_strain_grid)

    # plot I2 on a lat, lon grid
    lats = np.linspace(50, 85, 200)
    lons = np.linspace(-50, 30, 200)

    lat_grid, lon_grid = np.meshgrid(lats, lons)

    lat = lat_grid.flatten()
    lon = lon_grid.flatten()

    e_grid = gia.query(
        (date_range.year + (date_range.dayofyear - 1) / 365).values[-1] * np.ones_like(lat),
        lat,
        lon,
    )


    I1 = [gia.I1(e[:2, :2]) for e in e_grid]
    I1 = np.array(I1).reshape(lat_grid.shape)

    catalog = earthquakes.catalog

    rate_grid = []
    all_indices = []

    delta_t = date_range[1] - date_range[0]
    print(delta_t)

    # Space-time rate:
    for t1, t2 in zip(date_range[:-1], date_range[1:]):
        sub_catalog = catalog[(catalog.time < t2) & (catalog.time > t1)]
        indices = get_geometry_neighbors(
            sub_catalog,
            MAR.merged_geometry_meters,
            fun=radius_search,
            kwarg={"radius":  plate_boundary_metadata["stepsize"]/1e3},
        )
        all_indices.append(indices)
        rate_grid.append([len(i) for i in indices])

    rate_grid = np.array(rate_grid).T

    ################################################################################
    # Figure 2: Space-time rate of earthquakes

    crs = ccrs.NorthPolarStereo()
    fig, AX = plt.subplots(1,2,subplot_kw={"projection": crs}, figsize=(6.5, 4), dpi=300)

    ax=AX[0]
    color_range = [-2, 2]
    bdr = ax.scatter(
        MAR.merged_geometry_meters.xy[0][:-1], MAR.merged_geometry_meters.xy[1][:-1], c=-np.mean(normal_strain_grid, axis=1), s=5, 
        transform=ccrs.PlateCarree(), cmap="RdYlBu_r", vmin=color_range[0], vmax=color_range[1])
    ax.coastlines()


    RADIUS = 50  # km
    indices = get_geometry_neighbors(
        catalog, MAR.merged_geometry_meters, fun=radius_search, kwarg={"radius": RADIUS}
    )


    axins1 = ax.inset_axes(
        [0.45,0.15,0.45,0.03],
    )

    fig.colorbar(
        bdr, 
        cax=axins1,
        label="Normal nanostrain/yr",
        orientation="horizontal",
    )

    ax = AX[1]
    RADIUS = 50  # km
    indices = get_geometry_neighbors(
        catalog, MAR.merged_geometry_meters, fun=radius_search, kwarg={"radius": RADIUS}
    )
    rate = [len(i) for i in indices]

    bdr = ax.scatter(
        *MAR.merged_geometry_meters.xy, c=np.log10(rate), s=5, transform=ccrs.PlateCarree()
    )
    ax.coastlines()

    axins1 = ax.inset_axes(
        [0.45,0.15,0.45,0.03],
    )


    fig.colorbar(
        bdr,
        cax=axins1, 
        orientation="horizontal",
        label=r"$\log(N_{Eq})$" + f", in {RADIUS} km radius",
    )

    plt.tight_layout()
    plt.savefig(save_path + "grgia_map" + figure_suffix + ".pdf", dpi=300)

    ################################################################################
    # Figure 3: 

    #Get plate rates to normalize eq rate by
    NA_pole = [48.709, -78.167, 0.7486]  # deg/MA - DeMets et al. [1994]
    EU_pole = [61.066, -85.819, 0.8591]  # deg/MA - DeMets et al. [1994]

    def cartesian(p):
        return np.array([
            p[:, 2] * np.cos(p[:, 0]) * np.cos(p[:, 1]),
            p[:, 2] * np.cos(p[:, 0]) * np.sin(p[:, 1]),
            p[:, 2] * np.sin(p[:, 0]),
        ])

    def get_spreading_rate(
        lat: np.ndarray = None,
        lon: np.ndarray = None,
        pole1: list[float, float, float] = NA_pole,
        pole2: list[float, float, float] = EU_pole,
    ) -> np.ndarray:
        """Gets the relative plate rate at a plate boundary for each lat lon pair given two euler poles

        The `pole` contains the **** lon, lat **** and angular roation rate (assumed to be in deg/MA). 
        
        Note that a relatively exhaustive list of pole can be found in the following link:
        http://peterbird.name/oldFTP/PB2002/PB2002_poles.dat.txt

        Returns the spreading rate in km/Ma (or, equivalently mm/yr)

        """

        pole1, pole2, lat, lon = map(np.radians, [pole1, pole2, lat, lon])

        earth_radius_km = 6367
        locations = np.column_stack((lat, lon, np.ones_like(lat) * earth_radius_km))

        omega1, omega2, r = map(
            cartesian, [np.expand_dims(pole1, 0), np.expand_dims(pole2, 0), locations]
        )

        omega = omega1 - omega2

        v = np.cross(omega.T, r.T)

        return np.sqrt(np.sum(v * v, 1))

    plate_boundary_velocity = get_spreading_rate(
        lat=MAR.merged_geometry_meters.xy[1], lon=MAR.merged_geometry_meters.xy[0]
    )

    def summary_plot(x, y, cut, theshold=0, data_label=None, AX=None, annotate=True):
        """
        Summary plot of the data.

        Parameters
        ----------
        x : array-like
            The data to plot.
        y : array-like
            The data to plot.
        cut : separates the 'high' and 'low' values of x.
        theshold : optional buffer around the cut.
        AX : optional row of 3 axes to draw into (a new figure is made if None).
        annotate : draw the legend and the bootstrap title (turn off on
            repeated rows of a stacked figure).

        """

        if AX is None:
            _, AX = plt.subplots(1, 3, sharey=True, figsize=(6.5, 3))
        ax = AX[0]

        # Labels are flipped pos/neg to match convention since strain is *-1
        positive_index = x > cut + theshold
        negative_index = x < cut - theshold
        ax.scatter(-x, y, s=2, alpha=0.1, c="k")
        ax.scatter(-x[negative_index], y[negative_index], s=2, alpha=0.5, color="C3")
        ax.scatter(-x[positive_index], y[positive_index], s=2, alpha=0.5)
        ax.set_xlabel(f"{data_label}")
        ax.set_ylabel("Event rate / plate rate \n" + r"[(50 km segment)$^{-1}$ m$^{-1}$]")
        ax.axvline(-cut, c="lightgrey", ls="--", label="mean")
        if annotate:
            ax.legend()

        ax = AX[1]
        range_95th_percentile = np.percentile(y, [0.5, 99.5])
        bins = np.linspace(*range_95th_percentile, 10)
        ax.hist(
            y[negative_index],
            bins=bins,
            orientation="horizontal",
            alpha=0.7,
            label=f"strain<{cut - theshold}",
            density=True,
            color="C3",
        )
        ax.hist(
            y[positive_index],
            bins=bins,
            orientation="horizontal",
            alpha=0.7,
            label=f"strain>{cut + theshold}",
            density=True,
        )

        ax.set(
            xticks=[],
        )

        ax = AX[2]
        number_of_bootstrap_samples = 100000

        negative_boot_sample = bootstrap_statistic(
            y[negative_index],
            np.mean,
            boot=number_of_bootstrap_samples,
        )

        positive_boot_sample = bootstrap_statistic(
            y[positive_index],
            np.mean,
            boot=number_of_bootstrap_samples,
        )

        ax.hist(
            negative_boot_sample,
            orientation="horizontal",
            alpha=0.7,
            bins=50,
            color="C3",
        )

        ax.hist(
            positive_boot_sample,
            orientation="horizontal",
            alpha=0.7,
            bins=50,
            color="C0",
        )

        ax.set(
            xticks=[],
            title=(
                f"Bootstrap mean\n(N={number_of_bootstrap_samples:,})"
                if annotate
                else None
            ),
        )

        return AX

    def arrow_annotation(ax, y_pos=0.05, x_center=0.9, arrow_len=0.1, color="k", direction="outwards", orientation="horizontal"):

        if orientation == "horizontal":
            if direction == "outwards":
                # left-pointing arrow
                ax.annotate(
                    "", xy=(x_center - arrow_len, y_pos), xycoords="axes fraction",
                    xytext=(x_center, y_pos), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )

                # right-pointing arrow
                ax.annotate(
                    "", xy=(x_center + arrow_len, y_pos), xycoords="axes fraction",
                    xytext=(x_center, y_pos), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )
            elif direction == "inwards":
                # right-pointing arrow
                ax.annotate(
                    "", xy=(x_center, y_pos), xycoords="axes fraction",
                    xytext=(x_center - arrow_len, y_pos), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )

                # left-pointing arrow
                ax.annotate(
                    "", xy=(x_center, y_pos), xycoords="axes fraction",
                    xytext=(x_center + arrow_len, y_pos), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )
        elif orientation == "vertical":
            if direction == "outwards":
                # down-pointing arrow
                ax.annotate(
                    "", xy=(x_center, y_pos - arrow_len), xycoords="axes fraction",
                    xytext=(x_center, y_pos), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )

                # up-pointing arrow
                ax.annotate(
                    "", xy=(x_center, y_pos + arrow_len), xycoords="axes fraction",
                    xytext=(x_center, y_pos), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )
            elif direction == "inwards":
                # up-pointing arrow
                ax.annotate(
                    "", xy=(x_center, y_pos), xycoords="axes fraction",
                    xytext=(x_center, y_pos - arrow_len), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )

                # down-pointing arrow
                ax.annotate(
                    "", xy=(x_center, y_pos), xycoords="axes fraction",
                    xytext=(x_center, y_pos + arrow_len), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": color}
                )

    # unit conversion: events / catalog duration / node area / (mm/yr)  -> events / yr / m^2 / (m/yr)
    normalized_rate =((rate / plate_boundary_velocity)[1:] + (rate/plate_boundary_velocity)[:-1])/2 # strains are measured in between node points
    
    converted_rate = (
        normalized_rate 
        / ((earthquakes.end_time - earthquakes.start_time)/ np.timedelta64(1, 'Y'))
        # / (np.pi * (plate_boundary_metadata["stepsize"])**2) # m^2 
        / (1/1e3) # mm/yr -> m/yr 
    )
    
    # Both strain components share the same y variable (normalized event rate),
    # so they are stacked as rows of a single figure with a shared y axis.
    _, AX = plt.subplots(2, 3, sharey=True, figsize=(6.5, 5.5))

    cut = np.mean(np.mean(normal_strain_grid, axis=1))
    summary_plot(
        np.mean(normal_strain_grid, axis=1),
        converted_rate,
        cut,
        theshold=0,
        data_label=r"Time averaged $\dot{\epsilon}_N$ [s$^{-1}$]",
        AX=AX[0],
    )

    arrow_annotation(AX[0, 0], color="C3", direction="outwards", orientation="horizontal")
    arrow_annotation(AX[0, 0], color="C0", x_center=0.1, direction="inwards", orientation="horizontal")

    cut = np.mean(np.mean(strain_at_plate_boundary_grid[:, :, 2, 2], axis=1))

    summary_plot(
        np.mean(strain_at_plate_boundary_grid[:, :, 2, 2][1:], axis=1),
        converted_rate,
        cut,
        theshold=0,
        data_label=r"Time averaged $\dot{\epsilon}_z$ [s$^{-1}$]",
        AX=AX[1],
        annotate=False,
    )

    arrow_annotation(AX[1, 0], color="C3", y_pos=0.15, direction="outwards", orientation="vertical")
    arrow_annotation(AX[1, 0], color="C0", y_pos=0.15, x_center=0.1, direction="inwards", orientation="vertical")

    plt.tight_layout()
    plt.savefig(save_path + "grgia_histograms" + figure_suffix + ".pdf", dpi=300)


    ################################################################################
    # Figure 4: 
    fig, AX = plt.subplots(2, 1, figsize=(5, 5), dpi=200, gridspec_kw={'hspace': -0.6})

    extentional_indices = np.where(
        np.mean(normal_strain_grid, axis=1) < np.mean(normal_strain_grid)
    )[0]
    contraction_indices = np.where(
        np.mean(normal_strain_grid, axis=1) > np.mean(normal_strain_grid)
    )[0]

    # plot the space-averaged strain vs time
    ax = AX[0]
    v = np.mean(normal_strain_grid[extentional_indices, :], axis=0)
    ax.plot(
        date_range[1:],
        v - v[0],
        label="more contractional",
        color="C0",
    )

    v = np.mean(normal_strain_grid, axis=0)
    ax.plot(date_range[1:], v - v[0], color="k", label="space-averaged strain")

    v = np.mean(normal_strain_grid[contraction_indices, :], axis=0)
    ax.plot(
        date_range[1:],
        v - v[0],
        label="more extensional",
        color="C3",
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set(xticks=[], xlim=[date_range[1:].min(), date_range[1:].max()])

    axb = AX[1].twinx()

    axb.plot(
        date_range[1:],
        np.mean(rate_grid, axis=0),
        color="k",
    )

    axb.plot(
        date_range[1:],
        np.mean(rate_grid[contraction_indices, :], axis=0),
        color="C0",
        label="more contractional",
    )

    axb.plot(
        date_range[1:],
        np.mean(rate_grid[extentional_indices, :], axis=0),
        color="C3",
        label="more extensional",
    )

    axb.spines['top'].set_visible(False)
    axb.spines['left'].set_visible(False)
    AX[1].spines['top'].set_visible(False)
    AX[1].spines['left'].set_visible(False)
    AX[1].spines['bottom'].set_visible(False)
    AX[1].set(yticks=[])
    # remove facecolor
    axb.set_facecolor('none')
    AX[1].set_facecolor('none')

    axb.set(xlabel="Year", ylabel="Mean earthquake rate\n[events/yr per segment]", xlim=[date_range[1:].min(), date_range[1:].max()])
    ax.set(ylabel=f"Strain rate change since {date_range[1:].min().year}")


    # top left legend
    AX[0].legend(
        loc='upper left',
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(save_path + "grgia_strain_vs_time" + figure_suffix + ".pdf", dpi=300)



    ## ##############################################################################
    # supplemental figure: comparison of apertures and lags: 
    fig, ax = plt.subplots(figsize=(3,3))
 
    outlier_index = 12
    
    cont = np.mean(normal_strain_grid[contraction_indices, :], axis=0)
    ext = np.mean(normal_strain_grid[extentional_indices, :], axis=0)

    x = (cont - cont[0]) - (ext - ext[0])
    y = np.mean(rate_grid[extentional_indices, :], axis=0) - np.mean(rate_grid[contraction_indices, :], axis=0)

    # # remove outlinr

    # x = x[np.arange(len(x)) != outlier_index]
    # y = y[np.arange(len(y)) != outlier_index]

    lag = 3 # years

    ax.scatter(
        x[:-lag],
        y[lag:],
        s=4,
    )

    # a bootstrapped linear fit
    def boot_fit(x, y, n_boot=10000, ax=None):   
        slopes = []
        intercepts = []

        for i in range(n_boot):
            idx = np.random.choice(len(x), size=len(x), replace=True)
            x_sample = x[idx]
            y_sample = y[idx]
            coef = np.polyfit(x_sample, y_sample, 1)
            slopes.append(coef[0])
            intercepts.append(coef[1])

        # Plot the bootstrap linear fits (optional)
        if ax is not None:
            for s, ic in zip(slopes[:1000], intercepts[:1000]):  # Plot only first 100 for visibility
                ax.plot(x, s * x + ic, color='gray', lw=0.5, alpha=0.01)
            
        return slopes, intercepts

    slopes, intercepts = boot_fit(x[:-lag],y[lag:], ax=ax)


    # Plot median fit
    median_slope = np.median(slopes)
    median_intercept = np.median(intercepts)
    ax.plot(x, median_slope * x + median_intercept, color="k", lw=1, alpha=0.5, label='Bootstrapped fit')
    ax.set(
        xlabel="Strain rate change since {starttime}".format(starttime=GrGIA_strain_metadata['data_config']["starttime"]),
        ylabel=rf"$\Delta R$ ({lag} year lag)",
    )


    # Create inset in the top left corner
    # Create inset in the top left corner
    ax_inset = ax.inset_axes([0.05, 0.7, 0.5, 0.25])  # [left, bottom, width, height] in axes fraction; values position it in top left

    bins = np.linspace(min(slopes), max(slopes), 50)
    ax_inset.hist(slopes, bins=bins, color='grey', alpha=0.5, edgecolor=None)
    ax_inset.hist(slopes, bins=bins, color='grey', histtype='step', lw=0.5)

    ax_inset.set(
        xlabel="Slope",
        yticks=[],
    )

    # remove the top and right spines
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['left'].set_visible(False)
    ax_inset.axvline(median_slope, color='k', alpha=0.5, lw=1)
    ax_inset.text(
        0.97, 0.97,
        f"$p: ${np.mean(np.array(slopes)<0):.2f}\n $N$: {len(slopes)}",
        transform=ax_inset.transAxes,
        ha='right', va='top'
    )

    ax.axhline(0, color='k', lw=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    p = []
    med_slopes = []
    lags = np.arange(-8, 8)  # lags [years]
    for lag in lags:
        if lag > 0:
            xx, yy = x[:-lag], y[lag:]
        elif lag < 0:
            xx, yy = x[-lag:], y[:lag]
        else:  # lag == 0
            xx, yy = x, y
        slopes, intercepts = boot_fit(xx, yy)
        p_val = np.mean(np.array(slopes) < 0)
        med_slopes.append(np.mean(slopes))
        p.append(p_val)
        
    plt.savefig(save_path + "lag_correlation" + figure_suffix + ".pdf", dpi=300, bbox_inches="tight")
        
    fig, ax = plt.subplots(figsize=(3,1))
    ax.axvline(0, color='k', lw=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.plot(lags, p, c='indianred')
    ax.set(
        xlabel="Lag [years]", 
        ylabel="$p$-value",
        xticks=lags[1::2],   
    )

    ax.grid(True, which="major", axis="y", color="0.85", linewidth=1)
    ax.tick_params(axis='y', length=0)  # Remove tick marks but keep labels
    ax.set(
        yscale="log",
    )



    # Put y-axis ticks/labels on the right (matches your figure)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.text(-lags[0]/2 / (lags[-1]-lags[0]), 0.2, "acausal", ha='center', va='top', fontsize=8, transform=ax.transAxes)
    ax.axvspan(lags[0], 0, color="k", alpha=0.02, zorder=0)


    plt.savefig(save_path + "lag_vs_p" + figure_suffix + ".pdf", dpi=300, bbox_inches="tight")
    
    
    
#%% 

if __name__ == "__main__":

    data_dir = "data/"

    plate_boundary_metadata = {
        "filename": data_dir + "plate_boundaries/PB2002_boundaries.shp",
        "boundary_names": ["NA-EU", "EU-NA"],
        "stepsize": 50000, # meters (the point spacing of the plate boundary)
    }

    earthquake_catalog_metadata = {
        "ISC": {
            "filename": data_dir + "global_M4.csv",
            "query":{
                "minimum_magnitude": 4.0,
                "starttime": "1960-01-01",
                "endtime": "2022-01-01",
                "reload":False,
            },
            "buffer_km": 100,
        },
        "mag_completeness":4.0, # change this to avoid re-downloading the catalog
    }

    earthquake_metadata = earthquake_catalog_metadata["ISC"]  # Option here

    AVAILABLE_STRAIN_RATE_FILES = {
        "july25_thick": "Jul25/yearly_strain_rate_tensors_l90ump5lm3_smoothice_S.mat",
        "july25_long_term": "Jul25/presentday_strain_rate_tensors_l90ump5lm3_ICE6G_S_fixbeta.mat",
    }

    GrGIA_strain_metadata = {
        "filename": data_dir + AVAILABLE_STRAIN_RATE_FILES["july25_thick"],
        "data_key": "strain_out", # unfortunately this is not consistent accross datasets.
        "data_config":{
            "starttime": 1992,
            "endtime": 2019,
            "number_of_times": 27,
            "latitude_range": [-90, 90],
            "number_of_latitudes": 510,
            "longitude_range": [-180, 180],
            "number_of_longitudes": 1022,
        },
        "strain_units": [1e-9, "s^{-1}"],  # scale, unit
        "strain_positive_convention": "compression",
        "longterm_filename": data_dir + AVAILABLE_STRAIN_RATE_FILES["july25_long_term"], 
    }

    crs = ccrs.NorthPolarStereo()

    region_metadata = {
        "bounding_box": [-180, 55, 10, 90],
        "exclude_iceland":True,
        "iceland_bounds": [(-25.5, 63), (-25.5, 67), (-12.5, 67), (-12.5, 63), (-25.5, 63)],
    }
    
    reproduce_figures(
        figure_suffix="_mc4.0",
        plate_boundary_metadata=plate_boundary_metadata,
        earthquake_catalog_metadata=earthquake_catalog_metadata,
        GrGIA_strain_metadata=GrGIA_strain_metadata,
        region_metadata=region_metadata,
        crs=crs,
    )
    
    # change mc to 4.5
    earthquake_catalog_metadata["mag_completeness"] = 4.5
    reproduce_figures(
        figure_suffix="_mc4.5",
        plate_boundary_metadata=plate_boundary_metadata,
        earthquake_catalog_metadata=earthquake_catalog_metadata,
        GrGIA_strain_metadata=GrGIA_strain_metadata,
        region_metadata=region_metadata,
        crs=crs,
    )


# %%
