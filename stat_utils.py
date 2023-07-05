import numpy as np
from typing import Optional


def jitter(x, sigma):
    return x + np.random.normal(0, sigma, x.shape)


def get_bvalue(M, Mc, inc=0.0, apply_jitter=False):
    """Returns the b-value for a given set of magnitudes M."""

    M = M[M >= Mc]

    if apply_jitter:
        M = jitter(M, inc)
        utsu_correction = 0
    else:
        utsu_correction = inc * 0.5

    return np.log10(np.exp(1)) / (np.mean(M) - Mc + utsu_correction)


def get_bpositive(M, del_Mc, inc=0.0, apply_jitter=False):
    """Returns the b-value for a given set of magnitudes M using b positive.

    Note that this assumes that magntidudes are sorted in increasing temporal order."""

    delta_M = np.diff(M)
    delta_positive = delta_M[delta_M > 0]

    return get_bvalue(delta_positive, del_Mc, inc, apply_jitter)


def get_clustered_fraction(t):
    """Returns the fraction of clustered events for a given time series of events."""

    dt = np.diff(t)
    tau = dt / np.mean(dt)

    return 1 / np.std(tau)


def bootstrap_statistic(data, f, kwarg=None, boot=1000):
    """Returns the bootstrap distribution of a statistic `f` of a 1D dataset `data`."""

    if kwarg is None:
        kwarg = {}

    return np.array(
        [
            f(sample, **kwarg)
            for sample in np.random.choice(data, size=(boot, len(data)))
        ]
    )


def bootstrap_df(df, f, kwarg=None, boot=1000):
    """Returns the bootstrap distribution of a statistic `f` for a dataframe `df`."""

    if kwarg is None:
        kwarg = {}

    return np.array(
        [f(df.sample(n=len(df), replace=True), **kwarg) for _ in np.arange(boot)]
    )


def get_apositive(
    mag: np.ndarray, 
    t: np.ndarray, 
    dMc: float = 0.2, 
    Mref: float = 5, 
    b_pos: Optional[float] = None,
):
    """Returns the a(+) values for a given set of magnitudes M and 
    times t using b positive.
    
    If b_pos is not provided, it is calculated from the data using the dMc
    value provided.
    
    Input:
        mag: array of magnitudes
        t: array of times
        dMc: magnitude of completeness
        Mref: reference magnitude
        b_pos: b-value for positive magnitudes
        
    Output:
        r_pos: pairwise estimates of the rate of Mref earthquakes (r_pos = 1/dt_pos)
        dt_pos: pairwise interevent times, corrected for dMc but not scaled 
            to Mref
        t_measure: absolute time of the larger earthquake with M > mag + dMc.
            (Usefule for associating r_pos with the mean time of the pair.)
    
    """
    
    dt_pos = []
    t_measure = []
    
    assert not np.any(np.diff(t) < 0), "t must be sorted in increasing order."
    
    for mag_i, t_i in zip(mag,t):
        next_larger_event_bool= (
            (mag >= mag_i + dMc) &
            (t > t_i)
        )
        
        if np.any(next_larger_event_bool):
            # argmax returns the first True value in the array of next_larger_event_bool
            next_larger_index = np.argmax(next_larger_event_bool)
            dt_pos.append(t[next_larger_index] - t_i)
            t_measure.append(t[next_larger_index])
        else:
            dt_pos.append(np.nan)
            t_measure.append(np.nan)
            
    dt_pos = np.array(dt_pos)
    t_measure = np.array(t_measure)
            
    if b_pos is None:
        b_pos = get_bpositive(mag, dMc)
    
    dt_pos_unscaled = dt_pos * 10 ** (-b_pos * dMc)
    dt_pos = dt_pos_unscaled * 10 ** (-b_pos * (mag + dMc - Mref))
    
    r_pos = 1 / dt_pos
    
    return r_pos, dt_pos, t_measure
        