import numpy as np


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
