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


class PositiveStat:
    def __init__(
        self,
        time: np.ndarray = None,
        magnitude: np.ndarray = None,
        minimum_magnitude_delta: float = 0.1,
        rate_reference_magnitude: float = 5,
        magnitude_jitter:float = 0.01,
    ) -> None:
        
        """Set of positive catalog statistics directly following code shared by Nicholas Van Der Elst
        
        Usage:
        
        N = 100000
        times = np.random.random(N)
        times = np.sort(times)
        magnitudes = np.random.exponential(1,N)
        stat = PositiveStat(times,magnitudes)
        
        b_positive = stat.get_b()
        a_positive = stat.get_a()
        
        
        """
        
        self._raw_magnitude = magnitude
        self._raw_time = time
        self._flag_datetime = type(self._raw_time[0]) == np.datetime64
    
        if self._flag_datetime:
            self.start_time, self._raw_time_days = self._convert_to_days(self._raw_time)
        else:
            self._raw_time_days = self._raw_time
        
        self.minimum_magnitude_delta = minimum_magnitude_delta
        self.reference_magnitude = rate_reference_magnitude
        self.magnitude_jitter = magnitude_jitter
        
        self.check()
        
        sorted_indices = np.argsort(self._raw_time_days)
        self.time = self._raw_time_days[sorted_indices]
        self.magnitude = self._raw_magnitude[sorted_indices]
        
        self.magnitude += np.random.normal(0,self.magnitude_jitter, self.magnitude.shape)
        self.event_count = len(self.magnitude)
    
    def _convert_to_days(self, t):
        t0 = np.min(t)
        return t0,(t-t[0])/np.timedelta64(1,'D')
        
    def _convert_to_datetime(self,t0,t):
        return t0 + t*np.timedelta64(1,'D')
    
    def check(self):
        assert len(self._raw_magnitude) == len(self._raw_time)
        assert self.minimum_magnitude_delta >= 0
    
    def get_b(self) -> float:
        magnitude_differences = np.diff(self.magnitude)
        positive_indices = magnitude_differences >= self.minimum_magnitude_delta - 0.001 # Nicholas substracts 0.001 here for some reason?
        
        return 1/np.log(10) * 1/(
            np.mean(magnitude_differences[positive_indices]) - self.minimum_magnitude_delta
        )

    def get_a(self,referenced=True,N=1,filter='median'):
        t,a = self._get_a(referenced=referenced)
        
        if N>1:
            if filter=='median':
                a, t = self.moving_median(a,t,N)
            if filter=='mean':
                a, t = self.moving_average(a,t,N)
                
        if self._flag_datetime:
            t = self._convert_to_datetime(self.start_time, t)
        
        return t, a
    
    def _get_a(self, referenced=True) -> list[np.ndarray, np.ndarray]:
        
        positive_time_differences = []
        measurement_times = []
        measurement_mags = []
        
        for i in range(self.event_count):
            
            index_of_next_larger_event = np.argmax(
                (self.magnitude >= self.magnitude[i]  + self.minimum_magnitude_delta) & 
                (self.time > self.time[i])
            )
            # argmax returns the first max value so if all true or all false 
            # index_of_next_larger_event = 0
            
            if index_of_next_larger_event:
                positive_time_differences.append(
                    self.time[index_of_next_larger_event] - self.time[i]
                )
                measurement_times.append(
                    self.time[i]
                )  
                measurement_mags.append(self.magnitude[i])
        
        positive_time_differences, measurement_times, measurement_mags = [
            np.array(list) for list in [positive_time_differences, measurement_times, measurement_mags]
        ]
        
        b_positive = self.get_b()    
        
        if referenced is True:
            scaled_intervals = positive_time_differences * 10** -(
                b_positive * (measurement_mags + self.minimum_magnitude_delta - self.reference_magnitude)
            )
        else:
            scaled_intervals = positive_time_differences * 10**(-b_positive * self.minimum_magnitude_delta) 
        
        a_positive = 1/scaled_intervals
        
        sorted_indices = np.argsort(measurement_times)
        measurement_times, a_positive = [v[sorted_indices] for v in [measurement_times, a_positive]]
        
        return measurement_times, a_positive
    
    @staticmethod
    def moving_median(a, t=None, n=3):
        idx = np.arange(n) + np.arange(len(a)-n+1)[:,None]
        b = [row[row>0] for row in a[idx]]
        smoothed = np.array([np.median(c) for c in b])
        if t is None:
            return smoothed
        else:
            return smoothed, t[n-1:] # causal 
    
    @staticmethod
    def moving_average(a, t=None, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        smoothed = ret[n - 1:] / n
        if t is None:
            return smoothed
        else:
            return smoothed, t[n-1:] # causal 
    

            
        