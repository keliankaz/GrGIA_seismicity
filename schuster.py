import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# following: http://www.tectonics.caltech.edu/resources/schuster_spectrum/

class Schuster:
    
    tides = tides = dict(
        K2=0.4986,
        S2=0.5,
        M2=0.5175,
        N2=0.5274,
        K1=0.9973,
        P1=1.0028,
        O1=1.0758,
        T=13.661,
        S=14.765,
        A=27.555,
        E=31.812,
        SY=182.621,
        Y=365.260,
    )
    
    def __init__(
        self,
        times,
        t_start=None,
        t_end=None,
    ):
        self.raw_time = times
        
        if isinstance(self.raw_time[0],np.datetime64):
            _, self._time = self._convert_to_days(self.raw_time)
        else:
            self._time = self.raw_time
        
        self._times = np.sort(self._time) - np.min(self._time)
        
        if t_start is None: 
            self.t_start = self._times[0]
        if t_end is None:
            self.t_end = self._times[-1]
            
        self.duration = self.t_end - self.t_start

    def _convert_to_days(self, t):
        t0 = np.min(t)
        return t0,(t-t[0])/np.timedelta64(1,'D')

    def test(self, periods):
        return [self.test_period(period) for period in periods]
           
    def test_period(self, period):
        """ Calculates the log-probability that a walk is random """
        walk_x, walk_y = self.walk(period)
        return  np.exp(-(walk_x[-1]**2 + walk_y[-1]**2)/len(self._times))
        
    def walk(self, period):
        tlim = self.duration - self.duration%period
        t = self._times[self._times<=tlim]
        phase = np.mod(t,period)*2*np.pi/period
        walk_x = np.cumsum(np.cos(phase))
        walk_y = np.cumsum(np.sin(phase))
        
        return walk_x, walk_y
        
    def plot_walk(self,period, ax=None):
        
        if ax is None: 
            _,ax =plt.subplots()
        
        walk = self.walk(period)
        
        points = np.array(walk).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(0, len(walk[0]))
        lc = mpl.collections.LineCollection(segments, cmap='viridis',norm=norm)
        lc.set_array(np.arange(len(walk[0])))
        ax.add_collection(lc)
        ax.plot
        ax.autoscale()
        
        max_extent = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))
        
        ax.set(
            xlim=[-max_extent,max_extent],
            ylim=[-max_extent,max_extent],
            aspect='equal',
        )
        
        return ax
        
    def spectrum(self, periods=None, minimum_fraction=1/1000, number_of_periods=100, log=True, ax=None,include_tides=False):
        
        if periods is None:
            periods = np.logspace(
                np.log10(self.duration*minimum_fraction),
                np.log10(self.duration), 
                number_of_periods,
            )
        
        if ax is None: 
            _,ax =plt.subplots()
        
        ax.scatter(periods, self.test(periods),s=1,alpha=1)
        
        if include_tides is True:
            for k,v in self.tides.items():
                ax.axvline(v, lw=0.5,ls=':', label=f'{k}: {v}')
            ax.legend(bbox_to_anchor=(1,1))
    
        ax.set(
            xlabel='Period tested',
            ylabel=r"Schuster $p$-value",
            yscale='log',
        )
        
        ax.axhline(1,c='r')
        
        ax.invert_yaxis()
        if log:
            ax.set_xscale('log')
        
        return ax 
    

    
   
        
        
        
        
            