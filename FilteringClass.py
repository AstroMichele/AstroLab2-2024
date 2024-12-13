import numpy as np
import matplotlib.pyplot as plt
from wotan import flatten
from wotan import transit_mask

class Filtering:
    def __init__(self):
        self.transit_time = 2459685.398095
        self.period = 3.2130578 
        self.transit_duration = 2.169 / 24.     #hours to days

    def filtering(self, dictionary, duration_factor, window, break_tol, method=''):
        self.time = dictionary['time']
        self.sap_flux = dictionary['sap_flux']
        self.sap_flux_err = dictionary['sap_flux_error']

        self.sap_flux_flatten, self.sap_flatten_model = flatten(self.time, self.sap_flux, method=method, window_length=window, break_tolerance=break_tol, return_trend=True)
        self.mask = transit_mask(time=self.time, period=self.period, duration=self.transit_duration*duration_factor, T0=self.transit_time)
        self.sap_flux_flatten_masked, self.sap_flatten_model_masked = flatten(self.time, self.sap_flux, method=method, window_length=window, break_tolerance=break_tol, return_trend=True, mask=self.mask)

        self.flux_median_error = np.median(self.sap_flux_err/self.sap_flatten_model_masked)
        self.STD_masked, self.STD_notmasked = self.compute_STD(self.sap_flux_flatten, self.sap_flux_flatten_masked, self.mask)

        tol = 0.00001
        if (np.abs(self.STD_masked - self.flux_median_error) or np.abs(self.STD_notmasked - self.flux_median_error)) < tol:
            print('This is worth, give it a look!')
        else:
            print('You may discard this configuration, but give it a look. Just in case...')

        print(f'Median flux error =  {self.flux_median_error:.6f}')
        print(f'STD with mask =  {self.STD_masked:.6f}')
        print(f'STD without mask =  {self.STD_notmasked:.6f}')

    def compute_STD(self, sap_flux_flatten, sap_flux_flatten_masked, mask):
        std_masked = np.std(sap_flux_flatten_masked[~mask])
        std_notmasked = np.std(sap_flux_flatten[~mask])
        return std_masked, std_notmasked

    #defining methods to display some plots for visual inspection

    def make_plot_model(self):
        plt.title('TESS: original lightcurve and flattening model')
        plt.scatter(self.time, self.sap_flux, c='C0', s=3)
        plt.errorbar(self.time, self.sap_flux, yerr=self.sap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
        plt.plot(self.time, self.sap_flatten_model, c='C1')

        plt.xlabel('BJD_TDB [d]')
        plt.ylabel('TESS SAP flux [e/s]')
        
        plt.show()
    
    def make_plot_normalized(self):
        plt.title('TESS: normalize SAP lightcurve')

        plt.scatter(self.time, self.sap_flux_flatten, c='C0', s=3)
        plt.errorbar(self.time, self.sap_flux_flatten, yerr=self.sap_flux_err/self.sap_flatten_model, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)

        plt.xlabel('BJD_TDB [d]')
        plt.ylabel('TESS flattened SAP flux')
        
        plt.show()

    def make_plot_folded(self, duration_factor, mask):
        self.phase_folded_time = (self.time - self.transit_time - self.period/2)%self.period - self.period/2

        if mask==True:
            plt.scatter(self.phase_folded_time[self.mask], self.sap_flux_flatten[self.mask], s=3)
            plt.errorbar(self.phase_folded_time[self.mask], self.sap_flux_flatten[self.mask], yerr=self.sap_flux_err[self.mask]/self.sap_flatten_model[self.mask], ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
        else:
            plt.scatter(self.phase_folded_time, self.sap_flux_flatten, s=3)
            plt.errorbar(self.phase_folded_time, self.sap_flux_flatten, yerr=self.sap_flux_err/self.sap_flatten_model, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)

        plt.axvline(-(self.transit_duration*duration_factor)/2, c='C3')
        plt.axvline((self.transit_duration*duration_factor)/2, c='C3')

        plt.xlabel('Orbital phase [d]')
        plt.ylabel('TESS flattened SAP flux')
        
        plt.show()

    def make_plot_comparison(self):
        plt.title('TESS: comparison between models')
        plt.scatter(self.time, self.sap_flux, c='C0', s=3)
        plt.errorbar(self.time, self.sap_flux, yerr=self.sap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
        plt.plot(self.time, self.sap_flatten_model_masked, c='C1', zorder=1, label='Masked')
        plt.plot(self.time, self.sap_flatten_model, c='C2', zorder=2, label='Unmasked')

        plt.legend()
        plt.xlabel('BJD_TDB [d]')
        plt.ylabel('TESS SAP flux [e/s]')
        
        plt.show()

    def make_plot_comparison_methods(self, method='', color=''):
        plt.title('TESS: comparison between models')
        plt.scatter(self.time, self.sap_flux, c='C0', s=3, zorder=0)
        plt.errorbar(self.time, self.sap_flux, yerr=self.sap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
        plt.plot(self.time, self.sap_flatten_model_masked, zorder=1, label=method, c=color)

        plt.legend()
        plt.xlabel('BJD_TDB [d]')
        plt.ylabel('TESS SAP flux [e/s]')