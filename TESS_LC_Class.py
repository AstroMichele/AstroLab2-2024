import numpy as np
import matplotlib.pyplot as plt
import pickle
from wotan import flatten
from wotan import transit_mask

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

class Selection:
    def __init__(self, lcf_hdu, sector):
        self.bjd_tdb = lcf_hdu[1].data['TIME'] + lcf_hdu[1].header['BJDREFI'] + lcf_hdu[1].header['BJDREFF']
        time = lcf_hdu[1].data['TIME']
        self.time_offset = int(time[0]) + lcf_hdu[1].header['BJDREFI'] + lcf_hdu[1].header['BJDREFF']
        self.offset = str(int(time[0]) + lcf_hdu[1].header['BJDREFI'] + lcf_hdu[1].header['BJDREFF'])
        self.sap_flux = lcf_hdu[1].data['SAP_FLUX']
        self.sap_flux_err = lcf_hdu[1].data['SAP_FLUX_ERR']
        self.pdcsap_flux = lcf_hdu[1].data['PDCSAP_FLUX']
        self.pdcsap_flux_err = lcf_hdu[1].data['PDCSAP_FLUX_ERR']
        self.quality_bitmask = lcf_hdu[1].data['QUALITY']
        self.sector = str(sector)

    def selection(self):
        finite_selection = np.isfinite(self.pdcsap_flux)

        self.conservative_selection = ~(self.quality_bitmask > 0) & finite_selection

        plt.scatter(self.bjd_tdb[self.conservative_selection] - self.time_offset, self.sap_flux[self.conservative_selection],
            s=3, label='SAP - selected data')
        plt.scatter(self.bjd_tdb - self.time_offset, self.pdcsap_flux, s=5, label='PDCSAP')
        plt.scatter(self.bjd_tdb[~self.conservative_selection] - self.time_offset, self.sap_flux[~self.conservative_selection],
            s=3, c='r', label='SAP - excluded data')
        plt.errorbar(self.bjd_tdb[self.conservative_selection] - self.time_offset, self.sap_flux[self.conservative_selection],
            yerr=self.sap_flux_err[self.conservative_selection], fmt=' ', alpha=0.5, 
            ecolor='k', zorder=-1)
        plt.xlabel('BJD_TDB - '+self.offset+' [d]')
        plt.ylabel('Flux [$e^-/s$]')

        plt.legend()
        plt.show()

        sector_dictionary = {
        'offset' : self.time_offset,
        'time': self.bjd_tdb[self.conservative_selection],
        'sap_flux': self.sap_flux[self.conservative_selection],
        'sap_flux_error': self.sap_flux_err[self.conservative_selection],
        'pdcsap_flux': self.pdcsap_flux[self.conservative_selection],
        'pdcsap_flux_error': self.pdcsap_flux_err[self.conservative_selection],
        }

        pickle.dump(sector_dictionary, open('../Results/TESS/sector'+self.sector+'_selected.p', 'wb'))

        return sector_dictionary
        
    def manual_selection(self, time0):
        final_selection = self.conservative_selection & (self.bjd_tdb > time0)

        plt.scatter(self.bjd_tdb[self.conservative_selection] - self.time_offset, self.sap_flux[self.conservative_selection],
            s=5, label='SAP - selected data')
        plt.scatter(self.bjd_tdb - self.time_offset, self.pdcsap_flux, s=5, label='PDCSAP')


        plt.scatter(self.bjd_tdb[~self.conservative_selection] - self.time_offset, self.sap_flux[~self.conservative_selection],
            s=5, c='r', label='SAP - excluded data')
        plt.scatter(self.bjd_tdb[~final_selection & self.conservative_selection] - self.time_offset, self.sap_flux[~final_selection & self.conservative_selection],
            s=15, c='y', marker='x', label='SAP - manually excluded')
        plt.errorbar(self.bjd_tdb[self.conservative_selection] - self.time_offset, self.sap_flux[self.conservative_selection],
            yerr=self.sap_flux_err[self.conservative_selection], fmt=' ', alpha=0.5, 
            ecolor='k', zorder=-1)
        plt.xlabel('BJD_TDB - '+self.offset+' [d]')
        plt.ylabel('Flux [$e^-/s$]')

        plt.legend()
        plt.show()

        sector_dictionary = {
        'offset' : self.time_offset,
        'time': self.bjd_tdb[final_selection],
        'sap_flux': self.sap_flux[final_selection],
        'sap_flux_error': self.sap_flux_err[final_selection],
        'pdcsap_flux': self.pdcsap_flux[final_selection],
        'pdcsap_flux_error': self.pdcsap_flux_err[final_selection],
        }

        pickle.dump(sector_dictionary, open('../Results/TESS/sector'+self.sector+'_selected.p', 'wb'))

        return sector_dictionary

#############################################################################################################################################################

class Filtering:
    def __init__(self):
        self.transit_time = 2459685.398095
        self.period = 3.2130578 
        self.transit_duration = 2.169 / 24.     #hours to days

    def filtering(self, dictionary, window, break_tol, duration_factor, method=''):
        self.time = dictionary['time']
        self.time_offset = dictionary['offset']
        self.offset = str(self.time_offset)
        tol = 0.1

        #SAP
        self.sap_flux = dictionary['sap_flux']
        self.sap_flux_err = dictionary['sap_flux_error']

        self.sap_flux_flatten, self.sap_flatten_model = flatten(self.time, self.sap_flux, method=method, window_length=window, break_tolerance=break_tol, return_trend=True)
        self.mask = transit_mask(time=self.time, period=self.period, duration=self.transit_duration*duration_factor, T0=self.transit_time)
        self.sap_flux_flatten_masked, self.sap_flatten_model_masked = flatten(self.time, self.sap_flux, method=method, window_length=window, break_tolerance=break_tol, return_trend=True, mask=self.mask)

        self.sap_flux_median_error = np.median(self.sap_flux_err/self.sap_flatten_model_masked)
        self.STD_masked, self.STD_notmasked = self.compute_STD(self.sap_flux_flatten, self.sap_flux_flatten_masked, self.mask)

        if ((1 - self.sap_flux_median_error/self.STD_masked) or (1 - self.sap_flux_median_error/self.STD_notmasked)) < tol:
            print('This is worth, give it a look!')
        else:
            print('You may discard this configuration, but give it a look. Just in case...')

        print(f'Median flux error (SAP) =  {self.sap_flux_median_error:.6f}')
        print(f'STD with mask (SAP) =  {self.STD_masked:.6f}')
        print(f'STD without mask (SAP) =  {self.STD_notmasked:.6f}')
        print('##########################################')

        #PDCSAP
        self.pdcsap_flux = dictionary['pdcsap_flux']
        self.pdcsap_flux_err = dictionary['pdcsap_flux_error']

        self.pdcsap_flux_flatten, self.pdcsap_flatten_model = flatten(self.time, self.pdcsap_flux, method=method, window_length=window, break_tolerance=break_tol, return_trend=True)
        self.pdcmask = transit_mask(time=self.time, period=self.period, duration=self.transit_duration*duration_factor, T0=self.transit_time)
        self.pdcsap_flux_flatten_masked, self.pdcsap_flatten_model_masked = flatten(self.time, self.pdcsap_flux, method=method, window_length=window, break_tolerance=break_tol, return_trend=True, mask=self.pdcmask)

        self.pdcsap_flux_median_error = np.median(self.pdcsap_flux_err/self.pdcsap_flatten_model_masked)
        self.STD_masked_pdc, self.STD_notmasked_pdc = self.compute_STD(self.pdcsap_flux_flatten, self.pdcsap_flux_flatten_masked, self.pdcmask) 

        if ((1 - self.pdcsap_flux_median_error/self.STD_masked_pdc) or (1 - self.pdcsap_flux_median_error/self.STD_notmasked_pdc)) < tol:
            print('This is worth, give it a look!')
        else:
            print('You may discard this configuration, but give it a look. Just in case...')

        print(f'Median flux error (PDCSAP) =  {self.pdcsap_flux_median_error:.6f}')
        print(f'STD with mask (PDCSAP) =  {self.STD_masked_pdc:.6f}')
        print(f'STD without mask (PDCSAP) =  {self.STD_notmasked_pdc:.6f}')

    def compute_STD(self, flux_flatten, flux_flatten_masked, mask):
        std_masked = np.std(flux_flatten_masked[~mask])
        std_notmasked = np.std(flux_flatten[~mask])
        return std_masked, std_notmasked

    #defining methods to display some plots for visual inspection

    def make_plot_model(self, pdc):
        if pdc == False:
            plt.scatter(self.time - self.time_offset, self.sap_flux, c='C0', s=3)
            plt.errorbar(self.time - self.time_offset, self.sap_flux, yerr=self.sap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.plot(self.time - self.time_offset, self.sap_flatten_model_masked, c='C1')
            plt.ylabel('TESS SAP flux [$e^-/s$]')
        else:
            plt.scatter(self.time - self.time_offset, self.pdcsap_flux, c='C0', s=3)
            plt.errorbar(self.time - self.time_offset, self.pdcsap_flux, yerr=self.pdcsap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.plot(self.time - self.time_offset, self.pdcsap_flatten_model_masked, c='C1')
            plt.ylabel('TESS PDCSAP flux [$e^-/s$]')

        plt.xlabel('BJD_TDB - '+self.offset+' [d]')
        
        plt.show()
    
    def make_plot_normalized(self, pdc):

        if pdc == False:
            plt.scatter(self.time - self.time_offset, self.sap_flux_flatten_masked, c='C0', s=3)
            plt.errorbar(self.time - self.time_offset, self.sap_flux_flatten_masked, yerr=self.sap_flux_err/self.sap_flatten_model, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.ylabel('TESS flattened SAP flux')
        else:
            plt.scatter(self.time - self.time_offset, self.pdcsap_flux_flatten_masked, c='C0', s=3)
            plt.errorbar(self.time - self.time_offset, self.pdcsap_flux_flatten_masked, yerr=self.pdcsap_flux_err/self.pdcsap_flatten_model, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.ylabel('TESS flattened SAP flux')

        plt.xlabel('BJD_TDB - '+self.offset+' [d]')
        
        plt.show()

    def make_plot_folded(self, duration_factor, mask, pdc):
        self.phase_folded_time = (self.time - self.transit_time - self.period/2)%self.period - self.period/2
        if pdc == False:
            if mask == True:
                plt.scatter(self.phase_folded_time[self.mask], self.sap_flux_flatten_masked[self.mask], s=3)
                plt.errorbar(self.phase_folded_time[self.mask], self.sap_flux_flatten_masked[self.mask], yerr=self.sap_flux_err[self.mask]/self.sap_flatten_model[self.mask], ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            else:
                plt.scatter(self.phase_folded_time, self.sap_flux_flatten_masked, s=3)
                plt.errorbar(self.phase_folded_time, self.sap_flux_flatten_masked, yerr=self.sap_flux_err/self.sap_flatten_model, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.ylabel('TESS flattened SAP flux')
        else:
            if mask == True:
                plt.scatter(self.phase_folded_time[self.pdcmask], self.pdcsap_flux_flatten_masked[self.pdcmask], s=3)
                plt.errorbar(self.phase_folded_time[self.pdcmask], self.pdcsap_flux_flatten_masked[self.pdcmask], yerr=self.pdcsap_flux_err[self.pdcmask]/self.pdcsap_flatten_model[self.mask], ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            else:
                plt.scatter(self.phase_folded_time, self.pdcsap_flux_flatten_masked, s=3)
                plt.errorbar(self.phase_folded_time, self.pdcsap_flux_flatten_masked, yerr=self.pdcsap_flux_err/self.pdcsap_flatten_model, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.ylabel('TESS flattened PDCSAP flux')
        
        plt.axvline(-(self.transit_duration*duration_factor)/2, c='C3')
        plt.axvline((self.transit_duration*duration_factor)/2, c='C3')

        plt.xlabel('Orbital phase [d]')
        
        plt.show()

    def make_plot_comparison(self, pdc):
        if pdc == False:
            plt.scatter(self.time - self.time_offset, self.sap_flux, c='C0', s=3)
            plt.errorbar(self.time - self.time_offset, self.sap_flux, yerr=self.sap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.plot(self.time - self.time_offset, self.sap_flatten_model_masked, c='C1', zorder=1, label='Masked')
            plt.plot(self.time - self.time_offset, self.sap_flatten_model, c='C2', zorder=2, label='Unmasked')
            plt.ylabel('TESS SAP flux [$e^-/s$]')
        else:
            plt.scatter(self.time - self.time_offset, self.pdcsap_flux, c='C0', s=3)
            plt.errorbar(self.time - self.time_offset, self.pdcsap_flux, yerr=self.pdcsap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.plot(self.time - self.time_offset, self.pdcsap_flatten_model_masked, c='C1', zorder=1, label='Masked')
            plt.plot(self.time - self.time_offset, self.pdcsap_flatten_model, c='C2', zorder=2, label='Unmasked')
            plt.ylabel('TESS PDCSAP flux [$e^-/s$]')

        plt.legend(loc='best')
        plt.xlabel('BJD_TDB - '+self.offset+' [d]')
        
        plt.show()

    def make_plot_comparison_methods(self, pdc, method='', color=''):
        if pdc == False:
            plt.scatter(self.time - self.time_offset, self.sap_flux, c='C0', s=3, zorder=0)
            plt.errorbar(self.time - self.time_offset, self.sap_flux, yerr=self.sap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.plot(self.time - self.time_offset, self.sap_flatten_model_masked, zorder=1, label=method, c=color)
            plt.ylabel('TESS SAP flux [$e^-/s$]')
        else:
            plt.scatter(self.time - self.time_offset, self.pdcsap_flux, c='C0', s=3, zorder=0)
            plt.errorbar(self.time - self.time_offset, self.pdcsap_flux, yerr=self.pdcsap_flux_err, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
            plt.plot(self.time - self.time_offset, self.pdcsap_flatten_model_masked, zorder=1, label=method, c=color)
            plt.ylabel('TESS PDCSAP flux [$e^-/s$]')

        plt.legend(loc='best')
        plt.xlabel('BJD_TDB - '+self.offset+' [d]')