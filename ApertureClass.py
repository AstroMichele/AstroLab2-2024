import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits

from astropy.time import Time
from astropy import coordinates as coord
from astropy import units as u

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

class AperturePhotometry:
    def __init__(self):
        self.data_path = '../../group08_HAT-P-12_20230214/'

        self.readout_noise = 7.1
        self.gain = 1.91

        self.bias_std = 1.3

        #loading the median bias and its associated error
        self.median_bias = pickle.load(open('../../Results/median_bias.p', 'rb'))
        self.median_bias_error = pickle.load(open('../../Results/median_bias_error.p', 'rb'))

        #loading the median normalized flat and its associated error
        self.median_normalized_flat = pickle.load(open('../../Results/median_normalized_flat.p', 'rb'))
        self.median_normalized_flat_error = pickle.load(open('../../Results/median_normalized_flat_error.p', 'rb'))

        self.science_path = self.data_path + 'science/'
        self.science_list = np.genfromtxt(self.science_path + 'science_list', dtype=str)
        self.science_size = len(self.science_list)

        ylen, xlen = np.shape(self.median_bias)
        self.x_axis = np.arange(0, xlen, 1.)                 #we do not save them to save memory since they are not useful for the analysis
        self.y_axis = np.arange(0, ylen, 1.)
        self.X, self.Y = np.meshgrid(self.x_axis, self.y_axis)

    def aperture_photometry(self, x_coord, y_coord, inner_radius, outer_radius):

        self.airmass = np.empty(self.science_size)
        self.exptime = np.empty(self.science_size)
        self.julian_date = np.empty(self.science_size)
        self.midexp = np.empty(self.science_size)
        self.science_corrected = np.empty([156, 521, self.science_size])
        self.science_sky_corrected = np.empty([156, 521, self.science_size])
        self.photometry = np.empty(self.science_size)
        self.photometry_error = np.empty(self.science_size)
        self.x_refined = np.empty(self.science_size)
        self.y_refined = np.empty(self.science_size)
        self.aperture = np.empty(self.science_size)
        self.sky_background = np.empty(self.science_size)
        self.sky_background_error = np.empty(self.science_size)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.FWHM_X = np.empty(self.science_size)
        self.FWHM_Y = np.empty(self.science_size)

        for i_science, science_name in enumerate(self.science_list):
            science_fits = fits.open(self.science_path + science_name)
            self.airmass[i_science] = science_fits[0].header['AIRMASS']
            self.exptime[i_science] = science_fits[0].header['EXPTIME']
            self.julian_date[i_science] = science_fits[0].header['JD']
            self.midexp[i_science] = science_fits[0].header['JD'] + science_fits[0].header['EXPTIME']/2/86400

            science_data = science_fits[0].data * self.gain
            science_corr, science_corr_error = self.correct_science_frame(science_data)
            self.science_corrected[:, :, i_science] = science_corr

            science_fits.close()

            x_ref, y_ref = self.compute_centroid(science_corr, x_coord, y_coord)
            self.x_refined[i_science] = x_ref
            self.y_refined[i_science] = y_ref
            sky_bkg, sky_bkg_error = self.compute_sky_background(science_corr, science_corr_error, x_ref, y_ref)
            self.sky_background[i_science], self.sky_background_error[i_science] = sky_bkg, sky_bkg_error

        #aperture photometry

            science_sky_corr = science_corr - sky_bkg
            science_sky_corr_error = np.sqrt((sky_bkg*science_corr_error)**2 + (sky_bkg_error*science_corr)**2)
            self.science_sky_corrected[:, :, i_science] = science_sky_corr

            target_distance = np.sqrt((self.X-x_ref)**2 + (self.Y-y_ref)**2)

            inner_selection = (target_distance < inner_radius)
            total_flux = np.sum(science_sky_corr[inner_selection])

            aperture_size = 2
            aperture_selection = (target_distance < aperture_size)
            fractional_flux = np.sum(science_sky_corr[aperture_selection]) / total_flux

            while fractional_flux < 0.9:
                aperture_size += 0.05
                aperture_selection = (target_distance < aperture_size)
                fractional_flux = np.sum(science_sky_corr[aperture_selection]) / total_flux

            self.aperture[i_science] = aperture_size

            self.photometry[i_science] = np.sum(science_sky_corr[aperture_selection])
            self.photometry_error[i_science] = np.sqrt(np.sum(science_sky_corr_error[aperture_selection]**2))

        #FWHM

            total_flux = np.nansum(science_corr*inner_selection)

            flux_x = np.nansum(science_corr*inner_selection, axis=0) 
            flux_y = np.nansum(science_corr*inner_selection, axis=1) 

            cumulative_sum_x = np.cumsum(flux_x)/total_flux
            cumulative_sum_y = np.cumsum(flux_y)/total_flux

            self.FWHM_X[i_science] = self.compute_fwhm(self.x_axis, cumulative_sum_x)
            self.FWHM_Y[i_science] = self.compute_fwhm(self.y_axis, cumulative_sum_y)

        RA, DEC = science_fits[0].header['OBJCTRA'], science_fits[0].header['OBJCTDEC']
        star = coord.SkyCoord(RA, DEC, unit=(u.hourangle, u.deg), frame='icrs')
        time_object = Time(self.midexp, format='jd', scale='utc', location=('45.8472d', '11.569d'))
        self.bjd = (time_object.tdb + time_object.light_travel_time(star, ephemeris='jpl')).to_value('jd')

    def compute_centroid(self, science_frame, x_target_initial, y_target_initial, maximum_number_of_iterations=20):

        for i_iter in range(0, maximum_number_of_iterations):

            if i_iter == 0:
                # first iteration
                x_target_previous = x_target_initial
                y_target_previous = y_target_initial
            else:
                # using the previous result as starting point
                x_target_previous = x_target_refined
                y_target_previous = y_target_refined

            # 2D array with the distance of each pixel from the target star 
            target_distance = np.sqrt((self.X-x_target_previous)**2 + (self.Y-y_target_previous)**2)

            # Selection of the pixels within the inner radius
            annulus_sel = (target_distance < self.inner_radius)

            # Weighted sum of coordinates
            weighted_X = np.sum(science_frame[annulus_sel]*self.X[annulus_sel])
            weighted_Y = np.sum(science_frame[annulus_sel]*self.Y[annulus_sel])

            # Sum of the weights
            total_flux = np.sum(science_frame[annulus_sel])

            # Refined determination of coordinates
            x_target_refined = weighted_X/total_flux
            y_target_refined = weighted_Y/total_flux

            percent_variance_x = (x_target_refined-x_target_previous)/(x_target_previous) * 100.
            percent_variance_y = (y_target_refined-y_target_previous)/(y_target_previous) * 100.
            # exit condition: both percent variance are smaller than 0.1%
            if np.abs(percent_variance_x)<0.1 and  np.abs(percent_variance_y)<0.1:
                  break

        return x_target_refined, y_target_refined
    
    def compute_sky_background(self, science_frame, science_frame_error, x_pos, y_pos):
        target_distance = np.sqrt((self.X-x_pos)**2 + (self.Y-y_pos)**2)

        annulus_selection = (target_distance > self.inner_radius) & (target_distance<=self.outer_radius)

        #sky_flux_average = np.sum(science_frame[annulus_selection]) / np.sum(annulus_selection)
        sky_flux_median = np.median(science_frame[annulus_selection])
        sky_flux_error = np.mean(science_frame_error[annulus_selection])
        
        return sky_flux_median, sky_flux_error

    def correct_science_frame(self, science_data):
        science_debiased = science_data - self.median_bias
        science_corrected = science_debiased / self.median_normalized_flat

        science_debiased_error = np.sqrt(self.readout_noise**2 + science_debiased + self.median_bias_error**2)
        science_corrected_error = science_debiased * np.sqrt((science_debiased_error/science_debiased)**2 + (self.median_normalized_flat/self.median_normalized_flat_error)**2)

        return science_corrected, science_corrected_error

    def compute_fwhm(self, reference_axis, normalized_cumulative_distribution): 
        # Find the closest point to NCD= 0.15865 (-1 sigma)
        NCD_index_left = np.argmin(np.abs(normalized_cumulative_distribution-0.15865))
    
        # Find the closest point to NCD= 0.84135 (+1 sigma)
        NCD_index_right = np.argmin(np.abs(normalized_cumulative_distribution-0.84135))

        # We model the NCD around the -1sgima value with a polynomial curve. 
        # The independet variable is actually the normalized cumulative distribution, 
        # the depedent variable is the pixel position
        p_fitted = np.polynomial.Polynomial.fit(normalized_cumulative_distribution[NCD_index_left-1: NCD_index_left+2],
                                            reference_axis[NCD_index_left-1: NCD_index_left+2],
                                            deg=2)

        # We get a more precise estimate of the pixel value corresponding to the -1sigma position
        pixel_left = p_fitted(0.15865)

        # We repeat the step for the 1sigma value
        p_fitted = np.polynomial.Polynomial.fit(normalized_cumulative_distribution[NCD_index_right-1: NCD_index_right+2],
                                            reference_axis[NCD_index_right-1: NCD_index_right+2],
                                            deg=2)
        pixel_right = p_fitted(0.84135)
    
        FWHM_factor = 2 * np.sqrt(2 * np.log(2)) # = 2.35482
        FWHM = (pixel_right-pixel_left)/2. * FWHM_factor

        return FWHM
