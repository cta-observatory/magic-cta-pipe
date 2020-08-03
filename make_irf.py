# coding: utf-8

import datetime
import yaml
import argparse
import pandas as pd

import scipy
import numpy as np

import iminuit

import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription

import astropy.io.fits as pyfits
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.angle_utilities import angular_separation, position_angle

from matplotlib import pyplot, colors


def info_message(text, prefix='info'):
    """
    This function prints the specified text with the prefix of the current date

    Parameters
    ----------
    text: str

    Returns
    -------
    None

    """

    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"({prefix:s}) {date_str:s}: {text:s}")


class PSFProfileFunctor:
    def __init__(self, r, event_count):
        self.r = r
        self.event_count = event_count
        
        # The function signature to be interpreted by Minuit
        func_args = ('s', 'a2', 'a3', 'sigma1', 'sigma2', 'sigma3')
        self.__code__ = iminuit.util.make_func_code(func_args)
        
        # The following keeps np.vectorize happy
        self.__defaults__ = None

    def __call__(self, s, a2, a3, sigma1, sigma2, sigma3):
        return self.cstat_loss(s, a2, a3, sigma1, sigma2, sigma3)

    @staticmethod
    def psf_profile(r, s, a2, a3, sigma1, sigma2, sigma3):
        g1 = np.exp(-r**2 / (2 * sigma1**2))
        g2 = np.exp(-r**2 / (2 * sigma2**2))
        g3 = np.exp(-r**2 / (2 * sigma3**2))
        
        dn_domega = s / np.pi * (g1 + a2*g2 + a3*g3)
    
        return 2*np.pi * dn_domega
    
    @staticmethod
    def cstat(y, model_y, mode="Exact"):
        """
        A function that computes the C-statistics (Poissonian log-like) value of given data with respect to a given model.

        Parameters
        ----------
        y: array_like
            Data array.
        model_y: array_like
            Model array.
        mode: str, optional
            Defines the mode of calculation:
                - "Normalized": 2 x loglike will be returned.
                - "Chi2-like": 2 x loglike with an additional term subtracted, which brings the computed value close
                to chi2 distribution in the limit of large y and model_y.
                - any other string: a loglike value will be returned

        Returns
        -------
        array_like:
            The computed C-statistics values.
        """

        res = -1 * np.sum(y*np.lib.scimath.log(model_y) - model_y - scipy.special.gammaln(y+1))

        if mode == "Normalized":
            res *= 2

        if mode == "Chi2-like":
            #res = 2*res - np.sum(np.lib.scimath.log(2*np.pi*y))
            res = 2*res - np.sum(np.lib.scimath.log(2*np.pi*model_y))

        return res
   
        
    def mse_loss(self, s, a2, a3, sigma1, sigma2, sigma3):
        delta = self.event_count - self.psf_profile(self.r, s, a2, a3, sigma1, sigma2, sigma3)
        
        return (delta**2).sum()
    
    def cstat_loss(self, s, a2, a3, sigma1, sigma2, sigma3):
        model = self.psf_profile(self.r, s, a2, a3, sigma1, sigma2, sigma3)
        cs = self.cstat(self.event_count, model)
        
        return cs.sum()


class IRFGenerator:
    def __init__(self, mc_file_name):
        self.trig_shower_data = pd.read_hdf(mc_file_name, key='dl3/reco')
        self.sim_shower_data = pd.read_hdf(mc_file_name, key='dl3/original_mc')
        
        self.cuts = None
        
        self.min_energy = None
        self.max_energy = None
        self.n_energy_bins = None
        
        self.min_theta = None
        self.max_theta = None
        self.n_theta_bins = None
        
        self.min_migra = None
        self.max_migra = None
        self.n_migra_bins = None

    def set_cuts(self, cuts):
        self.cuts = cuts
    
    def set_energy_binning(self, min_energy, max_energy, n_energy_bins):
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.n_energy_bins = n_energy_bins
        
    def set_theta_binning(self, min_theta, max_theta, n_theta_bins):
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.n_theta_bins = n_theta_bins
        
    def set_migra_binning(self, min_migra, max_migra, n_migra_bins):
        self.min_migra = min_migra
        self.max_migra = max_migra
        self.n_migra_bins = n_migra_bins
    
    def _generate_psf_hdu(self):
        trig_shower_data = self.trig_shower_data.query(self.cuts)
        
        # Computing reconstruction offset angle
        offset = angular_separation(trig_shower_data['true_az'].values * u.rad,
                                    trig_shower_data['true_alt'].values * u.rad,
                                    trig_shower_data['az_reco_mean'].values * u.rad,
                                    trig_shower_data['alt_reco_mean'].values * u.rad)

        offset = offset.to(u.deg)
        
        # Computing camera off-center angle
        offcenter = angular_separation(trig_shower_data['true_az'].values * u.rad,
                                       trig_shower_data['true_alt'].values * u.rad,
                                       trig_shower_data['tel_az'].values * u.rad,
                                       trig_shower_data['tel_alt'].values * u.rad)

        offcenter = offcenter.to(u.deg)
        
        data = trig_shower_data.loc[slice(None), ['true_energy']]
        data['offset'] = offset
        data['offcenter'] = offcenter
        
        # Binning in energy
        energy_edges = np.logspace(np.lib.scimath.log10(self.min_energy), 
                                      np.lib.scimath.log10(self.max_energy), 
                                      self.n_energy_bins+1)
        energ_lo = energy_edges[:-1]
        energ_hi = energy_edges[1:]
        
        # Binning in off-center distance
        theta_edges = np.linspace(self.min_theta, 
                                     self.max_theta, 
                                     self.n_theta_bins+1)
        theta_lo = theta_edges[:-1]
        theta_hi = theta_edges[1:]
        
        # ----------------------
        # --- Evaluating PSF ---

        psf_params = dict()

        for param in ['s', 'a2', 'a3', 'sigma1', 'sigma2', 'sigma3']:
            psf_params[param] = np.zeros((self.n_energy_bins, self.n_theta_bins))
        
        fit_params = {
            's': 1e3,
            'a2': 0.01,
            'a3': 0,
            'sigma1': 0.1,
            'sigma2': 0.3,
            'sigma3': 0.1,
            
            'limit_s': (0, None),
            'limit_a2': (0, 0.1),
            'limit_a3': (0, 0.1),
            'limit_sigma1': (0, 1),
            'limit_sigma2': (0, 1),
            'limit_sigma3': (0, 1),
            
            'fix_a2': False,
            'fix_a3': True,
            'fix_sigma2': False,
            'fix_sigma3': True,
        }
        
        # PSF histogram grid
        offset_edges = np.linspace(0, 4, num=100)**0.5
        offset_centers = (offset_edges[1:] + offset_edges[:-1]) / 2
        
        for ei in range(self.n_energy_bins):
            for ti in range(self.n_theta_bins):
                energy_filter = f'(true_energy >= {energ_lo[ei]:.3e}) & (true_energy < {energ_hi[ei]:.3e})'
                theta_filter = f'(offcenter >= {theta_lo[ti]:.3e}) & (offcenter < {theta_hi[ti]:.3e})'
                event_filter = f'({energy_filter}) & ({theta_filter})'
                events = data.query(event_filter)
            
                psf_hist, _ = np.histogram(events['offset'], bins=offset_edges)
                
                fit_func = PSFProfileFunctor(offset_centers, psf_hist)
                
                fit_obj = iminuit.Minuit(fit_func, pedantic=False, print_level=0,
                                        **fit_params)
                fit_obj.migrad()
                
                for key in psf_params:
                    psf_params[key][ei, ti] = fit_obj.values[key]
                
                psf_params['s'][ei, ti] /= psf_hist.sum()

        # ----------------------
        
        # --------------------------
        # --- Converting to FITS ---
        col_energ_lo = pyfits.Column(name='ENERG_LO', unit='TeV', format=f'{energ_lo.size}E', array=[energ_lo])
        col_energ_hi = pyfits.Column(name='ENERG_HI', unit='TeV', format=f'{energ_hi.size}E', array=[energ_hi])
        col_theta_lo = pyfits.Column(name='THETA_LO', unit='deg', format=f'{theta_lo.size}E', array=[theta_lo])
        col_theta_hi = pyfits.Column(name='THETA_HI', unit='deg', format=f'{theta_hi.size}E', array=[theta_hi])

        col_scale = pyfits.Column(name='SCALE', unit='', format=f"{psf_params['s'].size:d}E", 
                                  array=[psf_params['s'].transpose()],
                                  dim=str(psf_params['s'].shape))

        col_ampl2 = pyfits.Column(name='AMPL_2', unit='', format=f"{psf_params['a2'].size:d}E", 
                                  array=[psf_params['a2'].transpose()],
                                  dim=str(psf_params['a2'].shape))

        col_ampl3 = pyfits.Column(name='AMPL_3', unit='', format=f"{psf_params['a3'].size:d}E", 
                                  array=[psf_params['a3'].transpose()],
                                  dim=str(psf_params['a3'].shape))

        col_sigma1 = pyfits.Column(name='SIGMA_1', unit='deg', format=f"{psf_params['sigma1'].size:d}E", 
                                   array=[psf_params['sigma1'].transpose()],
                                   dim=str(psf_params['sigma1'].shape))

        col_sigma2 = pyfits.Column(name='SIGMA_2', unit='deg', format=f"{psf_params['sigma2'].size:d}E", 
                                   array=[psf_params['sigma2'].transpose()],
                                   dim=str(psf_params['sigma2'].shape))

        col_sigma3 = pyfits.Column(name='SIGMA_3', unit='deg', format=f"{psf_params['sigma3'].size:d}E", 
                                   array=[psf_params['sigma3'].transpose()],
                                   dim=str(psf_params['sigma3'].shape))
        
        columns = [
            col_energ_lo, 
            col_energ_hi,
            col_theta_lo,
            col_theta_hi,
            col_scale,
            col_sigma1,
            col_ampl2,
            col_sigma2,
            col_ampl3,
            col_sigma3,
        ]

        # Creating HDU
        colDefs = pyfits.ColDefs(columns)
        psf_hdu = pyfits.BinTableHDU.from_columns(colDefs)
        psf_hdu.name = 'POINT SPREAD FUNCTION'
        
        psf_hdu.header['HDUDOC'] = 'https://github.com/open-gamma-ray-astro/gamma-astro-data-formats'
        psf_hdu.header['HDUVERS'] = '0.2'
        psf_hdu.header['HDUCLASS'] = 'GADF'
        psf_hdu.header['HDUCLAS1'] = 'RESPONSE'
        psf_hdu.header['HDUCLAS2'] = 'PSF'
        psf_hdu.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        psf_hdu.header['HDUCLAS4'] = 'PSF_3GAUSS'
        # --------------------------
        
        return psf_hdu
    
    def _generate_edisp_hdu(self):
        trig_shower_data = self.trig_shower_data.query(self.cuts)
        
        # Computing camera off-center angle
        offcenter = angular_separation(trig_shower_data['true_az'].values * u.rad,
                                       trig_shower_data['true_alt'].values * u.rad,
                                       trig_shower_data['tel_az'].values * u.rad,
                                       trig_shower_data['tel_alt'].values * u.rad)

        offcenter = offcenter.to(u.deg)

        # Energy migration
        migra = trig_shower_data['energy_reco_mean'] / trig_shower_data['true_energy']
        
        data = trig_shower_data.loc[slice(None), ['true_energy']]
        data['migra'] = migra
        data['offcenter'] = offcenter
        
        # Binning in energy
        energy_edges = np.logspace(np.lib.scimath.log10(self.min_energy), 
                                      np.lib.scimath.log10(self.max_energy), 
                                      self.n_energy_bins+1)
        energ_lo = energy_edges[:-1]
        energ_hi = energy_edges[1:]
        
        # Binning in off-center distance
        theta_edges = np.linspace(self.min_theta, 
                                     self.max_theta, 
                                     self.n_theta_bins+1)
        theta_lo = theta_edges[:-1]
        theta_hi = theta_edges[1:]
        
        # Binning in "migra" value
        migra_edges = np.logspace(np.lib.scimath.log10(self.min_migra), 
                                     np.lib.scimath.log10(self.max_migra),
                                     self.n_migra_bins+1)
        migra_lo = migra_edges[:-1]
        migra_hi = migra_edges[1:]
        
        # Computing the migration matrix
        data_ = [
            data['true_energy'].values,
            data['migra'].values,
            data['offcenter'].values,
        ]

        edges_ = [
            energy_edges,
            migra_edges,
            theta_edges
        ]

        migra_matrix, _ = np.histogramdd(data_, bins=edges_)
        
        # Normalizing the matrix
        migra_matrix_norms = migra_matrix.sum(axis=1)
        migra_matrix /= migra_matrix_norms[:, None, :]
        
        isnan = np.isnan(migra_matrix)
        migra_matrix[isnan] = 0
        
        # --------------------------
        # --- Converting to FITS ---
        col_energ_lo = pyfits.Column(name='ENERG_LO', unit='TeV', format=f'{energ_lo.size}E', array=[energ_lo])
        col_energ_hi = pyfits.Column(name='ENERG_HI', unit='TeV', format=f'{energ_hi.size}E', array=[energ_hi])
        col_theta_lo = pyfits.Column(name='THETA_LO', unit='deg', format=f'{theta_lo.size}E', array=[theta_lo])
        col_theta_hi = pyfits.Column(name='THETA_HI', unit='deg', format=f'{theta_hi.size}E', array=[theta_hi])
        col_migra_lo = pyfits.Column(name='MIGRA_LO', unit='', format=f'{migra_lo.size}E', array=[migra_lo])
        col_migra_hi = pyfits.Column(name='MIGRA_HI', unit='', format=f'{migra_hi.size}E', array=[migra_hi])

        col_migra_matrix = pyfits.Column(name='MATRIX', unit='', format=f"{migra_matrix.size:d}E", 
                                         array=[migra_matrix.transpose()],
                                         dim=str(migra_matrix.shape))
        
        columns = [
            col_energ_lo, 
            col_energ_hi,
            col_theta_lo,
            col_theta_hi,
            col_migra_lo,
            col_migra_hi,
            col_migra_matrix
        ]

        # Migration matrix HDU
        colDefs = pyfits.ColDefs(columns)
        migra_hdu = pyfits.BinTableHDU.from_columns(colDefs)
        migra_hdu.name = 'ENERGY DISPERSION'
        
        migra_hdu.header['HDUDOC'] = 'https://github.com/open-gamma-ray-astro/gamma-astro-data-formats'
        migra_hdu.header['HDUVERS'] = '0.2'
        migra_hdu.header['HDUCLASS'] = 'GADF'
        migra_hdu.header['HDUCLAS1'] = 'RESPONSE'
        migra_hdu.header['HDUCLAS2'] = 'EDISP'
        migra_hdu.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        migra_hdu.header['HDUCLAS4'] = 'EDISP_2D'
        # --------------------------
        
        return migra_hdu
            
    def _generate_aeff_hdu(self):
        trig_shower_data = self.trig_shower_data.query(self.cuts)
        
        # Computing camera off-center angle for triggered events
        offcenter = angular_separation(trig_shower_data['true_az'].values * u.rad,
                                       trig_shower_data['true_alt'].values * u.rad,
                                       trig_shower_data['tel_az'].values * u.rad,
                                       trig_shower_data['tel_alt'].values * u.rad)

        offcenter = offcenter.to(u.deg)
        
        trig_shower_data = trig_shower_data.loc[slice(None), ['true_energy']]
        trig_shower_data['offcenter'] = offcenter
        
        # Computing camera off-center angle for all simulated events
        offcenter = angular_separation(self.sim_shower_data['true_az'].values * u.rad,
                                       self.sim_shower_data['true_alt'].values * u.rad,
                                       self.sim_shower_data['tel_az'].values * u.rad,
                                       self.sim_shower_data['tel_alt'].values * u.rad)

        offcenter = offcenter.to(u.deg)
        
        sim_shower_data = self.sim_shower_data.loc[slice(None), ['true_energy']]
        sim_shower_data['offcenter'] = offcenter
        
        # Binning in energy
        energy_edges = np.logspace(np.lib.scimath.log10(self.min_energy), 
                                      np.lib.scimath.log10(self.max_energy), 
                                      self.n_energy_bins+1)
        energ_lo = energy_edges[:-1]
        energ_hi = energy_edges[1:]
        
        # Binning in off-center distance
        theta_edges = np.linspace(self.min_theta, 
                                     self.max_theta, 
                                     self.n_theta_bins+1)
        theta_lo = theta_edges[:-1]
        theta_hi = theta_edges[1:]
        
        trig_events_matrix, _, _ = np.histogram2d(trig_shower_data['true_energy'].values, 
                                                     trig_shower_data['offcenter'].values, 
                                                     bins=[energy_edges, theta_edges])
        
        sim_events_matrix, _, _ = np.histogram2d(sim_shower_data['true_energy'].values, 
                                                    sim_shower_data['offcenter'].values, 
                                                    bins=[energy_edges, theta_edges])
   
        # add and mod. by Y.Suda on 2020.02.17
        ntel = self.sim_shower_data['multiplicity'][0]
        #efficiency_matrix = trig_events_matrix / sim_events_matrix
        efficiency_matrix = trig_events_matrix / sim_events_matrix * ntel
        
        r_sim = 350.0  # m^2
        aeff_matrix = np.pi * r_sim**2 * efficiency_matrix
        
        # --------------------------
        # --- Converting to FITS ---
        col_energ_lo = pyfits.Column(name='ENERG_LO', unit='TeV', format=f'{energ_lo.size}E', array=[energ_lo])
        col_energ_hi = pyfits.Column(name='ENERG_HI', unit='TeV', format=f'{energ_hi.size}E', array=[energ_hi])
        col_theta_lo = pyfits.Column(name='THETA_LO', unit='deg', format=f'{theta_lo.size}E', array=[theta_lo])
        col_theta_hi = pyfits.Column(name='THETA_HI', unit='deg', format=f'{theta_hi.size}E', array=[theta_hi])

        col_aeff_matrix = pyfits.Column(name='EFFAREA', unit='m^2', format=f"{aeff_matrix.size}E", 
                                        array=[aeff_matrix.transpose()],
                                        dim=str(aeff_matrix.shape))
        
        columns = [
            col_energ_lo, 
            col_energ_hi,
            col_theta_lo,
            col_theta_hi,
            col_aeff_matrix
        ]

        # Aeff HDU
        colDefs = pyfits.ColDefs(columns)
        aeff_hdu = pyfits.BinTableHDU.from_columns(colDefs)
        aeff_hdu.name = 'EFFECTIVE AREA'
        
        aeff_hdu.header['HDUDOC'] = 'https://github.com/open-gamma-ray-astro/gamma-astro-data-formats'
        aeff_hdu.header['HDUVERS'] = '0.2'
        aeff_hdu.header['HDUCLASS'] = 'GADF'
        aeff_hdu.header['HDUCLAS1'] = 'RESPONSE'
        aeff_hdu.header['HDUCLAS2'] = 'EFF_AREA'
        aeff_hdu.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        aeff_hdu.header['HDUCLAS4'] = 'AEFF_2D'
        # --------------------------
        
        return aeff_hdu
    
    def _generate_bkg_hdu(self):
        bkg_shower_data = self.bkg_shower_data.query(self.cuts)

        # Compute elapsed observation time
        elapsed_time = np.array([])
        obs_id_list = np.array(bkg_shower_data.index.levels[0])

        for obs_item in obs_id_list:
            obs_item_events = bkg_shower_data.loc[(obs_item, slice(None), slice(None))]
            obs_event_mean_arr_time = obs_item_events.groupby(['obs_id', 'event_id'])['mjd'].mean()

            time_diff = np.diff(obs_event_mean_arr_time)*u.day.to(u.s)
            # excludes gaps of possible technical problems 
            time_diff = time_diff[np.where(time_diff < 3e-1)]

            elapsed_time = np.append(elapsed_time, np.sum(time_diff))

        elapsed_time = np.sum(elapsed_time)

        # Computing camera off-center angle for background events
        offcenter = angular_separation(bkg_shower_data['az_reco_mean'].values * u.rad,
                                       bkg_shower_data['alt_reco_mean'].values * u.rad,
                                       bkg_shower_data['tel_az'].values * u.rad,
                                       bkg_shower_data['tel_alt'].values * u.rad)
        offcenter = offcenter.to(u.deg)

        bkg_shower_data = bkg_shower_data.loc[slice(None), ['energy_reco_mean']]
        bkg_shower_data['offcenter'] = offcenter
        
        # Binning in energy
        energy_edges = np.logspace(np.lib.scimath.log10(self.min_energy), 
                                      np.lib.scimath.log10(self.max_energy), 
                                      self.n_energy_bins+1)
        energ_lo = energy_edges[:-1]
        energ_hi = energy_edges[1:]
        
        # Binning in off-center distance
        theta_edges = np.linspace(self.min_theta, 
                                     self.max_theta, 
                                     self.n_theta_bins+1)
        theta_lo = theta_edges[:-1]
        theta_hi = theta_edges[1:]
        
        bkg_event_matrix, _, _ = np.histogram2d(bkg_shower_data['energy_reco_mean'].values, 
                                                  bkg_shower_data['offcenter'].values, 
                                                  bins=[energy_edges, theta_edges])
        
        # Compute bin sizes for density
        theta_area   = np.pi * np.diff(theta_edges**2)
        energy_width = np.diff(energy_edges)

        bkg_matrix = bkg_event_matrix / elapsed_time / theta_area / energy_width.reshape((-1, 1))

        # --------------------------
        # --- Converting to FITS ---
        col_energ_lo = pyfits.Column(name='ENERG_LO', unit='TeV', format=f'{energ_lo.size}E', array=[energ_lo])
        col_energ_hi = pyfits.Column(name='ENERG_HI', unit='TeV', format=f'{energ_hi.size}E', array=[energ_hi])
        col_theta_lo = pyfits.Column(name='THETA_LO', unit='deg', format=f'{theta_lo.size}E', array=[theta_lo])
        col_theta_hi = pyfits.Column(name='THETA_HI', unit='deg', format=f'{theta_hi.size}E', array=[theta_hi])

        col_bkg_matrix = pyfits.Column(name='BKG', unit='s^-1 MeV^-1 sr^-1', format=f"{bkg_matrix.size}E", 
                                        array=[bkg_matrix.transpose()],
                                        dim=str(bkg_matrix.shape))
        
        columns = [
            col_energ_lo, 
            col_energ_hi,
            col_theta_lo,
            col_theta_hi,
            col_bkg_matrix
        ]

        # Aeff HDU
        colDefs = pyfits.ColDefs(columns)
        bkg_hdu = pyfits.BinTableHDU.from_columns(colDefs)
        bkg_hdu.name = 'BACKGROUND'
        
        bkg_hdu.header['HDUDOC'] = 'https://github.com/open-gamma-ray-astro/gamma-astro-data-formats'
        bkg_hdu.header['HDUVERS'] = '0.2'
        bkg_hdu.header['HDUCLASS'] = 'GADF'
        bkg_hdu.header['HDUCLAS1'] = 'RESPONSE'
        bkg_hdu.header['HDUCLAS2'] = 'BKG'
        bkg_hdu.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        bkg_hdu.header['HDUCLAS4'] = 'BKG_2D'
        # --------------------------
        
        return bkg_hdu

    def generate_irf(self, output_name):
        info_message('PSF HDU...', prefix='IRFGen')
        psf_hdu = self._generate_psf_hdu()
        
        info_message('EDISP HDU...', prefix='IRFGen')
        edisp_hdu = self._generate_edisp_hdu()
        
        info_message('AEFF HDU...', prefix='IRFGen')
        aeff_hdu = self._generate_aeff_hdu()
        
        #info_message('BACKGROUND HDU...', prefix='IRFGen')
        #bkg_hdu = self._generate_background_hdu()

        primary_hdu = pyfits.PrimaryHDU()
        
        #hdu_list = pyfits.HDUList([primary_hdu, aeff_hdu, psf_hdu, edisp_hdu, bkg_hdu])
        hdu_list = pyfits.HDUList([primary_hdu, aeff_hdu, psf_hdu, edisp_hdu])
        hdu_list.writeto(output_name, overwrite=True)


# =================
# === Main code ===
# =================

# --------------------------
# Adding the argument parser
arg_parser = argparse.ArgumentParser(description="""
This tools prepares IRFs based on the processed "test" MC files.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')

parsed_args = arg_parser.parse_args()
# --------------------------

# ------------------------------
# Reading the configuration file

file_not_found_message = """
Error: can not load the configuration file {:s}.
Please check that the file exists and is of YAML or JSON format.
Exiting.
"""

try:
    config = yaml.safe_load(open(parsed_args.config, "r"))
except IOError:
    print(file_not_found_message.format(parsed_args.config))
    exit()
# ------------------------------

# -----------------
# MAGIC definitions
# MAGIC telescope positions in m wrt. to the center of CTA simulations
magic_tel_positions = {
    1: [-27.24, -146.66, 50.00] * u.m,
    2: [-96.44, -96.77, 51.00] * u.m
}

# MAGIC telescope description
magic_optics = OpticsDescription.from_name('MAGIC')
magic_cam = CameraGeometry.from_name('MAGICCam')
magic_tel_description = TelescopeDescription(name='MAGIC', 
                                             tel_type='MAGIC', 
                                             optics=magic_optics, 
                                             camera=magic_cam)
magic_tel_descriptions = {1: magic_tel_description, 
                          2: magic_tel_description}
# -----------------

mc_file_name = config['data_files']['mc']['test_sample']['magic1']['reco_output']
irf_generator = IRFGenerator(mc_file_name)

irf_generator.set_energy_binning(min_energy=0.1, max_energy=30, n_energy_bins=10)
irf_generator.set_theta_binning(min_theta=0.0, max_theta=1.5, n_theta_bins=5)
irf_generator.set_migra_binning(min_migra=0.2, max_migra=5.0, n_migra_bins=5)

irf_generator.set_cuts(config['event_list']['cuts']['selection'])

irf_generator.generate_irf('crab_irf.fits')

## Looping over MC / data etc
#for data_type in config['data_files']:
    ## Using only the "test" sample
    #for sample in ['test_sample']:        
        #shower_data = pd.DataFrame()
        
        ## Reading data of all available telescopes and join them together
        #for telescope in config['data_files'][data_type][sample]:
            
            #info_message(f'Data "{data_type}", sample "{sample}", telescope "{telescope}"',
                         #prefix='ApplyRF')
