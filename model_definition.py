import functools

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
import enterprise.signals.signal_base as base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise import constants as const

import enterprise_extensions as ee
from enterprise_extensions import models, model_utils, blocks

def model_general(psrs, tm_var=False, tm_linear=False, tmparam_list=None,
                  tm_svd=False, tm_norm=True, noisedict=None, white_vary=False,
                  Tspan=None, modes=None, wgts=None, logfreq=False, nmodes_log=10,
                  common_psd='powerlaw', common_components=30,
                  log10_A_common=None, gamma_common=None,
                  common_logmin=None, common_logmax=None,
                  orf='crn', orf_names=None, orf_ifreq=0, leg_lmax=5,
                  upper_limit_common=None, upper_limit=False,
                  red_var=True, red_psd='powerlaw', red_components=30, upper_limit_red=None,
                  red_select=None, red_breakflat=False, red_breakflat_fq=None,
                  bayesephem=False, be_type='setIII_1980', is_wideband=False, use_dmdata=False,
                  dm_var=False, dm_type='gp', dm_psd='powerlaw', dm_components=30,
                  upper_limit_dm=None, dm_annual=False, dm_chrom=False, dmchrom_psd='powerlaw',
                  dmchrom_idx=4, gequad=False, coefficients=False, pshift=False,
                  select='backend', tm_marg=False, dense_like=False):
    """
    Reads in list of enterprise Pulsar instances and returns a PTA
    object instantiated with user-supplied options.
    :param tm_var: boolean to vary timing model coefficients.
        [default = False]
    :param tm_linear: boolean to vary timing model under linear approximation.
        [default = False]
    :param tmparam_list: list of timing model parameters to vary.
        [default = None]
    :param tm_svd: stabilize timing model designmatrix with SVD.
        [default = False]
    :param tm_norm: normalize the timing model design matrix, or provide custom
        normalization. Alternative to 'tm_svd'.
        [default = True]
    :param noisedict: Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
        [default = None]
    :param white_vary: boolean for varying white noise or keeping fixed.
        [default = False]
    :param Tspan: timespan assumed for describing stochastic processes,
        in units of seconds. If None provided will find span of pulsars.
        [default = None]
    :param modes: list of frequencies on which to describe red processes.
        [default = None]
    :param wgts: sqrt summation weights for each frequency bin, i.e. sqrt(delta f).
        [default = None]
    :param logfreq: boolean for including log-spaced bins.
        [default = False]
    :param nmodes_log: number of log-spaced bins below 1/T.
        [default = 10]
    :param common_psd: psd of common process.
        ['powerlaw', 'spectrum', 'turnover', 'turnover_knee,', 'broken_powerlaw']
        [default = 'powerlaw']
    :param common_components: number of frequencies starting at 1/T for common process.
        [default = 30]
    :param log10_A_common: value of fixed log10_A_common parameter for
        fixed amplitude analyses.
        [default = None]
    :param gamma_common: fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
        [default = None]
    :param common_logmin: specify lower prior for common psd. This is a prior on log10_rho
        if common_psd is 'spectrum', else it is a prior on log amplitude
    :param common_logmax: specify upper prior for common psd. This is a prior on log10_rho
        if common_psd is 'spectrum', else it is a prior on log amplitude
    :param orf: comma de-limited string of multiple common processes with different orfs.
        [default = crn]
    :param orf_names: comma de-limited string of process names for different orfs. Manual
        control of these names is useful for embedding model_general within a hypermodel
        analysis for a process with and without hd correlations where we want to avoid
        parameter duplication.
        [default = None]
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
        [default = 0]
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function.
        [default = 5]
    :param upper_limit_common: perform upper limit on common red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param upper_limit: apply upper limit priors to all red processes.
        [default = False]
    :param red_var: boolean to switch on/off intrinsic red noise.
        [default = True]
    :param red_psd: psd of intrinsic red process.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt', 'infinitepower']
        [default = 'powerlaw']
    :param red_components: number of frequencies starting at 1/T for intrinsic red process.
        [default = 30]
    :param upper_limit_red: perform upper limit on intrinsic red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param red_select: selection properties for intrinsic red noise.
        ['backend', 'band', 'band+', None]
        [default = None]
    :param red_breakflat: break red noise spectrum and make flat above certain frequency.
        [default = False]
    :param red_breakflat_fq: break frequency for 'red_breakflat'.
        [default = None]
    :param bayesephem: boolean to include BayesEphem model.
        [default = False]
    :param be_type: flavor of bayesephem model based on how partials are computed.
        ['orbel', 'orbel-v2', 'setIII', 'setIII_1980']
        [default = 'setIII_1980']
    :param is_wideband: boolean for whether input TOAs are wideband TOAs. Will exclude
        ecorr from the white noise model.
        [default = False]
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if is_wideband.
        [default = False]
    :param dm_var: boolean for explicitly searching for DM variations.
        [default = False]
    :param dm_type: type of DM variations.
        ['gp', other choices selected with additional options; see below]
        [default = 'gp']
    :param dm_psd: psd of DM GP.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt']
        [default = 'powerlaw']
    :param dm_components: number of frequencies starting at 1/T for DM GP.
        [default = 30]
    :param upper_limit_dm: perform upper limit on DM GP. Note that when perfoming
        upper limits it is recommended that the spectral index also be
        fixed to a specific value.
        [default = False]
    :param dm_annual: boolean to search for an annual DM trend.
        [default = False]
    :param dm_chrom: boolean to search for a generic chromatic GP.
        [default = False]
    :param dmchrom_psd: psd of generic chromatic GP.
        ['powerlaw', 'spectrum', 'turnover']
        [default = 'powerlaw']
    :param dmchrom_idx: spectral index of generic chromatic GP.
        [default = 4]
    :param gequad: boolean to search for a global EQUAD.
        [default = False]
    :param coefficients: boolean to form full hierarchical PTA object;
        (no analytic latent-coefficient marginalization)
        [default = False]
    :param pshift: boolean to add random phase shift to red noise Fourier design
        matrices for false alarm rate studies.
        [default = False]
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    Default PTA object composition:
        1. fixed EFAC per backend/receiver system (per pulsar)
        2. fixed EQUAD per backend/receiver system (per pulsar)
        3. fixed ECORR per backend/receiver system (per pulsar)
        4. Red noise modeled as a power-law with 30 sampling frequencies
           (per pulsar)
        5. Linear timing model (per pulsar)
        6. Common-spectrum uncorrelated process modeled as a power-law with
           30 sampling frequencies. (global)
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'
    gp_priors = [upper_limit_red, upper_limit_dm, upper_limit_common]
    if all(ii is None for ii in gp_priors):
        amp_prior_red = amp_prior
        amp_prior_dm = amp_prior
        amp_prior_common = amp_prior
    else:
        amp_prior_red = 'uniform' if upper_limit_red else 'log-uniform'
        amp_prior_dm = 'uniform' if upper_limit_dm else 'log-uniform'
        amp_prior_common = 'uniform' if upper_limit_common else 'log-uniform'

    # timing model
    if not tm_var and not use_dmdata:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel()
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                       coefficients=coefficients)
    
    # find the maximum time span to set GW frequency sampling
    if Tspan is not None:
        Tspan = Tspan
    else:
        Tspan = model_utils.get_tspan(psrs)

    # common red noise block
    crn = []
    if orf_names is None:
        orf_names = orf
    for elem, elem_name in zip(orf.split(','), orf_names.split(',')):
        if elem == 'zero_diag_bin_orf' or elem == 'zero_diag_legendre_orf':
            log10_A_val = log10_A_common
        else:
            log10_A_val = None
        crn.append(blocks.common_red_noise_block(psd=common_psd, prior=amp_prior_common, Tspan=Tspan,
                                          components=common_components,
                                          log10_A_val=log10_A_val, gamma_val=gamma_common,
                                          delta_val=None, orf=elem, name='gw_{}'.format(elem_name),
                                          orf_ifreq=orf_ifreq, leg_lmax=leg_lmax,
                                          coefficients=coefficients, pshift=pshift, pseed=None,
                                          logmin=common_logmin, logmax=common_logmax))
        # orf_ifreq only affects freq_hd model.
        # leg_lmax only affects (zero_diag_)legendre_orf model.
    crn = functools.reduce((lambda x, y: x+y), crn)
    s += crn

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + blocks.white_noise_block(vary=white_vary, inc_ecorr=True,
                                       gp_ecorr=True, select=select)
            models.append(s2(p))
        else:
            s4 = s + blocks.white_noise_block(vary=white_vary, inc_ecorr=False,
                                   select=select)
            models.append(s4(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    return pta