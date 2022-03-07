from __future__ import division

import os, time, sys

import numpy as np
import scipy
import scipy.linalg as sl
import acor

import enterprise
from enterprise.signals import selections


class PTABlockGibbs(object):
    
    """Gibbs-based pulsar-timing periodogram analysis.
    
    Based on:
    
        Article by van Haasteren & Vallisneri (2014),
        "New advances in the Gaussian-process approach 
        to pulsar-timing data analysis",
        Physical Review D, Volume 90, Issue 10, id.104012
        arXiv:1407.1838
        
        Code based on https://github.com/jellis18/gibbs_student_t
        
    Authors: 
    
        S. R. Taylor
    
    Example usage:
    
        > gibbs = PTABlockGibbs(pta, hypersample='conditional', ecorrsample='mh', psr=psr)
        > x0 = x0 = np.concatenate([p.sample().flatten() for p in gibbs.params])
        > gibbs.sample(x0, outdir='./', 
                       niter=10000, resume=False)
        
     
    """
    
    def __init__(self, pta, hypersample='conditional', 
                 redsample='mh'):
        """
        Parameters
        -----------
        pta : object
            instance of a pta object for a single pulsar
        hypersample: string
            method to draw free spectral coefficients from conditional posterior
            ('conditional' = analytic; 'mh' = short MCMC chain)
        ecorrsample: string
            method to draw ECORR coefficients from conditional posterior
            ('conditional' = analytic; 'mh' = short MCMC chain)
        psr: enterprise pulsar object
            pass an enterprise pulsar object to get ecorr selections
        """

        self.pta = pta
        self.hypersample = hypersample
        self.redsample = redsample
        if np.any(['basis_ecorr' in key for 
                   key in self.pta._signal_dict.keys()]):
            pass
        else:
            print('ERROR: Gibbs outlier analysis must use basis_ecorr, not kernel ecorr')

        # For now assume one pulsar
        self._residuals = self.pta.get_residuals()

        # auxiliary variable stuff
        xs = [p.sample() for p in pta.params]
        self._b = [np.zeros(self.pta.get_basis(xs)[ii].shape[1]) 
                    for ii in range(len(self.pta.pulsars))]

        # for caching
        self.TNT = []
        self.d = []
        
        # grabbing priors on gw free spectral power values
        for ct, par in enumerate([p.name for p in self.params]):
            if 'rho' in par and 'gw' in par: ind = ct
        rho_priors = str(self.params[ind].params[0])
        rho_priors = rho_priors.split('(')[1].split(')')[0].split(', ')
        self.rhomin_gw, self.rhomax_gw = (10**(2*float(rho_priors[0].split('=')[1])), 
                                          10**(2*float(rho_priors[1].split('=')[1])))

        # grabbing priors on gw free spectral power values
        for ct, par in enumerate([p.name for p in self.params]):
            if 'rho' in par and 'red' in par: ind = ct
        rho_priors = str(self.params[ind].params[0])
        rho_priors = rho_priors.split('(')[1].split(')')[0].split(', ')
        self.rhomin_red, self.rhomax_red = (10**(2*float(rho_priors[0].split('=')[1])), 
                                            10**(2*float(rho_priors[1].split('=')[1])))
        
        # find basis indices of GW process
        self.gwid = []
        for pname in self.pta.pulsars:
            ct = 0
            psigs = [sig for sig in self.pta.signals.keys() if pname in sig]
            for sig in psigs:
                Fmat = self.pta.signals[sig].get_basis()
                if 'gw' in self.pta.signals[sig].name:
                    self.gwid.append(ct + np.arange(0,Fmat.shape[1]))

                # Avoid None-basis processes.
                # Also assume red + GW signals share basis.
                if Fmat is not None and 'red' not in sig:
                    ct += Fmat.shape[1]

        self.red_sig = []
        self.gw_sig = None
        for sig in self.pta.signals:
            if 'red' in self.pta.signals[sig].name:
                self.red_sig.append(self.pta.signals[sig])
            if 'gw' in self.pta.signals[sig].name:
                self.gw_sig = self.pta.signals[sig]

                       
    @property
    def params(self):
        ret = []
        for param in self.pta.params:
            ret.append(param)
        return ret
    
    @property
    def param_names(self):
        ret = []
        for p in self.params:
            if p.size:
                for ii in range(0, p.size):
                    ret.append(p.name + "_{}".format(ii))
            else:
                ret.append(p.name)
        return ret
    
    def map_params(self, xs):
        ret = {}
        ct = 0
        for p in self.params:
            n = p.size if p.size else 1
            ret[p.name] = xs[ct : ct + n] if n > 1 else float(xs[ct])
            ct += n
        return ret


    def get_rho_param_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if 'rho' in par and 'gw' in par:
                ind.append(ct)
        return np.array(ind)


    def get_hyper_param_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if 'red' in par:
                if 'log10_A' in par or 'gamma' in par or 'rho' in par:
                    ind.append(ct)
        return np.array(ind)


    def get_efacequad_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if 'efac' in par or 'equad' in par:
                ind.append(ct)
        return np.array(ind)
    

    def get_ecorr_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if 'ecorr' in par:
                ind.append(ct)
        return np.array(ind)


    def update_rho_params(self, xs):

        # get hyper parameter indices
        rind = self.get_rho_param_indices()

        xnew = xs.copy()
        if self.hypersample == 'conditional':

            rho_tmp = 10**np.linspace(np.log10(self.rhomin_gw), 
                                      np.log10(self.rhomax_gw), 1000)
            pdf = np.zeros((len(rind),len(rho_tmp),len(self.pta.pulsars)))
            for ii in range(len(self.pta.pulsars)):
            
                tau = self._b[ii][self.gwid[ii]]**2
                tau = tau[::2] + tau[1::2]

                if self.red_sig:
                    irn = np.array(self.red_sig[ii].get_phi(self.map_params(xnew)))[::2]
                else:
                    irn = np.zeros(tau.shape[0])
                
                ratio = tau[:,None] / np.add.outer(irn, rho_tmp)
                pdf[:,:,ii] = ratio * np.exp(-ratio/2) * np.log(10)

            pdf = np.prod(pdf,axis=2)
            cdf = np.cumsum(pdf,axis=1)
            cdf /= cdf.max(axis=1)[:,None]

            u = np.random.uniform(size=cdf.shape[0])
            idx = np.array([np.searchsorted(cdf[ii,:], u[ii], side='left') 
                            for ii in range(u.shape[0])]) - 1
            rhonew = np.take_along_axis(rho_tmp, idx, axis=0)
            
            xnew[rind] = 0.5*np.log10(rhonew)
          
        else:
            # get initial log-likelihood and log-prior
            lnlike0, lnprior0 = self.get_lnlikelihood(xs), self.get_lnprior(xs)
            
            for ii in range(10):

                # standard gaussian jump (this allows for different step sizes)
                q = xnew.copy()
                sigmas = 0.05 * len(rind)
                probs = [0.1, 0.15, 0.5, 0.15, 0.1]
                sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
                scale = np.random.choice(sizes, p=probs)
                par = np.random.choice(rind, size=1) # only one hyper param at a time
                q[par] += np.random.randn(len(q[par])) * sigmas * scale

                # get log-like and log prior at new position
                lnlike1, lnprior1 = self.get_lnlikelihood(q), self.get_lnprior(q)

                # metropolis step
                diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
                if diff > np.log(np.random.rand()):
                    xnew = q
                    lnlike0 = lnlike1
                    lnprior0 = lnprior1
                else:
                    xnew = xnew
        
        return xnew


    def update_hyper_params(self, xs, iters=None):

        # get hyper parameter indices
        hind = self.get_hyper_param_indices()

        xnew = xs.copy()
        if self.redsample == 'conditional':

            rho_red_tmp = 10**np.linspace(np.log10(self.rhomin_red), 
                                          np.log10(self.rhomax_red), 1000)
            rhonew = []
            for ii in range(len(self.pta.pulsars)):
            
                tau = self._b[ii][self.gwid[ii]]**2
                tau = tau[::2] + tau[1::2]

                # should each pulsar call its own gw_sig?
                gw = np.array(self.gw_sig.get_phi(self.map_params(xnew)))[::2]
                
                ratio = tau[:,None] / np.add.outer(gw, rho_red_tmp)
                pdf = ratio * np.exp(-ratio/2) * np.log(10)

                cdf = np.cumsum(pdf,axis=1)
                cdf /= cdf.max(axis=1)[:,None]

                u = np.random.uniform(size=cdf.shape[0])
                idx = np.array([np.searchsorted(cdf[ii,:], u[ii], side='left') 
                                for ii in range(u.shape[0])]) - 1
                rhonew.append(np.take_along_axis(rho_red_tmp, idx, axis=0))
                
            xnew[hind] = 0.5*np.log10(np.concatenate(rhonew))

        else:
        
            # get initial log-likelihood and log-prior
            lnlike0, lnprior0 = self.get_lnlikelihood(xs), self.get_lnprior(xs)

            if iters is not None:
                
                short_chain = np.zeros((iters,len(hind)))
                for ii in range(iters):

                    # standard gaussian jump (this allows for different step sizes)
                    q = xnew.copy()
                    sigmas = 0.05 * len(hind)
                    probs = [0.1, 0.15, 0.5, 0.15, 0.1]
                    sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
                    scale = np.random.choice(sizes, p=probs)
                    par = np.random.choice(hind, size=1) # only one hyper param at a time
                    q[par] += np.random.randn(len(q[par])) * sigmas * scale

                    # get log-like and log prior at new position
                    lnlike1, lnprior1 = self.get_lnlikelihood(q), self.get_lnprior(q)

                    # metropolis step
                    diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
                    if diff > np.log(np.random.rand()):
                        xnew = q
                        lnlike0 = lnlike1
                        lnprior0 = lnprior1
                    else:
                        xnew = xnew

                    short_chain[ii,:] = q[hind]

                self.cov_hyper = np.cov(short_chain[100:,:],rowvar=False)
                self.sigma_hyper = np.diag(self.cov_hyper)**0.5
                self.svd_hyper = np.linalg.svd(self.cov_hyper)
                self.aclength_hyper = int(np.max([int(acor.acor(short_chain[100:,jj])[0]) 
                                                    for jj in range(len(hind))]))

            elif iters is None:
                
                for ii in range(self.aclength_hyper):

                    # standard gaussian jump (this allows for different step sizes)
                    q = xnew.copy()
                    sigmas = 0.05 * len(hind)
                    probs = [0.1, 0.15, 0.5, 0.15, 0.1]
                    sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
                    scale = np.random.choice(sizes, p=probs)
                    par = np.random.choice(hind, size=1) # only one hyper param at a time
                    q[par] += np.random.randn(len(q[par])) * sigmas * scale

                    # get log-like and log prior at new position
                    lnlike1, lnprior1 = self.get_lnlikelihood(q), self.get_lnprior(q)

                    # metropolis step
                    diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
                    if diff > np.log(np.random.rand()):
                        xnew = q
                        lnlike0 = lnlike1
                        lnprior0 = lnprior1
                    else:
                        xnew = xnew
        
        return xnew


    def update_white_params(self, xs, iters=None):

        # get white noise parameter indices
        wind = self.get_efacequad_indices()

        xnew = xs.copy()
        lnlike0, lnprior0 = self.get_lnlikelihood_white(xnew), self.get_lnprior(xnew)
        
        if iters is not None:
            
            short_chain = np.zeros((iters,len(wind)))
            for ii in range(iters):
                # standard gaussian jump (this allows for different step sizes)
                q = xnew.copy()
                sigmas = 0.05 * len(wind)
                probs = [0.1, 0.15, 0.5, 0.15, 0.1]
                sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
                scale = np.random.choice(sizes, p=probs)
                par = np.random.choice(wind, size=1)
                q[par] += np.random.randn(len(q[par])) * sigmas * scale

                # get log-like and log prior at new position
                lnlike1, lnprior1 = self.get_lnlikelihood_white(q), self.get_lnprior(q)

                # metropolis step
                diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
                if diff > np.log(np.random.rand()):
                    xnew = q
                    lnlike0 = lnlike1
                    lnprior0 = lnprior1
                else:
                    xnew = xnew
                    
                short_chain[ii,:] = q[wind]
                
            self.cov_white = np.cov(short_chain[100:,:],rowvar=False)
            self.sigma_white = np.diag(self.cov_white)**0.5
            self.svd_white = np.linalg.svd(self.cov_white)
            self.aclength_white = int(np.max([int(acor.acor(short_chain[100:,jj])[0]) 
                                                  for jj in range(len(wind))]))
            
        elif iters is None:
            
            for ii in range(self.aclength_white):
                # standard gaussian jump (this allows for different step sizes)
                q = xnew.copy()
                sigmas = 0.05 * len(wind)
                probs = [0.1, 0.15, 0.5, 0.15, 0.1]
                sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
                scale = np.random.choice(sizes, p=probs)
                par = np.random.choice(wind, size=1)
                q[par] += np.random.randn(len(q[par])) * sigmas * scale
                
                #q[wind] += np.random.multivariate_normal(np.zeros(len(wind)),
                #                                         2.38**2 * self.cov_white / len(wind))
                #q[wind] += (2.38/ len(wind)) * np.dot(np.random.randn(len(wind)), 
                #                                      np.sqrt(self.svd_white[1])[:, None] * 
                #                                      self.svd_white[2])
                #par = np.random.choice(wind, size=1)
                #sigmas = self.sigma_white[list(wind).index(par)]
                #q[par] += 2.38 * np.random.randn(len(q[par])) * sigmas

                # get log-like and log prior at new position
                lnlike1, lnprior1 = self.get_lnlikelihood_white(q), self.get_lnprior(q)

                # metropolis step
                diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
                if diff > np.log(np.random.rand()):
                    xnew = q
                    lnlike0 = lnlike1
                    lnprior0 = lnprior1
                else:
                    xnew = xnew
             
        return xnew
    
    
    def update_ecorr_params(self, xs, iters=None):

        # get white noise parameter indices
        eind = self.get_ecorr_indices()

        xnew = xs.copy()
        
        #if self.ecorrsample == 'conditional':
        #    print('ERROR: Not working yet...')
        #    for ii,inds_sel in enumerate(self.ecorr_inds_sel):
        #        jvals = self._b[self.ecid[0]+inds_sel]
        #        tau = np.sum(jvals)**2 / len(jvals)
        #        #tau = np.mean(jvals**2)
        #        eta = np.random.uniform(0, 1-np.exp((tau/self.ecorrmax) - (tau/self.ecorrmin)))
        #        jnew = tau / ((tau/self.ecorrmax) - np.log(1-eta))
        #        xnew[eind[ii]] = 0.5*np.log10(jnew)
        #        print(ii,inds_sel,tau,eta,jnew,xnew)
                
        #else:
        lnlike0, lnprior0 = self.get_lnlikelihood(xnew), self.get_lnprior(xnew)
        
        if iters is not None:
            
            short_chain = np.zeros((iters,len(eind)))
            for ii in range(iters):

                # standard gaussian jump (this allows for different step sizes)
                q = xnew.copy()
                sigmas = 0.05 * len(eind)
                probs = [0.1, 0.15, 0.5, 0.15, 0.1]
                sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
                scale = np.random.choice(sizes, p=probs)
                par = np.random.choice(eind, size=1)
                q[par] += np.random.randn(len(q[par])) * sigmas * scale

                # get log-like and log prior at new position
                lnlike1, lnprior1 = self.get_lnlikelihood(q), self.get_lnprior(q)

                # metropolis step
                diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
                if diff > np.log(np.random.rand()):
                    xnew = q
                    lnlike0 = lnlike1
                    lnprior0 = lnprior1
                else:
                    xnew = xnew
                    
                short_chain[ii,:] = q[eind]
                
            self.cov_ecorr = np.cov(short_chain[100:,:],rowvar=False)
            self.sigma_ecorr = np.diag(self.cov_ecorr)**0.5
            self.svd_ecorr = np.linalg.svd(self.cov_ecorr)
            self.aclength_ecorr = int(np.max([int(acor.acor(short_chain[100:,jj])[0]) 
                                                  for jj in range(len(eind))]))
            
        elif iters is None:
            
            for ii in range(self.aclength_ecorr):
                q = xnew.copy()
                sigmas = 0.05 * len(eind)
                probs = [0.1, 0.15, 0.5, 0.15, 0.1]
                sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
                scale = np.random.choice(sizes, p=probs)
                par = np.random.choice(eind, size=1)
                q[par] += np.random.randn(len(q[par])) * sigmas * scale
                
                #q[eind] += np.random.multivariate_normal(np.zeros(len(eind)),
                #                                         2.38**2 * self.cov_ecorr / len(eind))
                #q[eind] += (2.38/ len(eind)) * np.dot(np.random.randn(len(eind)), 
                #                                      np.sqrt(self.svd_ecorr[1])[:, None] * 
                #                                      self.svd_ecorr[2])
                #par = np.random.choice(eind, size=1)
                #sigmas = self.sigma_ecorr[list(eind).index(par)]
                #q[par] += 2.38 * np.random.randn(len(q[par])) * sigmas

                # get log-like and log prior at new position
                lnlike1, lnprior1 = self.get_lnlikelihood(q), self.get_lnprior(q)

                # metropolis step
                diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
                if diff > np.log(np.random.rand()):
                    xnew = q
                    lnlike0 = lnlike1
                    lnprior0 = lnprior1
                else:
                    xnew = xnew
                    
        return xnew

    
    def update_b(self, xs): 

        # map parameter vector
        params = self.map_params(xs)

        # get auxiliaries
        Nvec = self.pta.get_ndiag(params)
        phiinv = self.pta.get_phiinv(params, logdet=False)
        residuals = self._residuals

        T = self.pta.get_basis(params)
        if not self.TNT and not self.d:
            for ii in range(len(self.pta.pulsars)):
                self.TNT.append(np.dot(T[ii].T, T[ii] / Nvec[ii][:,None]))
                self.d.append(np.dot(T[ii].T, residuals[ii]/Nvec[ii]))
        #d = self.pta.get_TNr(params)[0]
        #TNT = self.pta.get_TNT(params)[0]

        # Red noise piece
        b = []
        for ii in range(len(self.pta.pulsars)):
            Sigma = self.TNT[ii] + np.diag(phiinv[ii])

            try:
                u, s, _ = sl.svd(Sigma)
                mn = np.dot(u, np.dot(u.T, self.d[ii])/s)
                Li = u * np.sqrt(1/s)
            except np.linalg.LinAlgError:
                Q, R = sl.qr(Sigma)
                Sigi = sl.solve(R, Q.T)
                mn = np.dot(Sigi, self.d[ii])
                u, s, _ = sl.svd(Sigi)
                Li = u * np.sqrt(1/s)

            b.append(mn + np.dot(Li, np.random.randn(Li.shape[0])))

        return b

    
    def get_lnlikelihood_white(self, xs):

        # map parameters
        params = self.map_params(xs)
        matrix = self.pta.get_ndiag(params)[0]
        
        # Nvec and Tmat
        Nvec = matrix
        Tmat = self.pta.get_basis(params)[0]

        # whitened residuals
        mn = np.dot(Tmat, self._b)
        yred = self._residuals - mn

        # log determinant of N
        logdet_N = np.sum(np.log(Nvec))

        # triple product in likelihood function
        rNr = np.sum(yred**2/Nvec)

        # first component of likelihood function
        loglike = -0.5 * (logdet_N + rNr)

        return loglike


    def get_lnlikelihood(self, xs):

        # map parameter vector
        params = self.map_params(xs)

        # get auxiliaries
        Nvec = self.pta.get_ndiag(params)
        phiinv = self.pta.get_phiinv(params, 
                                     logdet=True)
        residuals = self._residuals

        T = self.pta.get_basis(params)
        if not self.TNT and not self.d:
            for ii in range(len(self.pta.pulsars)):
                self.TNT.append(np.dot(T[ii].T, T[ii] / Nvec[ii][:,None]))
                self.d.append(np.dot(T[ii].T, residuals[ii]/Nvec[ii]))

        # start likelihood calculations
        loglike = 0
        for ii in range(len(self.pta.pulsars)):

            # log determinant of N
            logdet_N = np.sum(np.log(Nvec[ii]))

            # triple product in likelihood function
            rNr = np.sum(residuals[ii]**2/Nvec[ii])

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr)

            # Red noise piece
            Sigma = self.TNT[ii] + np.diag(phiinv[ii][0])

            try:
                #print(ii,Sigma)
                cf = sl.cho_factor(Sigma)
                expval = sl.cho_solve(cf, self.d[ii])
            except np.linalg.LinAlgError:
                return -np.inf

            logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
            loglike += 0.5 * (np.dot(self.d[ii], expval) - 
                            logdet_sigma - phiinv[ii][1])

        return loglike

        
    def get_lnprior(self, params):
        # map parameter vector if needed
        params = params if isinstance(params, dict) else self.map_params(params)

        return np.sum([p.get_logpdf(params=params) for p in self.params])


    def sample(self, xs, outdir='./', niter=10000, resume=False):

        print(f'Creating chain directory: {outdir}')
        os.system(f'mkdir -p {outdir}')
        
        self.chain = np.zeros((niter, len(xs)))
        #self.bchain = np.zeros((niter, len(self._b)))
        
        self.iter = 0
        startLength = 0
        xnew = xs
        if resume:
            print('Resuming from previous run...')
            # read in previous chains
            tmp_chains = []
            tmp_chains.append(np.loadtxt(f'{outdir}/chain.txt'))
            #tmp_chains.append(np.loadtxt(f'{outdir}/bchain.txt'))
            
            # find minimum length
            minLength = np.min([tmp.shape[0] for tmp in tmp_chains])
            
            # take only the minimum length entries of each chain
            tmp_chains = [tmp[:minLength] for tmp in tmp_chains]
            
            # pad with zeros if shorter than niter
            self.chain[:tmp_chains[0].shape[0]] = tmp_chains[0]
            #self.bchain[:tmp_chains[1].shape[0]] = tmp_chains[1]
            
            # set new starting point for sampling
            startLength = minLength
            xnew = self.chain[startLength-1]
            
        tstart = time.time()
        for ii in range(startLength, niter):
            self.iter = ii
            self.chain[ii, :] = xnew
            #self.bchain[ii,:] = self._b
            
            if ii==0:
                self._b = self.update_b(xs)

            self.TNT = []
            self.d = []

            # update efac/equad parameters
            if self.get_efacequad_indices().size != 0:
                if ii==0:
                    xnew = self.update_white_params(xnew, iters=1000)
                else:
                    xnew = self.update_white_params(xnew, iters=None)
            
            # update ecorr parameters
            if self.get_ecorr_indices().size != 0:
                if ii==0:
                    xnew = self.update_ecorr_params(xnew, iters=1000)
                else:
                    xnew = self.update_ecorr_params(xnew, iters=None)

            # update red noise parameters
            if self.get_hyper_param_indices().size != 0:
                if self.redsample == 'conditional':
                    xnew = self.update_hyper_params(xnew)
                else:
                    if ii==0:
                        xnew = self.update_hyper_params(xnew, iters=100)
                    else:
                        xnew = self.update_hyper_params(xnew, iters=None)

            # update hyper-parameters
            xnew = self.update_rho_params(xnew)

            # if accepted update quadratic params
            if np.all(xnew != self.chain[ii,-1]):
                self._b = self.update_b(xnew)


            if ii % 100 == 0 and ii > 0:
                sys.stdout.write('\r')
                sys.stdout.write('Finished %g percent in %g seconds.'%(ii / niter * 100, 
                                                                       time.time()-tstart))
                sys.stdout.flush()
                np.savetxt(f'{outdir}/chain.txt', self.chain[:ii+1, :])
                #np.savetxt(f'{outdir}/bchain.txt', self.bchain[:ii+1, :])