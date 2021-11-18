import numpy as N  
import numpy.linalg
from cora.signal import corr21cm
from drift.core import skymodel


class kspace_cartesian():
    def __init__(self, kpar_start, kpar_end, kpar_dim, kperp_start, kperp_end, kperp_dim, pipeline_info):
        self.alpha_dim = kpar_dim * kperp_dim 
        self.kperp_dim = kperp_dim
        self.kpar_dim = kpar_dim
        self.telescope = pipeline_info.telescope
        self.beamtransfer = pipeline_info.beamtransfer
        self.kltrans = pipeline_info
        
        self.k_par_boundaries = N.linspace(kpar_start, kpar_end, kpar_dim + 1)
        self.k_par_centers = 0.5 * (self.k_par_boundaries[:-1] + self.k_par_boundaries[1:])
        
        self.k_perp_boundaries = N.linspace(kperp_start, kperp_end, kperp_dim + 1)
        self.k_perp_centers = 0.5 * (self.k_perp_boundaries[:-1] + self.k_perp_boundaries[1:])
        
        Aux1, Aux2 = N.broadcast_arrays(self.k_par_centers[:, N.newaxis], self.k_perp_centers)
        Aux = (Aux1 ** 2 + Aux2 ** 2) ** 0.5 
        self.k_centers = Aux.flatten()

    def make_Response_matrix(self, ind): # t
        pk_ind = self.make_binning_function(ind)
        return self.make_clzz(pk_ind)
        
    def make_binning_function(self, band_ind):
        row_ind = int(band_ind / self.kperp_dim)
        col_ind = int(band_ind % self.kperp_dim)
        
        kpar_s = self.k_par_boundaries[row_ind]
        kpar_e = self.k_par_boundaries[row_ind+1]
        kperp_s = self.k_perp_boundaries[col_ind]
        kperp_e = self.k_perp_boundaries[col_ind+1]
        
        def band(k, mu):

            kpar = k * mu
            kperp = k * (1.0 - mu ** 2) ** 0.5

            parb = (kpar >= kpar_s) * (kpar <= kpar_e)
            perpb = (kperp >= kperp_s) * (kperp < kperp_e)

            return (parb * perpb).astype(N.float64)

        return band
    
    def make_binning_power(self):
        cr = corr21cm.Corr21cm()
        cr.ps_2d = False
        return cr.ps_vv(self.k_centers)
        
    def make_clzz(self, pk):
        """Make an angular powerspectrum from the input matter powerspectrum.
        -------
        clzz : [lmax+1, nfreq, nfreq]
        """
        crt = corr21cm.Corr21cm(ps=pk, redshift=1.5)
        crt.ps_2d = True

        clzz = skymodel.im21cm_model(
            self.telescope.lmax,
            self.telescope.frequencies,
            self.telescope.num_pol_sky,
            cr=crt,
            temponly=True,
        )
        
        return clzz         
    
    
    
class Fisher_analysis(kspace_cartesian):
    
    def init_fiducial_reference_ps(self):
        self.p_alpha_list = []
        for i in range(self.alpha_dim): 
            p_alpha = self.make_Response_matrix(i)
            self.p_alpha_list.append(p_alpha)
    
        self.alpha_vec = self.make_binning_power()
        npol = self.telescope.num_pol_sky
        ldim = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq
        
        if self.svd_cut:
            self.p_0 = N.zeros((npol,npol,ldim,ldim,nfreq,nfreq))  # reference power spectrum
            for i in range(self.alpha_dim): 
                for j in range(ldim):
                    self.p_0[0,0,j,j,:,:] += self.alpha_vec[i] * self.p_alpha_list[i][j,:,:]
        else:
            self.p_0 = N.zeros((npol,npol,ldim,nfreq,nfreq))  # reference power spectrum
            for i in range(self.alpha_dim):
                self.p_0[0,0,:,:,:] += self.alpha_vec[i] * self.p_alpha_list[i]
            
        return
            
    def make_fisher(self, svd_cut="False"):
        
        self.svd_cut = svd_cut
        self.init_fiducial_reference_ps()
        self.make_foreground_covariance()
        
        fisher=N.zeros((self.alpha_dim, self.alpha_dim))
        for a in range(self.alpha_dim):
            for b in range(self.alpha_dim):
                if a <= b :
                    result=0
                    for m in range(self.telescope.mmax):
                        result += self.make_fisherM(a,b,m)
                        fisher[a,b]=result
                        fisher[b,a]=fisher[a,b]
        return fisher
                        
    def make_fisherM(self,a,b,m):
        cv_noise = self.make_noise_covariance(m)
        kl_covariance = self.project_covariance_tele_to_kl_m(m, cv_noise) + self.project_covariance_sky_to_kl_m(m, self.p_0 + self.cv_fg)
        nfreq = self.telescope.nfreq
        kl_len = kl_covariance.shape[1]
        shape = (nfreq*kl_len,nfreq*kl_len)
        kl_covariance=kl_covariance.reshape(shape)
    
        c_plus_n_inverse = N.linalg.inv(kl_covariance) 
        
        lside = self.telescope.lmax+1
        npol = self.telescope.num_pol_sky
        
        if self.svd_cut:
            pa = N.zeros((npol,npol,lside,nfreq,nfreq))
            pa[0,0,:,:,:] = self.p_alpha_list[a]
            pa = self.project_covariance_sky_to_kl_m(m, pa).reshape(shape)
            
            pb = N.zeros((npol,npol,lside,nfreq,nfreq))
            pb[0,0,:,:,:] = self.p_alpha_list[b]
            pb = self.project_covariance_sky_to_kl_m(m, pb).reshape(shape)
        else:
            pa = N.zeros((npol,npol,lside,lside,nfreq,nfreq))
            for i in N.arange(lside):
                pa[0,0,i,i,:,:] = self.p_alpha_list[a][i,:,:]
            pa = self.project_covariance_sky_to_kl_m(m, pa).reshape(shape)

            pb = N.zeros((npol,npol,lside,lside,nfreq,nfreq))
            for i in N.arange(lside):
                pb[0,0,i,i,:,:] = self.p_alpha_list[b][i,:,:]
            pb = self.project_covariance_sky_to_kl_m(m, pb).reshape(shape)
            
        result = N.trace( c_plus_n_inverse @ pa @ c_plus_n_inverse @ pb )
        return result
    
    def project_covariance_tele_to_kl_m(self, m, cv_noise):
        """
        This function projects the noise convariance matrices to the KL basis.
            Noise covaricance (nfreq, ntel, nfreq, ntel)
            - from telescope to KL
            - type = "KlUt" (kl_len, nfreq, ntel)
        """
        if self.svd_cut:
            N_svd = self.beamtransfer.project_matrix_diagonal_telescope_to_svd(m, cv_noise)
            N_kl  = self.kltrans.project_matrix_svd_to_kl(m, N_svd, threshold=None)
        else:
            trans_KU = self.getTransfer(m, type="KlUt")
            kl_len, nfreq, ntel = trans_KU.shape
            N_kl = N.zeros((nfreq, kl_len, nfreq, kl_len))
            for i in N.arange(nfreq):
                for j in N.arange(nfreq):
                    N_kl[i, :, j, :] = trans_KU[:, i, :] @ cv_noise[i, :, j, :] @ trans_KU.T.conj()[:, j, :]
        
        return N_kl
        
    def project_covariance_sky_to_kl_m(self, m, p_0):
        """
        This function projects the convariance matrices to the KL basis.
        Conceptually:
            Cosmological and foreground covariance (pol2, pol1, l, freq1, freq2)
            - from Sky to KL.
            - type = "KlUtB" (kl_len, nfreq, npol, lmax+1)
        """
        if self.svd_cut:
            p0_kl = self.kltrans.project_matrix_sky_to_kl(m, p_0, threshold=None)
        else:
            trans_KUB = self.getTransfer(m, type="KlUtB")
            kl_len, nfreq, npol, lside = trans_KUB.shape
            p0_kl = N.zeros((nfreq, kl_len, nfreq, kl_len))
            for i in N.arange(nfreq):
                for j in N.arange(nfreq):
                    A = trans_KUB[:, i, :, :] # shape: (kl_len, npol, lside)
                    B = trans_KUB.T.conj()[:, :, j, :] # shape: (lside, npol, kl_len)
                    P = p_0[:, :, :, :, i,j] # shape: (npol,npol,lside,lside)
                    aux = N.tensordot(A, P , axes=([1,2],[2,1])) # (kl_len, npol, lside)
                    aux = N.tensordot(aux, B ,axes=([1,2],[0,1])) # (kl_len, kl_len)
                    p0_kl[i, :, j, :] = aux
        
        return p0_kl
        
        


    
    def getTransfer(self, m, type = None):
        """ 
        Choose one of the following types of linear transformations: 
        1: B      (from sky to telescope, i.e., beamtransfer)  
                   - This projection contracts "Pol" and "l"; emergent index: "baseline";  per-frequency operation
        2: Ut     (from telescope to the SVD basis)            
                   - This projection contracts "baseline";    emergent index: "SVD basis"; per-frequency operation
        3: Kl     (from the SVD basis to the KL basis)         
                   - This projection contracts "SVD basis";   emergent index: "KL basis";  universal over frequency
        4: UtB    (from sky to the SVD basis)                  
        5: KlUt   (from telescope to the KL basis)
        6: KlUtB  (from sky to the KL basis)  
     
        The shapes of the transfer matrices are:
            for "B":    (nfreq, 2, npairs, npol, lmax+1), where svd_len = min((lmax+1)*npol, ntel);
            for "Ut":   (nfreq, svd_len, ntel), where ntel = 2*npairs and "2" denotes +/- m-modes, or say, the 
                        two different orientations of each baseline;
            for "Kl":   (kl_len, svd_len)
            for "UtB":  (nfreq, svd_len, npol, lmax+1)
            for "KlUt": (kl_len, nfreq, ntel)
            for "KlUtB":(kl_len, nfreq, npol, lmax+1) 
        """
        if type=="B":
            transfer = self.beamtransfer.beam_m(m) # fetch the beam file
        elif type=="Ut":
            svdfile = h5py.File(self.beamtransfer._svdfile(m), "r")
            transfer = svdfile["beam_ut"]
            svdfile.close()
        elif type=="Kl":
            transfer = self.kltrans.modes_m(m, threshold=None)[1]
        elif type=="UtB":
            transfer = self.beamtransfer.beam_svd(m)
        elif type=="KlUt":
            KL = self.kltrans.modes_m(m, threshold=None)[1]
            svdfile = h5py.File(self.beamtransfer._svdfile(m), "r")
            Ut = svdfile["beam_ut"]
            svdfile.close()
            transfer = N.tensordot(KL,Ut,axes=([1,],[1,])) 
        elif type=="KlUtB":
            KL = self.kltrans.modes_m(m, threshold=None)[1]
            UtB = self.beamtransfer.beam_svd(m)
            transfer =  N.tensordot(KL,UtB,axes=([1,],[1,])) 
        else: 
            print("Failed to fetch the transfer array  :-( ")   

        return transfer
    

# Cv_fg_sky = self.make_foreground_covariance()
# Cv_noise_tele = slef.make_noise_covariance(m)
# 
    def make_foreground_covariance(self):
        """ This function is basically Shaw's code.
        -------
        cv_fg  : np.ndarray[pol2, pol1, l, freq1, freq2]
        Result : np.ndarray[pol2, pol1, l, l, freq1, freq2]
        """
        
        KLclass = self.kltrans
        npol = self.telescope.num_pol_sky
        
            # If not polarised then zero out the polarised components of the array
        if KLclass.use_polarised:
            cv_fg = skymodel.foreground_model(
                    self.telescope.lmax,
                    self.telescope.frequencies,
                    npol,
                    pol_length=KLclass.pol_length,
                )
        else:
            cv_fg = skymodel.foreground_model(
                    self.telescope.lmax, self.telescope.frequencies, npol, pol_frac=0.0
                )

        if self.svd_cut:
            a, b, c, d, e = cv_fg.shape
            Result = N.zeros((a, b, c, c, d, e))
            for i in N.arange(c):
                Result[:,:,i,i,:,:] = cv_fg[:,:,i,:,:]
        else:
            Result = cv_fg
            
        self.cv_fg = Result
            
        return
        
    
    
    def make_noise_covariance(self, m):
        # Noise covariance in the telescope/visibility space.
        # Return: (nfreq,ntel,nfreq,ntel)
        bl = N.arange(self.telescope.npairs)
        bl = N.concatenate((bl, bl))
        npower = self.telescope.noisepower(
            bl[N.newaxis, :], N.arange(self.telescope.nfreq)[:, N.newaxis]
        ).reshape(self.telescope.nfreq, self.beamtransfer.ntel) 

        if self.svd_cut:
            NoiseCov = npower
        else:  
            NoiseCov = N.zeros((self.telescope.nfreq, self.beamtransfer.ntel, self.telescope.nfreq, self.beamtransfer.ntel))
                
            for i in N.arange(self.telescope.nfreq):
                for j in N.arange(self.beamtransfer.ntel):
                    NoiseCov[i,j,i,j] = npower[i,j]
                
        return NoiseCov
        
        



    
    #beam = self.beamtransfer.beam_m(m).reshape((self.telescope.nfreq, self.ntel, npol, self.telescope.lmax + 1))
    