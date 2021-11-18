import numpy as N  
import numpy.linalg

pdim = ?  # number of dimensions for the sky (without foregorunds) ps = (num of l)*(num of freq)

# the above needs to be appropriately initialized

alpha_dim = 6   # number of params for parametric model of cosmological power spectrum (number of band powers)

p_alpha_list=[]
for i in range(alpha_dim): 
  p_alpha=N.zeros((pdim,pdim))
  # the above needs to be initialized to the alpha basis shape for the power spectrum
  p_alpha_list.append(p_alpha)

alpha_vec=N.zero((alpha_dim,))
#needs to be initialized 

# initialize fiducial power spectrum (true sky model)
p_0=N.zeros((pdim,pdim))  # reference power spectrum
for i in range(alpha_dim): 
  p_0+=alpha_vec[i]*p_alpha_list[i]

def make_fisher():
  fisher=N.zeros((alpha_dim, alpha_dim))
  for a in range(alpha_dim): 
    for b in range(alpha_dim): 
      if a <= b :
        result=0
        # this is the part one wants to parallelize 
        for m in range(m_max):
          result+=make_fisherM(a,b,m)
        fisher[a,b]=result
        fisher[b,a]=fisher[a,b]

def make_fisherM(a,b,m):
   p0=make_data covariance_m(m,p_0)
   c_plus_n_inverse=numpy.linalg.inv(p0+noise(m))
   pa=make_data covariance_m(m,p_alpha_list[a])
   pa=make_data covariance_m(m,p_alpha_list[b])
   result=N.trace( c_plus_n_inverse @ pa @ c_plus_n_inverse @ pb )
   return(result)

def make_data covariance_m(m,p_cov):
  transfer=getTransfer(m)
  result= transfer @ p_cov @ N.transpose(transfer)
  return(result)

def noise_m():
  # this has to be filled in with visibility measurement noise, covariance 
  # matrix, possibly projected onto a subspace of smaller dimension 
  pass

def getTransfer(m):
  pass
  # to be filled in, transfer matrix from sky signal to visibility vector, possibly projected onto a subspace of smaller dimension 

