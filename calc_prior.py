# calc_prior
# calculate two priors: corresponding to condition / rv and dist are not known case

def calc_prior(gname, Nk, ra, de, pmra_obs=False, epmra_obs=False, pmde_obs=False, epmde_obs=False, rv_obs=False, erv_obs=False, dist_obs=False, edist_obs=False,goru=False):

    import coords
    import numpy as np

    pdir = './'

    if np.isnan(rv_obs): rv_obs=False
    if np.isnan(erv_obs): erv_obs=False
    if np.isnan(dist_obs): dist_obs=False
    if np.isnan(edist_obs): edist_obs=False


    gl_obs, gb_obs = coords.eq2gal(ra, de, b1950=False)

    pm_obs = np.sqrt(pmra_obs**2.+pmde_obs**2.)
    epm_obs = np.sqrt((pmra_obs/pm_obs*epmra_obs)**2. + (pmde_obs/pm_obs*epmde_obs)**2.)

    if goru =='u':  PDFfile = pdir+'PriorPDF_rv_dist_b_pm_%s_%.1f_uniformXYZ.txt' %(gname,10000000.0)    
    elif goru == 'g' : PDFfile = pdir+'PriorPDF_rv_dist_b_pm_%s_%.1f.txt' %(gname,10000000.0)


    rv, rvpdf, dist,distpdf, gb, gbpdf, pm, pmpdf = np.loadtxt(PDFfile,delimiter=',',unpack=True)

    drv = np.abs(rv[1]-rv[0]) ; ddist = np.abs(dist[1]-dist[0]) ; dgb = np.abs(gb[1]-gb[0]) ; dpm = np.abs(pm[1]-pm[0])

    prior_pm = 0. ; prior_gb = 0. ; prior_rv = 0.; prior_dist =0.
    for j in range(len(rv)):    
        prior_pm = prior_pm + np.exp(-0.5*(pm[j]-pm_obs)**2./epm_obs**2.)*pmpdf[j]*dpm

    idx,num = min(enumerate(gb), key=lambda x: abs(x[1]-gb_obs))
    prior_gb = gbpdf[idx]*dgb
    if rv_obs: 
        for j in range(len(rv)):
            prior_rv = prior_rv + np.exp(-0.5*(rv[j]-rv_obs)**2./erv_obs**2.)*rvpdf[j]*drv
    else: prior_rv = 1. ; erv_obs = 1.0

    if dist_obs:
        for j in range(len(rv)):
            prior_dist = prior_dist + np.exp(-0.5*(dist[j]-dist_obs)**2./edist_obs**2.)*distpdf[j]*ddist
    else: prior_dist = 1. ; edist_obs = 1.0

    prior_pm = prior_pm/epm_obs
    prior_gb = prior_gb
    prior_rv = prior_rv/erv_obs
    prior_dist = prior_dist/edist_obs

    fin_prior = Nk * prior_pm * prior_gb * prior_rv * prior_dist

       
 #   print fin_prior
    return fin_prior,prior_pm,prior_gb,prior_rv,prior_dist,pm_obs,gb_obs,rv_obs,dist_obs

