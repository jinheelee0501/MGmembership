from datetime import datetime
import struct
import numpy as np
from calc_prior import *
from calc_likeli_array import *
from calc_likeli_uniformXYZ import *

t0 = datetime.now()
print 'START at %s' %(t0)

# group information

groups = ['BPMG','THA','ABDMG','COL', 'CAR',  'TWA', 'ARG','FLD1','FLD2','FLD3','FLD4']

youngn = 390007
oldn = 1601130

totfld = youngn + oldn
nn = 87.93

population = [ nn,     nn,   nn,    nn,   nn,    nn,   nn, totfld*0.25,totfld*0.29,totfld*0.26,totfld*0.19]


const=1.
d2r = np.pi/180.

Xmean =   [ 19.7,   6.74, -2.53, -28.11, 10.09,  19.10, 15.04,      0,     0,     0,   0]
Ymean =   [ -2.8, -21.79,  1.28, -25.78,-51.63, -54.16,-21.69,      0,     0,     0,   0]
Zmean =   [-14.4, -36.05,-16.34, -28.56,-14.85,  21.54, -8.09,      0,     0,     0,   0]
sigX =    [ 22.2,   3.90, 16.33,  10.55,  5.78,   4.98, 12.07,    200,   200,   200, 200]
sigY =    [ 32.5,  10.62, 19.95,  17.63, 11.34,   7.16, 15.51,    200,   200,   200, 200]
sigZ =    [ 67.0,   20.1, 23.47,  28.33, 29.79,  22.57, 27.43,    200,   200,   200, 200]
phiXYZ =  [-24.9,  -28.2,  57.3,  -25.7,  18.4,   25.3, -12.4,      0,     0,     0,   0]
thetaXYZ= [ 77.2,  263.1,  51.9,  -35.5, -16.5,   60.8, -73.0,      0,     0,     0,   0]
psiXYZ =  [-18.4,   21.1,  88.7,  -62.2, -64.9,   80.4, -51.9,      0,     0,     0,   0]
meanU =   [-10.7,   -9.7, -6.96, -12.14,-10.72, -11.12,-21.54,  -17.1, -32.0,   2.2, 22.1]
meanV =   [-16.0, -20.47,-27.23, -21.29,-22.23, -18.88,-12.24,  -18.0, -16.2,  -0.8,-16.8]
meanW =   [ -9.3,  -0.78, -13.9,  -5.61, -5.67,  -5.63, -4.63,   -6.3,  -8.1,  -6.7, -9.5]
sigU =    [  1.4,   1.05,  1.18,   0.51,  0.31,    0.9,  0.87,    6.8,  15.8,   8.8, 16.1]
sigV =    [  1.5,   1.68,  1.68,   1.27,  0.65,   1.56,  1.67,    8.1,  16.5,   9.8, 16.8]
sigW =    [  2.4,   2.38,  1.94,   1.69,  1.08,   2.78,  2.74,   14.2,  19.4,  13.9, 21.3]
phiUVW =  [105.8,  -52.0, -54.4,  143.4, -68.0, -158.7,  76.1,   48.5, -54.7,  88.8, 110.9]
thetaUVW= [-51.2,  -30.2, 185.8,   22.7, -61.6,  -55.3,  55.9,  -86.2, -80.7,   1.2, -67.8]
psiUVW =  [ 84.5,    1.6,  10.7,  -68.8, -86.4,   -5.4,  29.4,  -56.2, 112.6, -97.4, 104.1]









phiUVW = np.array(phiUVW)     * d2r
thetaUVW = np.array(thetaUVW) * d2r
psiUVW = np.array(psiUVW)     * d2r
phiXYZ = np.array(phiXYZ)     * d2r
thetaXYZ = np.array(thetaXYZ) * d2r
psiXYZ = np.array(psiXYZ)     * d2r

# environmental set-up
gidxs  = np.arange(0,len(groups),1)

## parameter setup

gorus        = [ 'u', 'g', 'g', 'g', 'g', 'g', 'g',  'u', 'u', 'u', 'u']
sharpsmooths = ['sh', '-', '-', '-', '-', '-', '-', 'sh','sh','sh','sh']
scaleconsts  = [ 1.2,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ,   1 ,  1 ,  1 ,  1 ]
uses         = [ 1 ,   1 ,  1 ,  1 ,  1 ,  1 ,  1 ,   1 ,  1 ,  1 ,  1 ]
 
# input data
#pos = raw_input("ra de (deg) => ")
rastr = raw_input("ra (hh mm ss) ==> ")
destr = raw_input("de (dd mm ss) ==> ")
pms = raw_input("pmra epmra pmde epmde (mas/yr)==>")
plxs = raw_input("plx eplx (mas; if not exist, press enter) ==>")
rvs = raw_input("rv erv (km/s; if not exist, press enter) ==>")

rah,ram,ras = [float(x) for x in rastr.split()]
ded,dem,des = [float(x) for x in destr.split()]
ra = (rah+ram/60.0+ras/3600.0)*15.0
de = abs(ded) + dem/60.0+des/3600.
if ded < 0: de = -de

pmra,epmra,pmde,epmde = pms.split()

ra,de = float(ra),float(de)
pmra,epmra,pmde,epmde = float(pmra),float(epmra),float(pmde),float(epmde)

try:
    plx,eplx = plxs.split()
    plx,eplx = float(plx),float(eplx)
except:
    plx,eplx,dist,edist= False,False,False,False

try:
    rv,erv = rvs.split()
    rv,erv = float(rv),float(erv)
except:
    rv,erv = False,False

     
pmra = pmra*0.001 ; epmra = epmra*0.001 ; pmde = pmde*0.001 ; epmde = epmde*0.001

if epmra < 1e-4: epmra = 1e-4
if epmde < 1e-4 : epmde = 1e-4
if np.isnan(erv): erv = abs(rv*0.1)
if np.isnan(eplx): eplx = abs(plx*0.05)
if np.isnan(epmra) : epmra = abs(pmra*0.2)
if np.isnan(epmde) : epmde = abs(pmde*0.2)

radeg,dedeg= ra,de
# Begin calculation
if plx:
    dist = 1000./plx ; edist = 1000.*eplx/plx**2.



priors = [] ; likelihoods = [] ; posteriors = []
statdists,statrvs = [],[]
LX,LY,LZ,LU,LV,LW = [],[],[],[],[],[]

for gid, goru, sharpsmooth, use in zip(gidxs, gorus, sharpsmooths, uses):
    if use == 0: continue
                
    Xg = Xmean[gid] ; Yg = Ymean[gid] ; Zg = Zmean[gid] ; dXg = sigX[gid] ; dYg = sigY[gid] ; dZg = sigZ[gid] ; phiUVWg = phiUVW[gid] ; thetaUVWg = thetaUVW[gid] ; psiUVWg = psiUVW[gid]
    Ug = meanU[gid] ; Vg = meanV[gid] ; Wg = meanW[gid] ; dUg = sigU[gid] ; dVg = sigV[gid] ; dWg = sigW[gid] ; phiXYZg = phiXYZ[gid] ; thetaXYZg = thetaXYZ[gid] ; psiXYZg = psiXYZ[gid] 
    
    priordata = calc_prior(groups[gid],population[gid],radeg,dedeg,pmra,epmra,pmde,epmde,rv_obs=rv,erv_obs=erv,dist_obs=dist,edist_obs=edist,goru=goru)
    
    prior = priordata[0]
    prior_pm,prior_gb,prior_rv,prior_dist,pm0,gb0,rv0,dist0 = priordata[1],priordata[2],priordata[3],priordata[4],priordata[5],priordata[6],priordata[7],priordata[8]

    if goru == 'g':
        likeliresult = calc_likeli_array(groups[gid],Xg,Yg,Zg,dXg,dYg,dZg,phiXYZg,thetaXYZg,psiXYZg,Ug,Vg,Wg,dUg,dVg,dWg,phiUVWg,thetaUVWg,psiUVWg,radeg,dedeg,pmra,epmra,pmde,epmde,v=rv,ev=erv,d=dist,ed=edist)
    if goru == 'u':
        likeliresult = calc_likeli_uniformXYZ(groups[gid],Xg,Yg,Zg,dXg,dYg,dZg,phiXYZg,thetaXYZg,psiXYZg,Ug,Vg,Wg,dUg,dVg,dWg,phiUVWg,thetaUVWg,psiUVWg,radeg,dedeg,pmra,epmra,pmde,epmde,v=rv,ev=erv,d=dist,ed=edist,sharpsmooth=sharpsmooth)

    priors.append(prior)
    likelihoods.append(likeliresult[0])
    statdists.append(likeliresult[1])
    statrvs.append(likeliresult[2])
    LX.append(likeliresult[3])
    LY.append(likeliresult[4])
    LZ.append(likeliresult[5])
    LU.append(likeliresult[6])
    LV.append(likeliresult[7])
    LW.append(likeliresult[8])


priors = np.array(priors) ; likelihoods = np.array(likelihoods)
statdists = np.array(statdists) ; statrvs = np.array(statrvs)


posterior2 = priors * likelihoods
posteriors = 100.*posterior2/np.sum(posterior2)
index =np.argmax(posteriors)
statdist = statdists[index]
statrv = statrvs[index]


print "----------Membership probability--------------"
print "BPMG   : %.1f" %posteriors[0]
print "Tuc-Hor: %.1f" %posteriors[1]
print "AB Dor : %.1f" %posteriors[2] 
print "Columba: %.1f" %posteriors[3]
print "Carina : %.1f" %posteriors[4]
print "TWA    : %.1f" %posteriors[5]
print "Argus  : %.1f" %posteriors[6]
print "Field  : %.1f" %sum(posteriors[7:])

if (plx is False) & (index != 7):
    print "----statistical distance = %.1f pc"  %statdist
if (rv is  False) & (index != 7):
    print "----statistical rv = %.1f km/s"  %(statrv)


 
