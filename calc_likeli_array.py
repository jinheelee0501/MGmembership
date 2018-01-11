# calc_likeli_array.py

# calculate likelihoods according to measured information
# return stat dist, rv 

def calc_TA(rra, dde): 
    import numpy as np
    d2r = np.pi/180.

    c_ra = np.cos(rra*d2r)
    s_ra = np.sin(rra*d2r)
    c_de = np.cos(dde*d2r)
    s_de = np.sin(dde*d2r)
    A1 = np.array([[c_ra,s_ra,0],[s_ra,-c_ra,0],[0,0,-1]])
    A2 = np.array([[c_de,0,-s_de],[0,-1,0],[-s_de,0,-c_de]])
    A = np.dot(A1,A2)

# for NGP
    a_ngp = 192.859508
    d_ngp = 27.128336
    t_ngp = 122.932

#    c_ra_ngp = np.cos(a_ngp*d2r)
#    s_ra_ngp = np.sin(a_ngp*d2r)
#    c_de_ngp = np.cos(d_ngp*d2r)
#    s_de_ngp = np.sin(d_ngp*d2r)
#    c_t_ngp = np.cos(t_ngp*d2r)
#    s_t_ngp = np.sin(t_ngp*d2r)

#    T1 = np.array([[c_t_ngp,s_t_ngp,0],[s_t_ngp,-c_t_ngp,0],[0,0,1]])
#    T2 = np.array([[-s_de_ngp,0,c_de_ngp],[0,-1,0],[c_de_ngp,0,s_de_ngp]])
#    T3 = np.array([[c_ra_ngp,s_ra_ngp,0],[s_ra_ngp,-c_ra_ngp,0],[0,0,1]])

#    Test = np.dot(T1,T2)
#    T = np.dot(Test,T3)

    #T = np.array([[-0.06699,-0.87276,-0.48354],[0.49273,-0.45035,0.74458],[-0.86760,-0.18837,0.46020]])
    T = np.array([[-0.0548755604,-0.8734370902,-0.4838350155],[0.4941094279,-0.4448296300,0.7469822445],[-0.8676661490,-0.1980763734,0.4559837762]])  # Gagne
    TA = np.dot(T,A)

    return TA

def calc_likeli_array(group_name,X,Y,Z,dX,dY,dZ,a1,a2,a3,U,V,W,dU,dV,dW,b1,b2,b3,ra,de,pra,epra,pde,epde,v=False,ev=False,d=False,ed=False):

    const = -1
    a1 = a1*const ; a2 = a2*const ; a3 = a3*const
    b1 = b1*const ; b2 = b2*const ; b3 = b3*const

    import coords
    import numpy as np

    if np.isnan(v) : v = False
    if np.isnan(ev) : ev = False
    if np.isnan(d) : d = False
    if np.isnan(ed) : ed = False
  
    gl_obs, gb_obs = coords.eq2gal(ra,de,b1950=False)

# zyx rotation
    R1_S = np.array([[np.cos(a1),-np.sin(a1),0.],[np.sin(a1),np.cos(a1),0.],[0.,0.,1.]])
    R2_S = np.array([[np.cos(a2),0.,np.sin(a2)],[0.,1.,0.],[-np.sin(a2),0.,np.cos(a2)]])
    R3_S = np.array([[1.,0.,0.],[0.,np.cos(a3),-np.sin(a3)],[0.,np.sin(a3),np.cos(a3)]])

    Rot_S1 = np.dot(R2_S,R3_S)
    Rot_SFin = np.dot(R1_S,Rot_S1)
    Rot_SFin = np.dot(R3_S,np.dot(R2_S,R1_S)) 
# zyx
    R1_D = np.array([[np.cos(b1),-np.sin(b1),0],[np.sin(b1),np.cos(b1),0],[0.,0.,1.]])
    R2_D = np.array([[np.cos(b2),0,np.sin(b2)],[0.,1.,0.],[-np.sin(b2),0,np.cos(b2)]])
    R3_D = np.array([[1.,0.,0.],[0,np.cos(b3),-np.sin(b3)],[0,np.sin(b3),np.cos(b3)]])

    Rot_D1 = np.dot(R2_D,R3_D)
    Rot_DFin = np.dot(R1_D,Rot_D1)
    Rot_DFin = np.dot(R3_D,np.dot(R2_D,R1_D))

    pdir = './'
    infile = pdir+'PriorPDF_rv_dist_b_pm_%s_%.1f.txt' %(group_name,10000000.0)
    rv,rvpdf,dist,distpdf,gb,gbpdf,pm,pmpdf = np.loadtxt(infile,delimiter=',',unpack=True)
    didx = min(range(len(dist)), key=lambda ii: abs(dist[ii]-100))
    ridx ,= np.where(rvpdf > 0.)
    distrange = range(0,didx,5)
    rvrange = range(ridx[0],ridx[-1],len(ridx)/20)

    dist_err = np.zeros(len(dist))
    ddist = abs(dist[distrange[1]]-dist[distrange[0]])
    rv_err = np.zeros(len(rv)) 
    drv = abs(rv[rvrange[1]]-rv[rvrange[0]])   

# calculate (actual) likellihood

    if d: 
        dist = [d] ; dist_err = [ed] 
        ddist= 1.0 ; distpdf = [1.] ; distrange=[0]
    if v:
        rv = [v] ; rv_err = [ev] 
        drv = 1.0 ;rvpdf = [1.] ; rvrange=[0]

    like = 0.

    xx = np.cos(gb_obs*np.pi/180.)*np.cos(gl_obs*np.pi/180.)
    yy = np.cos(gb_obs*np.pi/180.)*np.sin(gl_obs*np.pi/180.)
    zz = np.sin(gb_obs*np.pi/180.)

    TA = calc_TA(ra,de)    
#    K = 4.74057
    K = 4.743717361  # Gagne IDL procedure

    resultarr = np.zeros([len(distrange)+1,len(rvrange)+1])
    #print rvrange,len(rvrange)
    #print distrange,len(distrange)
    #print resultarr.shape

    for iid, ii in enumerate(distrange): # 50 iteration #(len(dist)):
        resultarr[iid+1,0] = dist[ii]

        x = xx*dist[ii]
        y = yy*dist[ii]
        z = zz*dist[ii]
        dx = xx*dist_err[ii]
        dy = yy*dist_err[ii]
        dz = zz*dist_err[ii]

        xp = Rot_SFin[0][0]*(x-X) + Rot_SFin[0][1]*(y-Y) + Rot_SFin[0][2]*(z-Z) + X
        yp = Rot_SFin[1][0]*(x-X) + Rot_SFin[1][1]*(y-Y) + Rot_SFin[1][2]*(z-Z) + Y
        zp = Rot_SFin[2][0]*(x-X) + Rot_SFin[2][1]*(y-Y) + Rot_SFin[2][2]*(z-Z) + Z

        dxp = np.sqrt((Rot_SFin[0][0]*dx)**2 + (Rot_SFin[0][1]*dy)**2 + (Rot_SFin[0][2]*dz)**2)
        dyp = np.sqrt((Rot_SFin[1][0]*dx)**2 + (Rot_SFin[1][1]*dy)**2 + (Rot_SFin[1][2]*dz)**2)
        dzp = np.sqrt((Rot_SFin[2][0]*dx)**2 + (Rot_SFin[2][1]*dy)**2 + (Rot_SFin[2][2]*dz)**2)

        Xerr = np.sqrt(dX**2 + dxp**2)
        Yerr = np.sqrt(dY**2 + dyp**2)
        Zerr = np.sqrt(dZ**2 + dzp**2)
        
        likeX = np.exp(-0.5*(xp-X)**2./Xerr**2.) / Xerr / np.sqrt(2.*np.pi)
        likeY = np.exp(-0.5*(yp-Y)**2./Yerr**2.) / Yerr / np.sqrt(2.*np.pi)
        likeZ = np.exp(-0.5*(zp-Z)**2./Zerr**2.) / Zerr / np.sqrt(2.*np.pi)

        for jjr, jj in enumerate(rvrange): # range(len(rv)):
            #print "DIST and RV= ",dist[ii],dist_err[ii],rv[jj],ev
            resultarr[0,jjr+1] = rv[jj]

            arr = np.array([rv[jj], K*pra*dist[ii], K*pde*dist[ii]])
            u,v,w = np.dot(TA, arr)

            du = np.sqrt( (TA[0][0]*ev)**2.+(TA[0][1]*K)**2*((epra*dist[ii])**2+(pra*dist_err[ii])**2+(epra*dist_err[ii])**2) + (TA[0][2]*K)**2*((epde*dist[ii])**2+(pde*dist_err[ii])**2+(epde*dist_err[ii])**2 ))
            dv = np.sqrt( (TA[1][0]*ev)**2.+(TA[1][1]*K)**2*((epra*dist[ii])**2+(pra*dist_err[ii])**2+(epra*dist_err[ii])**2) + (TA[1][2]*K)**2*((epde*dist[ii])**2+(pde*dist_err[ii])**2+(epde*dist_err[ii])**2 ))
            dw = np.sqrt( (TA[2][0]*ev)**2.+(TA[2][1]*K)**2*((epra*dist[ii])**2+(pra*dist_err[ii])**2+(epra*dist_err[ii])**2) + (TA[2][2]*K)**2*((epde*dist[ii])**2+(pde*dist_err[ii])**2+(epde*dist_err[ii])**2 ))

            up = Rot_DFin[0][0]*(u-U) + Rot_DFin[0][1]*(v-V) + Rot_DFin[0][2]*(w-W) + U
            vp = Rot_DFin[1][0]*(u-U) + Rot_DFin[1][1]*(v-V) + Rot_DFin[1][2]*(w-W) + V
            wp = Rot_DFin[2][0]*(u-U) + Rot_DFin[2][1]*(v-V) + Rot_DFin[2][2]*(w-W) + W

            dup = np.sqrt((Rot_DFin[0][0]*du)**2 + (Rot_DFin[0][1]*dv)**2 + (Rot_DFin[0][2]*dw)**2 )
            dvp = np.sqrt((Rot_DFin[1][0]*du)**2 + (Rot_DFin[1][1]*dv)**2 + (Rot_DFin[1][2]*dw)**2 )
            dwp = np.sqrt((Rot_DFin[2][0]*du)**2 + (Rot_DFin[2][1]*dv)**2 + (Rot_DFin[2][2]*dw)**2 )

            Uerr = np.sqrt(dup**2. + dU**2.) 
            Verr = np.sqrt(dvp**2. + dV**2.)
            Werr = np.sqrt(dwp**2. + dW**2.)

            likeU = np.exp(-0.5*(up-U)**2./Uerr**2.) / Uerr / np.sqrt(2.*np.pi)
            likeV = np.exp(-0.5*(vp-V)**2./Verr**2.) / Verr / np.sqrt(2.*np.pi)
            likeW = np.exp(-0.5*(wp-W)**2./Werr**2.) / Werr / np.sqrt(2.*np.pi)

            like =  rvpdf[jj] * distpdf[ii] * likeX * likeY * likeZ * likeU * likeV * likeW * drv * ddist    

            resultarr[iid+1,jjr+1] = like
    likearr = np.array(resultarr[1:,1:])

   
    idx = np.where(likearr == np.max(likearr))
    statdist = resultarr[idx[0]+1,0]
    statrv = resultarr[0,idx[1]+1]

    result = [likearr.sum(), statdist ,statrv]

#    return result
    return result[0],result[1],result[2],likeX,likeY,likeZ,likeU,likeV,likeW
    #return result[0],likeX,likeY,likeZ,likeU,likeV,likeW#,x,y,z,u,v,w,rv,dist,pmra_obs,pmde_obs

