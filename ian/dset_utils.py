from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.linalg import norm


plasma_clrs = [(0.9411764705882353, 0.9764705882352941, 0.12941176470588237),
 (0.9882352941176471, 0.7078431372549019, 0.1882352941176471),
 (0.9294117647058824, 0.4745098039215686, 0.3254901960784314),
 (0.7941176470588235, 0.2784313725490196, 0.4725490196078431),
 (0.611764705882353, 0.09019607843137255, 0.6196078431372549),
 (0.3607843137254902, 0.00784313725490196, 0.6411764705882353),
 (0.050980392156862744, 0.03137254901960784, 0.5294117647058824)]
nclrs = len(plasma_clrs)
frac = int(256/nclrs)

newcolors = []
for i,clr in enumerate(plasma_clrs):
    if i == nclrs-1 and frac*nclrs < 256:
        frac += 256-frac*nclrs
    newcolors += [clr]*frac
assert len(newcolors) == 256
plasma7r_cmp = ListedColormap(newcolors)

def plot2dScatter(X,colors=None,f_ax=None,axisEqual=True,noTicks=True,colorbar=False,cmap=None,s=10,
    barlims=None,axLabels=True,**kwargs):
    X = X[:,:2]
    if f_ax is None:
        f,ax = plt.subplots(1,1,figsize=(4,3))
    else:
        f,ax = f_ax
    if colors is None:
        colors = np.zeros(X.shape[0])

    sc = ax.scatter(*X.T,s=s,c=colors,cmap=cmap,**kwargs)
    if axLabels:
        ax.set(xlabel='x',ylabel='y')
    if noTicks:
        ax.set(xticks=[],yticks=[])
    if axisEqual:
        ax.axis('equal')
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cbar = f.colorbar(sc, cax=cax, orientation='vertical')
        #cbar.ax.tick_params(labelsize=8) 
    return ax

def clipped_cmap(cmap_name, vmin, vmax, vmin_clip=None, vmax_clip=None):
    """
    Return a clipped but color-mapping preserving Norm and Colormap.

    The returned Norm and Colormap map data values to the same colors as
    would  `Normalize(vmin, vmax)`  with *cmap_name*, but values below
    *vmin_clip* and above *vmax_clip* are mapped to under and over values
    instead.
    """
    if vmin_clip is None:
        vmin_clip = vmin
    if vmax_clip is None:
        vmax_clip = vmax
    
    assert vmin <= vmin_clip < vmax_clip <= vmax
    cmin = (vmin_clip - vmin) / (vmax - vmin)
    cmax = (vmax_clip - vmin) / (vmax - vmin)

    big_cmap = cm.get_cmap(cmap_name, 512)
    new_cmap = ListedColormap(big_cmap(np.linspace(cmin, cmax, 256)))
    new_norm = Normalize(vmin_clip, vmax_clip)
    return new_norm, new_cmap

def plot3dScatter(X,colors=None,f_ax=None,figsize=(7,7),colorbar=False,axisEqual=True,
                       noTicks=True,cmap='viridis',s=5,colorbar_lbl='',barlims=None,angle=(40,60),
                       depthshade=False,axLabels=True,myAxisScl=None,**kwargs):
    X = X[:,:3] - X[:,:3].mean(0)
    if colors is None:
        colors = np.zeros(X.shape[0])
    if f_ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig,ax = f_ax
    norm = None
    if barlims is not None:
        norm,cmap = clipped_cmap(cmap,barlims[0],barlims[1])
    sc = ax.scatter(*X.T,s=s,c=colors,cmap=cmap,norm=norm,depthshade=depthshade,**kwargs)
    if axLabels:
        ax.set(xlabel='x',ylabel='y',zlabel='z')
    if noTicks:
        ax.set(xticks=[],yticks=[],zticks=[])
    if myAxisScl is not None:
        ax.auto_scale_xyz(*myAxisScl)
    elif axisEqual:
        ax.auto_scale_xyz(*[[X.min(),X.max()]]*3)
    else:
        ax.auto_scale_xyz(*[[X[:,ci].min(),X[:,ci].max()] for ci in range(3)])
    ax.view_init(*angle)
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical',fraction=0.02, pad=0.0, label=colorbar_lbl)
    return ax


def addZeroCol(X,n=1):
    if X.ndim == 1:#auto-add second dim for plotting
        return np.hstack([X[:,None],np.zeros((X.shape[0],n))])
    else:
        return np.hstack([X,np.zeros((X.shape[0],n))])
        
def getBaseX(side,dim,sampling,strict_side=True,seed=0):
    np.random.seed(seed)
    if sampling == 'trigrid' and dim == 1:
        sampling = 'sqgrid'
    if sampling == 'trigrid':
        _,baseX=ndtrigrid(side,dim)
    elif sampling == 'sqgrid':
        rng = np.arange(side,dtype='float64') - side//2
        coords = np.meshgrid(*([rng]*dim))
        baseX = np.hstack([c.ravel()[:,None] for c in coords])
    elif sampling == 'uniform':
        baseX = np.random.random((side**dim,dim))*(side-1) -  side//2
    elif sampling == 'normal':
        baseX = np.random.randn(*(side**dim,dim))*(side//2)
    elif sampling == 'sunflower':
        assert dim == 2
        num_pts = side**2
        indices = np.arange(0, num_pts, dtype=float) + 0.5

        r = np.sqrt(indices/num_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x,y = r*np.cos(theta), r*np.sin(theta)
        baseX = np.hstack([x[:,None],y[:,None]])
        baseX -= baseX.mean(0)
        baseX *= side//2
    else:
        raise ValueError('sampling method not recognized')

    return baseX

def ndtrigrid(side,Ndim,step=1,strict_side=True,centering=True):

    # ported to python from:
    # https://www.mathworks.com/matlabcentral/fileexchange/73041-n-dimension-regular-triangular-grid

    from numpy import sqrt,reshape,floor,ones,zeros

    #umax = side/2.; umin = -umax
    # Create grid
    v = side#abs(umax-umin);
    ndimvect, stpvect = [], []
    for p in range(1,Ndim+1):

        # Variable value computation    
        q = step*sqrt((p+1)/2/p)
        w = int(floor(v/q))

        ndimvect.append(w)
        stpvect.append(q)
    M = zeros(ndimvect+[Ndim]); # vecteur coord (X Y Z T, ...) preallocation

    #https://stackoverflow.com/questions/24432209/python-index-an-array-using-the-colon-operator-in-an-arbitrary-dimension
    def make_array_index(p, Ndim):
        subsfield = [0] * (Ndim+1)
        for j in range(Ndim):
            subsfield[j] = slice(None)
        subsfield[-1] = p
        return tuple(subsfield)

    for p in range(1,Ndim+1):
        # Variable values computation
        q = step*sqrt((p+1)/2/p); # Ndim step function of Ndim simplex edge
        w = int(floor(v/q)) # Nb samples

        s = np.arange(w)[:,None] # sampling vector
        if p >= 2:
            s = reshape(s,[1]*(p-1) + [w]); # update dimension

        samplemat = s * q; # Ndim dimension sampling vector
        newshape = ndimvect[:p-1] + [ 1 ] + ndimvect[p:]

        while samplemat.ndim < len(newshape):
            samplemat = np.expand_dims(samplemat,-1)
        assert len(newshape) == samplemat.ndim
        #https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy

        samplemat = np.tile(samplemat,newshape); # update samplemat dimension : Ndim dimension sampling matrix

        coord_index = make_array_index(p-1, Ndim)

        u = M[coord_index] + samplemat;
        M[coord_index] = u
    for p in range(1,Ndim): # (Ndim-1) succesive shifts from one dimension to the folowing
        for m in range(p):

            findvect = (np.mod(np.arange(M.shape[p]),2) == 1)[None,:]

            if p >= 2:
                findvect = reshape(findvect, np.r_[[1]*p, M.shape[p]]);


            # vector to set a vertex back to the simplex centre of the previous dimension
            shiftvect = stpvect[m]/(m+2)

            shiftmat = findvect * shiftvect;
            newshape = ndimvect[:p] + [1] + ndimvect[p+1:]

            while shiftmat.ndim < len(newshape):
                shiftmat = np.expand_dims(shiftmat,-1)
            assert len(newshape) == shiftmat.ndim


            # Enlarge vector capacity
            shiftmat = np.tile(shiftmat, newshape)

            # Update -> recursive shiftin the direction of the new dimension, (p+1)
            coord_index = make_array_index(m, Ndim)

            u = M[coord_index] + shiftmat;
            M[coord_index] = u

    if centering:
        for p in range(Ndim):

            coord_index = make_array_index(p, Ndim)
            u = M[coord_index]
            mu = u.mean()


            centered = u - mu #+ (umax+umin)/2
            M[coord_index] = centered
    if strict_side:
        subsfield = [0] * (Ndim+1)
        for j in range(Ndim):
            subsfield[j] = slice(None,side)
        subsfield[-1] = slice(None)
        M = M[tuple(subsfield)]
    for p in range(Ndim):
        coord_index = make_array_index(p, Ndim)
        Xp = M[coord_index].ravel()[:,None]
        if p == 0:
            X = Xp
        else:
            X = np.concatenate([X,Xp],axis=1)
    return M,X

def dset_gridcatplane(arclen=1.5,Zlen=2,a=.2,n=13,noise_std=0,seed=0,plot=False):

    # N = 2*int(Zlen*arclen*n*n) - 1

    if noise_std: np.random.seed(seed)
        
    s = np.tile(np.linspace(-arclen,arclen,int(2*arclen)*n),Zlen*n)
    s += np.random.randn(s.size)*noise_std
    sx = a*np.log((s + np.sqrt((s**2 + a**2)))/a)
    sy = a*np.cosh(np.log((s + np.sqrt((s**2 + a**2)))/a))


    Z = np.repeat(np.linspace(0,Zlen,Zlen*n),2*arclen*n)
    Z += np.random.randn(Z.size)*noise_std

    data = np.array([sx,sy,Z]).T
    data -= data.mean(0)
    colors = s

    Y = np.concatenate([s[:,None],Z[:,None]],1)

    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()
    
    return {'X':data,'c':colors,'2d_coords':[0,2], 'Y':Y}


def dset_crown(radius = 1, scale = 2., s=.5, step = .05,noise_std = 0.025,seed=0,plot=False):
    

    #maxrs = np.r_[np.arange(0,np.round(1+step,2),step)]#,np.arange(1,-step,-step)]
    ds = np.arange(-2,2,step)
    maxzs = scale*np.exp(-ds**2/(2*s**2))
    
    zstep = np.sqrt((1-radius*np.cos(step/4*2*np.pi))**2 + (radius*np.sin(step/4*2*np.pi))**2)# + 0.0001

    X, Y, Z = None, None, None
    all_rs = []
    for zi,maxz in enumerate(np.round(maxzs,2)):
        
        r = zi*step#arclen so far

        if maxz <= zstep:
            zs = [0]
        else:
            zs = np.arange(0,maxz,zstep)
        
        rs = [r]*len(zs)
        all_rs += rs
        rs = np.array(rs)
        
        xs = radius*np.cos(rs/4*2*np.pi)
        ys = radius*np.sin(rs/4*2*np.pi)
        
        if X is None:
            X = xs
            Y = ys
            Z = zs
        else:
            X = np.concatenate([X,xs])
            Y = np.concatenate([Y,ys])
            Z = np.concatenate([Z,zs])


    data = np.stack([X, Y, Z],axis=1)
    if noise_std > 0:
        np.random.seed(seed)
        data += np.random.randn(*data.shape)*noise_std
    data -= data.mean(0)
    colors = all_rs

    if plot:
        print(data.shape)
        plot3dwithColorbar(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[1,2]}

def dset_spintop(start=-1.5, stop=.75,scale = 1/4., s=.5, step = .075,noise_std = 0.025,seed=0,plot=False):


    ds = np.arange(start,stop+step,step)
    maxrs = scale*np.exp(-ds**2/(2*s**2))

    X, Y, Z = None, None, None
    for zi,maxr in enumerate(np.round(maxrs,2)):
    #     X, Y, Z = None, None, None
        z = zi*step

        if maxr <= step:
            rs = [0]
        else:
            rs = np.arange(0,maxr,step)
        for r in rs:
            if r == 0:
                xs, ys = np.zeros(1), np.zeros(1)
            else:
                thetas = np.linspace(0,2*np.pi,int(2*np.pi/(step/r)))[:-1]
                xs, ys = r*np.cos(thetas), r*np.sin(thetas) 
            zs = z*np.ones(xs.size)


            if X is None:
                X = xs
                Y = ys
                Z = zs
            else:
                X = np.concatenate([X,xs])
                Y = np.concatenate([Y,ys])
                Z = np.concatenate([Z,zs])


    data = np.stack([X, Y, Z],axis=1)
    if noise_std > 0:
        np.random.seed(seed)
        data += np.random.randn(*data.shape)*noise_std
    data -= data.mean(0)
    colors = Z

    if plot:
        print(data.shape)
        plot3dwithColorbar(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[1,2]}



# set parameters
def dset_gridspiral(noise_std=.01,global_noise_std=0,m=200,seed=10,normal_noise=0,length_phi=15,plot=False):
    # length_phi = 15   #length of spiral in angular direction
    # noise_std = 0.01      #noise strength along arc length
    # normal_noise = 0 #normal to spiral tangent
    # global_noise_std = 0.#25
    # m=200

    np.random.seed(seed)
    # create dataset
    phi = length_phi*np.linspace(0,1,m) + noise_std*(np.random.randn(m))
    xi = np.random.rand(m)
    #Z = length_Z*np.linspace(0,1,m)#np.random.rand(m)
    X = 1./6*(phi + normal_noise*xi)*np.sin(phi)
    Y = 1./6*(phi + normal_noise*xi)*np.cos(phi)

    data = np.array([X, Y]).T
    data -= data.mean(0)
    data += global_noise_std*np.random.random(data.shape)

    colors = phi
    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[0,1]}

from sklearn.datasets import make_blobs
def dset_2dGaussian(cluster_ns = [100,100],cluster_stds = [.5,.75],centers=None,seed=0,plot=False):

    data, y = make_blobs(n_samples=cluster_ns, centers=centers, random_state=seed, cluster_std=cluster_stds, shuffle=False)
    data -= data.mean(0)

    colors = np.concatenate([i*np.ones(cn) for i,cn in enumerate(cluster_ns)])
    colors = np.zeros(sum(cluster_ns))
    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()
    return {'X':data,'c':colors,'2d_coords':[0,1]}

def dset_2dplane(sideNx = 1., nPointsX=15,sideNy=1., nPointsY=15, param_noise=False,
    gauss_noise_std=.05, seed=0, plot=False):

    np.random.seed(seed)

    if param_noise:
        X = np.random.random(nPointsX*nPointsY) - .5
        Y = np.random.random(nPointsY*nPointsX) - .5
    else:
        X = np.linspace(-sideNx,sideNx,nPointsX)
        Y = np.linspace(-sideNy,sideNy,nPointsY)
        X = np.tile(X,nPointsY)
        Y = np.repeat(Y,nPointsX)

    line = np.concatenate([X.ravel()[:,None],Y.ravel()[:,None]],1)#,Z.ravel()[:,None]],1)
    data = line + np.random.random(line.shape)*gauss_noise_std
    data -= data.mean(0)
    colors = X
    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[0,1]}

def dset_stingray(plot=False):
    
    """This is a dataset created to look like a stingray, with a 2-d body and a 1-d tail"""

    from scipy.spatial.distance import squareform, pdist
    
    #form body by uniformly sampling points under cropped gaussian curves
    gauss = lambda x,s: np.exp(-x**2/s**2)
    body_widths = np.r_[gauss(np.linspace(-.4,0,15),.25),gauss(np.linspace(0,2,30),1)]
    body_widths -= body_widths.min()
    body_y = np.linspace(-.4,2,len(body_widths))
    body_y -= body_y.min()
    xrng = body_y.ptp()
    body_y /= xrng
    body_widths /= xrng

    #form tail from a piece of spiral
    np.random.seed(0)
    n = 40
    i = np.arange(n)
    th = np.linspace(1.05*np.pi/3,np.pi,n)
    r = np.cos(th) + 1
    scl = .3
    x, y = r*scl*np.cos(th), r*scl*np.sin(th)
    llim, rlim = 0,n
    tail_x = -x[llim:rlim]*2 + body_y.max() + (body_y[-1]-body_y[-2])
    tail_y = y[llim:rlim]

    tail_x = tail_x[::-1]
    tail_y = tail_y[::-1]

    tailX = np.hstack([tail_x[:,None],tail_y[:,None]])
    D1 = squareform(pdist(tailX))
    delta_tail = np.mean(np.diag(D1,k=1))*1.75

    #make sampling approximately uniform along tail
    to_del = []
    for i in range(1,n-1):
        for prev in range(i-1,-1,-1):
            if prev not in to_del:
                break
        if D1[i,prev] < delta_tail and D1[i,i+1] < delta_tail:
            to_del.append(i)
    tailX = np.delete(tailX,to_del,axis=0)

    tailX = tailX[:15]

    #spread out body points
    xsamples = len(body_y)
    pts = []
    for xi,x in enumerate(body_y[:-1]):
        if xi % 2 == 1: continue
        h = body_widths[xi]
        ysamples = max(1,int(round(h*xsamples)))
        yvals = np.linspace(-h,h,ysamples)
        pts += list(zip([x]*ysamples,yvals))
    #append tail
    pts += list(tailX)
    pts = np.array(pts)
    
    #subsample to add deletion noise
    np.random.seed(1)
    pct = .2
    N = pts.shape[0]
    sub = np.random.choice(range(N),int(pct*N),False)
    data = np.delete(pts,sub,axis=0)
    N = data.shape[0]
    colors = None

    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[0,1]}



