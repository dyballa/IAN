from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.linalg import norm

#TODO: https://mathworld.wolfram.com/HyperspherePointPicking.html

limbs = np.array([[257, 186],       [245, 201],       [263, 202],       [273, 196],       [277, 210],
       [258, 213],       [253, 227],       [265, 226],       [282, 222],       [261, 241],      
        [275, 237],       [289, 237],       [273, 252],       [277, 262],       [288, 253],       
        [299, 249],       [299, 264],       [289, 269],       [244, 215],       [290, 279],       
        [302, 278],       [310, 268],       [291, 290],       [297, 300],       [307, 291],       
        [317, 282],       [321, 270],       [331, 273],       [331, 285],       [324, 296],       
        [312, 306],       [300, 314],       [311, 327],       [316, 318],       [328, 311],       
        [337, 299],       [346, 291],       [345, 278],       [360, 282],       [356, 293],       
        [350, 305],       [340, 316],       [333, 325],       [321, 329],       [330, 337],       
        [348, 334],       [350, 322],       [363, 319],       [360, 309],       [367, 301],       
        [380, 295],       [370, 286],       [314, 338],       [298, 325],       [298, 337],       
        [287, 324],       [274, 329],       [281, 341],       [291, 352],       [297, 360],       
        [278, 368],       [262, 350],       [254, 330],       [236, 324],       [237, 343],       
        [238, 362],       [257, 373],       [257, 389],       [235, 385],       [215, 370],       
        [213, 348],       [204, 327],       [188, 338],       [177, 357],       [202, 385],       
        [172, 381],       [193, 416],       [213, 401],       [212, 424],       [237, 411],       
        [224, 443],       [183, 452],       [166, 424],       [134, 421],       [139, 397],       
        [141, 357],       [163, 335],       [160, 310],       [126, 322],       [ 98, 356],       
        [111, 383],       [ 91, 419],       [119, 449],       [151, 450],       [155, 479],       
        [186, 478],       [210, 469],       [181, 511],       [137, 527],       [124, 492],       
        [ 82, 507],       [ 81, 470],       [ 36, 454],       [ 54, 420],       [ 57, 380],       
        [ 28, 341],       [ 67, 314],       [ 14, 394],       [314, 352],       [270, 179],       
        [254, 167],       [239, 179],       [235, 161],       [223, 174],       [229, 193],       
        [215, 187],       [224, 209],       [330, 352],       [344, 349],       [355, 361],       
        [363, 346],       [363, 331],       [378, 326],       [377, 314],       [387, 307],       
        [391, 320],       [342, 263],       [359, 266],       [373, 272],       [386, 281],       
        [393, 294],       [328, 262],       [314, 253],       [315, 239],       [307, 229],       
        [297, 223],       [296, 205],       [288, 194],       [289, 174],       [279, 162],       
        [264, 152],       [250, 146],       [213, 199],       [298, 184],       [235, 221],       
        [326, 249],       [304, 194],       [307, 210],       [370, 358],       [387, 346],       
        [377, 341],       [407, 334],       [399, 357],       [378, 368],       [383, 384],       
        [403, 384],       [409, 369],       [420, 354],       [436, 348],       [426, 369],       
        [427, 392],       [413, 394],       [394, 401],       [406, 419],       [423, 409],       
        [440, 415],       [448, 402],       [447, 381],       [456, 372],       [451, 355],       
        [461, 349],       [470, 362],       [461, 386],       [465, 404],       [452, 428],       
        [426, 432],       [393, 272],       [376, 259],       [391, 257],       [401, 245],       
        [412, 234],       [423, 226],       [434, 224],       [448, 222],       [461, 222],       
        [473, 222],       [490, 219],       [498, 209],       [502, 194],       [502, 181],       
        [495, 170],       [487, 163],       [477, 158],       [468, 157],       [456, 159],       
        [445, 163],       [432, 166],       [420, 164],       [412, 154]])

horseshoe = np.array([[262. ,  93. ],  [243. ,  93. ],  [245. , 118. ],  [217. , 120. ],  [226. , 139. ],       
    [218. , 104. ],       [181. , 128. ],       [195. , 140. ],       [211. , 159. ],       [181. , 172. ],   
        [174. , 157. ],       [195. , 168. ],       [164.5, 178. ],       [159. , 146. ],       [154. , 190. ],
               [179. , 198. ],       [193. , 183. ],       [193. , 204. ],       [170. , 225. ],       
               [158. , 217. ],       [163. , 202. ],       [139. , 202. ],       [142. , 178. ],       
               [144. , 224. ],       [147. , 242. ],       [161. , 253. ],       [176. , 241. ],       
               [189. , 219. ],       [197. , 240. ],       [203. , 229. ],       [182. , 255. ],       
               [180. , 271. ],       [197. , 279. ],       [200. , 269. ],       [205. , 254. ],       
               [218. , 248. ],       [230. , 245. ],       [229. , 258. ],       [218. , 268. ],       
               [217. , 284. ],       [230. , 288. ],       [241. , 279. ],       [244. , 272. ],       
               [258. , 261. ],       [241. , 258. ],       [257. , 281. ],       [248. , 294. ],       
               [265. , 273. ],       [271. , 283. ],       [284. , 280. ],       [296. , 285. ],       
               [307. , 283. ],       [317. , 286. ],       [327. , 293. ],       [337. , 299. ],       
               [329. , 286. ],       [329. , 272. ],       [340. , 272. ],       [347. , 288. ],       
               [369. , 288. ],       [359. , 279. ],       [358. , 294. ],       [375. , 274. ],       
               [358. , 265. ],       [346. , 254. ],       [373. , 253. ],       [363. , 240. ],       
               [385. , 253. ],       [395. , 241. ],       [385. , 231. ],       [388. , 264. ],       
               [405. , 230. ],       [397. , 219. ],       [374. , 221. ],       [386. , 204. ],       
               [408. , 206. ],       [401. , 195. ],       [419. , 195. ],       [416. , 172. ],       
               [403. , 178. ],       [393. , 186. ],       [424. , 184. ],       [398. , 164. ],       
               [412. , 151. ],       [389. , 145. ],       [390. , 123. ],       [374. , 107. ],       
               [381. ,  95. ],       [396. ,  90. ],       [394. , 105. ],       [403. , 100. ],       
               [409. , 120. ],       [401. , 128. ],       [421. , 133. ],       [409. , 137. ],       
               [424. , 150. ],       [380. , 119. ],       [400. , 148. ],       [427. , 162. ],       
               [418. , 213. ],       [389. , 132. ],       [207. , 143. ],       [200. , 120. ],       
               [188. , 156. ],       [150. , 161. ]])

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

#https://plotly.com/python/3d-mesh/#intensity-values-defined-on-vertices-or-cells
def loadOFF(filename,path='toolbox_graph'):
    if filename[-4:].lower() != '.off':
        filename += '.off'
    
    with open(f"{path}/{filename}") as f: # open the file for reading
        empty = 0
        for li,line in enumerate(f): # iterate over each line

            if li == 0:
                print(line)
                assert line.strip() == "OFF"
                continue
            if li == 1:
                n = int(line.split()[0])
                X = np.zeros((n,3))
                continue
            if not line.strip(): #skip empty lines
                empty += 1
                continue
            if li-2+empty >= n:
                break

            X[li-2,[0,2,1]] = list(map(float,line.split())) # split it by whitespace
    X -= X.mean(0)
    return X

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

def dset_noisycatcurve(arclen=2,a=.8,N=100,seed=0,plot=False):
    # arclen = 2
    # a = .8
    # N = 100
    np.random.seed(seed)
    s = arclen*np.random.rand(N)
    sx = a*np.log((s + np.sqrt((s**2 + a**2)))/a)
    sy = a*np.cosh(np.log((s + np.sqrt((s**2 + a**2)))/a))

    sx_ = -sx[N//2:][::-1]
    sy_ = sy[N//2:][::-1]
    mirroredx = np.r_[sx_,sx[:N//2]]
    mirroredy = np.r_[sy_,sy[:N//2]]
    s = np.r_[-s[N//2:][::-1],s[:N//2]]

    data = np.array([mirroredx,mirroredy]).T
    data -= data.mean(0)
    colors = s
    
    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[0,1]}

def dset_noisycatplane(arclen=1.5,Zlen=2,a=.2,N=1000,seed=0,plot=False):
    # arclen = 2
    # Zlen = 2
    # a = .2
    # N = 1000
    np.random.seed(seed)
    s = arclen*np.random.rand(N)
    sx = a*np.log((s + np.sqrt((s**2 + a**2)))/a)
    sy = a*np.cosh(np.log((s + np.sqrt((s**2 + a**2)))/a))

    sx_ = -sx[N//2:][::-1]
    sy_ = sy[N//2:][::-1]
    mirroredx = np.r_[sx_,sx[:N//2]]
    mirroredy = np.r_[sy_,sy[:N//2]]
    s = np.r_[-s[N//2:][::-1],s[:N//2]]

    Z = Zlen*np.random.rand(N)

    data = np.array([mirroredx,mirroredy,Z]).T
    data -= data.mean(0)
    colors = s

    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[0,2]}

def dset_gridcatplane(arclen=1.5,Zlen=2,a=.2,n=13,noise_std=0,seed=0,plot=False):
    # arclen = 1.5
    # Zlen = 2
    # a = .2
    # n = 13
    # N = 2*int(Zlen*arclen*n*n) - 1
    # noise_std = 0.02
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

def dset_swisscheese(params = [(1,.35,-1.25,.35),
                                 (-1,.2,0,.2),
                                 (.75,.3,1.25,.3)],max_prob = 1, noise_std=0,seed=0,plot=False):
    np.random.seed(seed)

    rng = np.linspace(-2,2,40)
    x,y = np.meshgrid(rng,rng)
    exps = np.array([1-.5*np.exp(-(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy**2)) for x0,sx,y0,sy in params])
    probs = np.sum(exps,axis=0)
    probs = (probs-probs.min())/(probs.max()-probs.min())
    
    
    samples = np.random.rand(*probs.shape) < max_prob*probs
    X,Y = x[samples].flatten(), y[samples].flatten()
    data = np.array([X,Y]).T

    data -= data.mean(0)
    if noise_std: data += np.random.randn(*data.shape)*noise_std*(np.diff(rng)[0])
    colors = X

    if plot:
        print(data.shape)
        plt.imshow(probs); plt.show()
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[0,1]}

def dset_dumbbell2d(step=.15,startd=-.75,endd=2.,sig=.5,scl=1.25,noise_std=0,plot=False):

    #maxrs = np.r_[np.arange(0,np.round(1+step,2),step)]#,np.arange(1,-step,-step)]
    ds = np.arange(startd,endd+step,step)
    maxrs = scl*np.exp(-ds**2/(2*sig**2))
    maxrs = np.r_[maxrs,maxrs[::-1]]

    X, Y, Z = None, None, None
    lastn = 0
    for zi,maxr in enumerate(np.round(maxrs,2)):
        z = zi*step

        if maxr <= step:
            rs = [step]
        elif zi == 0 or zi == len(maxrs)-1:
            rs = np.arange(0,maxr,step)
        else:
            rs = [maxr]#np.arange(0,maxr,step)
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
        
        lastn = 0

    N = X.shape[0]
    if noise_std: np.random.seed(seed)
    data = np.stack([X, Y, Z],axis=1) + np.random.random((N,3))*noise_std
    data -= data.mean(0)
    colors = Z
    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[1,2]}

def dset_dumbbell1d_3d(step=.075,dstep=.15,startd=.25,endd=2.,sig=.5,scl=.5,noise_std=0,plot=False):
    step = .075

    #maxrs = np.r_[np.arange(0,np.round(1+step,2),step)]#,np.arange(1,-step,-step)]
    ds = np.arange(startd,endd+dstep,dstep)
    maxrs = scl*np.exp(-ds**2/(2*sig**2))
    maxrs = np.r_[maxrs,maxrs[::-1]]

    X, Y, Z = None, None, None
    lastn = 0
    for zi,maxr in enumerate(np.round(maxrs,2)):

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
        #print(X.shape[0]-lastn,end=',')
        lastn= 0#X.shape[0]
    if noise_std: np.random.seed(seed)
    data = np.stack([X, Y, Z],axis=1) + np.random.random((N,3))*noise_std
    data -= data.mean(0)
    colors = Z

    if plot:
        print(data.shape)
        plot3dwithColorbar(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[1,2]}

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

def dset_ring(param_noise = True,gauss_noise_std = 0,seed = 0,N = 100,plot=False):

    np.random.seed(seed)
    if param_noise:
        ts = np.random.random(N)*2*np.pi
        ts.sort()
    else:
        ts = np.arange(0,2*np.pi,2*np.pi/N)
    data = np.concatenate([np.cos(ts)[:,None],np.sin(ts)[:,None]],1)

    if gauss_noise_std > 0:
        data += np.random.randn(*data.shape)*gauss_noise_std
    data -= data.mean()
    print(data.shape)
    from matplotlib.colors import hsv_to_rgb
    colors = [hsv_to_rgb([i/float(N),.8,.8]) for i in range(N)]
    
    if plot:
        print(data.shape)
        plot2dScatter(data,colors); plt.show()

    return {'X':data,'c':colors,'2d_coords':[0,1]}

# set parameters
def dset_spiral(global_noise_std=0,m=300,seed=10,normal_noise=0,length_phi=15,plot=False):

    np.random.seed(seed)
    # create dataset
    phi = length_phi*np.random.random(m)
    xi = np.random.random(m)

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

#https://pydiffmap.readthedocs.io/en/master/jupyter%20notebook%20tutorials/Swiss_Roll/Swiss_Roll.html
def dset_swissroll(m=2000,length_phi=15,length_Z=15,sigma=0.1,scale=1/6.,plot=False):
    # set parameters
    # length_phi = 15   #length of swiss roll in angular direction
    # length_Z = 15     #length of swiss roll in z direction
    # sigma = 0.1       #noise strength
    # m = 2000         #number of samples

    # create dataset
    phi = length_phi*np.random.rand(m)
    xi = np.random.rand(m)
    Z = length_Z*np.random.rand(m)
    X = scale*(phi + sigma*xi)*np.sin(phi)
    Y = scale*(phi + sigma*xi)*np.cos(phi)

    data = np.array([X, Y, Z]).T
    data -= np.mean(data,0)

    colors = phi
    if plot:
        print(data.shape)
        plot3dwithColorbar(data,colors); plt.show()
    return {'X':data,'c':colors,'2d_coords':[1,2]}

def dset_gridswissroll(h=20,dens=45,ExParam=1.,noise_std=0,seed=0,plot=False):

    tt = np.tile(np.linspace(1.5,8.5,dens),h)[:,None]
    tt.sort()
    height = h*np.repeat(np.linspace(0.1,.9,dens),h)[:,None]#np.random.random((N,1))
    data = np.concatenate([tt*np.cos(tt), height, ExParam*tt*np.sin(tt)],axis=1)
    data -= data.mean(0)
    if noise_std > 0:
        np.random.seed(seed)
        data += np.random.randn(*data.shape)*noise_std

    colors = tt.flatten()
    if plot:
        print(data.shape)
        plot3dwithColorbar(data,colors); plt.show()
    return {'X':data,'c':colors,'2d_coords':[1,2]}


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



