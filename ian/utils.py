import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches
import matplotlib
import scipy as sp
import time
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import KDTree
import sys

################## UTILS ##################


def subps(nrows, ncols, rowsz=3, colsz=4, d3=False, axlist=False):
    """ Create figure and axes objects for an nrows-by-ncols grid of subplots.
    Setting `d3` to True enables 3d projection.
    Set `axlist` to True  if you want a list of axes returned even for a single subplot. """
    if d3:
        f = plt.figure(figsize=(ncols*colsz,nrows*rowsz))
        axes = [[f.add_subplot(nrows,ncols,ri*ncols+ci+1, projection='3d') for ci in range(ncols)] \
                for ri in range(nrows)]
        #make one-dimensional if there is a single row
        if nrows == 1:
            axes = axes[0]
            if ncols == 1:
                axes = axes[0]
    else:
        f,axes = plt.subplots(nrows,ncols,figsize=(ncols*colsz,nrows*rowsz))

    if axlist and ncols*nrows == 1:
        axes = [axes]
    return f,axes

def getTri(Adj):
    """ Returns a copy of a 2-d array with its lower triangular entries zeroed out.
    Useful for removing redudant entries in an undirected graph's adjacency matrix. """
    Adjtri = Adj.copy()
    Adjtri[np.tril_indices(Adj.shape[0])] = 0
    return Adjtri

def my_kendalltau(y):
    
    # W.R. Knight, "A Computer Method for Calculating Kendall's Tau with
    # Ungrouped Data", Journal of the American Statistical Association, Vol. 61,
    # No. 314, Part 1, pp. 436-439, 1966.
    i = 0; j = 0
    s = 0
    n = len(y)

    while i < n:
        j = i + 1
        while j < n:
            if y[i] > y[j]:
                s += 1
            j += 1
        i += 1

    tau = 1 - (4.*s/(n*(n-1.)))
    return tau

def plotDataGraph(X, edges, c=None, nodecmap=None, s=5, plot3d=False, edge_color=[.753,.753,.753,.8],
    f_ax=None, colorbar=False, nodeis=False, subset=[], label=None, noTicks=True, axisEqual=True,
    highlight_edges=[], hl_color=[1,0,0,1], hl_style='-.', hl_width=1, edge_style='-', edge_width=1, **kwargs):
    
    """ Plots an unweighted graph with node positions given by `X`. 
    `edges` can be a list of edges or an adjacency matrix. """

    if type(edges) == np.ndarray and edges.ndim == 2:
        assert edges.shape[0] == edges.shape[1]
        edges = list(zip(*np.nonzero(getTri(edges))))
    elif sp.sparse.issparse(edges):
        assert edges.ndim == 2 and edges.shape[0] == edges.shape[1]
        edges = list(zip(*sp.sparse.triu(edges,k=1).nonzero()))
    
    if f_ax is None:
        f,ax = subps(1,1,d3=plot3d)
    else:
        f,ax = f_ax
    plotX = X
    if subset:
        assert type(subset) == list
        plotX = X[subset]
    if plot3d:
        sc = ax.scatter(*plotX[:,:3].T,s=s,c=c,cmap=nodecmap,label=label,**kwargs)
        if axisEqual:
            ax.auto_scale_xyz(*[[X.min(),X.max()]]*3)
        else:
            ax.auto_scale_xyz(*[[X[:,ci].min(),X[:,ci].max()] for ci in range(3)])
    else:
        sc = ax.scatter(*plotX[:,:2].T,s=s,c=c,cmap=nodecmap,label=label,**kwargs)
        if axisEqual:
            ax.axis('equal')
    if nodeis:
        if subset:
            for i in subset:
                ax.text(*X[i],f'{i}')
        else:
            for i in range(X.shape[0]):
                ax.text(*X[i],f'{i}')
    if colorbar:
        f.colorbar(sc,ax=ax)

    if edge_color is not None:
        line_list = []
        edge_colors = []
        edge_styles = []
        edge_widths = []
        for xi,xj in edges:
            if subset and (xi not in subset or xj not in subset): continue
            
            if (xi,xj) in highlight_edges:
                edge_colors.append(hl_color)
                edge_styles.append(hl_style)
                edge_widths.append(hl_width)
            else:
                edge_colors.append(edge_color)
                edge_styles.append(edge_style)
                edge_widths.append(edge_width)
            if plot3d:
                line_list.append([(X[xi,0],X[xi,1],X[xi,2]),(X[xj,0],X[xj,1],X[xj,2])])
            else:
                line_list.append([(X[xi,0],X[xi,1]),(X[xj,0],X[xj,1])])
        if plot3d:
            lines = Line3DCollection(line_list, colors=edge_colors, linewidths=edge_widths, linestyles=edge_styles, zorder=-1)
        else:
            lines = LineCollection(line_list, colors=edge_colors, linewidths=edge_widths, linestyles=edge_styles, zorder=-1)
        ax.add_collection(lines)

    if noTicks:
        ax.set(xticks=[],yticks=[]);
        if plot3d: ax.set(zticks=[]);
    return (f,ax)

def plotlyGraph(pos,edges,node_names,node_color="#0343df",colorscale=None,node_size=5):
    import plotly.graph_objects as go
    edges = list(zip(*sp.sparse.triu(edges,k=1).nonzero()))
    
    
    traces = []
    #EDGES TRACE
    x_edges = []; y_edges = []; z_edges = [];
    for (u,v) in edges:
#         if (u,v) in hl_edges: continue
        x_edges += [pos[u,0], pos[v,0], None]
        y_edges += [pos[u,1], pos[v,1], None]
        z_edges += [pos[u,2], pos[v,2], None]
    traces.append( go.Scatter3d(x=x_edges,
                    y=y_edges,
                    z=z_edges,
                    mode='lines',
                    line=dict(color='black', width=2),
                    hoverinfo='none')
                 )
    traces.append( go.Scatter3d(x=pos[:,0],
                             y=pos[:,1],
                            z=pos[:,2],
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=node_size,
                                        color=node_color, #color the nodes according to their community
                                        #colorscale=['lightgreen','magenta'], #either green or mageneta
                                        line=dict(color='black', width=0.5)),
                            hovertext=node_names,
                            hoverinfo='text')
                 )
    #also need to create the layout for our plot
    dcranges = []
    assert pos.shape[1] == 3
    for dc in range(3):
        rng = pos[:,dc].max() - pos[:,dc].min()
        dcranges.append( (pos[:,dc].min()-rng*.05, pos[:,dc].max()+rng*.05) )
    showticklbl = False
    layout = go.Layout(title="",
                    height=1000,
                    showlegend=False,
                    scene={
                        "aspectmode":'data',
#                             "aspectratio":dict(x=dcranges[0][1]-dcranges[0][0],
#                                          y=dcranges[1][1]-dcranges[1][0],
#                                         z=dcranges[2][1]-dcranges[2][0]),
                       "xaxis": dict(showbackground=True,showline=True,zeroline=False,
                showgrid=True,showticklabels=showticklbl,title='', range=dcranges[0]),
                       "yaxis": dict(showbackground=True,showline=True,zeroline=False,
                showgrid=True,showticklabels=showticklbl,title='', range=dcranges[1]),
                       "zaxis": dict(showbackground=True,showline=True,zeroline=False,
                showgrid=True,showticklabels=showticklbl,title='', range=dcranges[2])},
                    margin=dict(t=100),
                    hovermode='closest')

    fig = go.Figure(data=traces, layout=layout)

    fig.show()
    return

def plotWeightedGraph(myX, K, nodecolors=None, nodecmap=None, s=5, edge_width=1,
                f_ax=None, wthresh=1e-2, noTicks=True, rgb=(0,0,0), maxw=1, nodeLbl=None):

    """ Plots a weighted graph using the weights as the alpha opacity channel for the edges."""

    if f_ax is None:
        f,ax = subps(1,1)
    else:
        f,ax = f_ax

    if sp.sparse.issparse(K):
        wedges = list(zip(*sp.sparse.triu(K,k=1).nonzero()))
    else:    
        Ktri = getTri(K)
        Ktri[Ktri < wthresh] = 0 #don't show edges too thin to see, for speed
        wedges = list(map(tuple,zip(*np.nonzero(Ktri))))
    
    if len(wedges) > 50000:
        print(f'Too many edges ({len(wedges)}), this may take a long time to plot...', flush=True)

    lines = LineCollection([[(myX[xi,0],myX[xi,1]) for xi in e] for e in wedges],
                           colors=[(rgb[0],rgb[1],rgb[2],K[xi,xj]/maxw) for (xi,xj) in wedges], lw=edge_width, zorder=-1)
    ax.add_collection(lines)
    ax.scatter(*myX[:,:2].T, s=s, c=nodecolors, cmap=nodecmap, label=nodeLbl)
    ax.axis('equal')
    if noTicks:
        ax.set(xticks=[], yticks=[])
    return f,ax

def pwdists(X, square=True, sqdists=False):

    """Get pairwise Euclidean distances from an N-by-p data matrix.
    A square matrix is returned if `square` is True (default).
    Squared distances are returned if `sqdists` is True. """

    ds = pdist(X, 'sqeuclidean' if sqdists else 'euclidean')
    if square:
        return squareform(ds)
    return ds


def knndists(D1, k, skip_self=False, sqdists=False):

    """Get an N-by-k matrix of distances to k nearest neighbors from either
    an N-by-p data matrix or an N-by-N matrix of (non-squared) distances."""

    assert type(skip_self) == bool
    skip_self = int(skip_self)

    if D1.shape[1] != D1.shape[0]:
        #assume data matrix
        X = D1
        tree = KDTree(X)
        dists,ind = tree.query(X, k=k+skip_self)
        ind = ind[:,skip_self:]
        D1k = dists[:,skip_self:]
        if sqdists:
            np.square(D1k,out=D1k)
    else:
        D1k = np.zeros((D1.shape[0],k))
        ind = np.zeros_like(D1k, dtype='int64')
        for i in range(D1.shape[0]):
            argsrt_i = D1[i].argsort()
            ind[i] = argsrt_i[skip_self:skip_self+k]
            D1k[i] = D1[i,ind[i]]
    return D1k,ind



def plotScales(X,A,optsigs,circleColor='g',node_is=None,cmap=None,sigcolors=None,nLvls=1,scalesScl=1,
                f_axes=None,showGraph=True,nodeLabels=False,subset=[],stats_minmax=None,sigstyle=None,circleLabel=None):
    
    lvl_styles = [('-',1.),(':',1),('--',.75)]
    assert nLvls <= len(lvl_styles)
    if f_axes is None:
        naxes =  2 if (showGraph and len(A) > 0) else 1
        f,axes = subps(1,naxes,6,6)
        if naxes == 1:
            axes = [axes]
    else:
        f,axes = f_axes
    assert len(axes) >= 1
    ax0 = axes[0]
    if showGraph:
        plotDataGraph(X,A,f_ax=(f,ax0),subset=subset,c='k',alpha=.8)
        #ax0.scatter(*X[subset][:,:2].T,s=15,c='k',alpha=.8)

    if len(subset) == 0:
        subset = range(X.shape[0])    
    if cmap is not None:
        if sigcolors is None:
            sigcolors = optsigs
        if stats_minmax is not None:
            vmin, vmax = stats_minmax
        else:
            vmin, vmax = min(sigcolors), max(sigcolors)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.get_cmap(cmap)
        sigColors = [cmap(norm(si)) for si in sigcolors]
    else:
        sigColors = [circleColor]*X.shape[0]
    
    for ii,xi in enumerate(subset):
        si_opt = optsigs[xi]
        
        radius = si_opt*scalesScl
        for ll,li in enumerate(range(1,nLvls+1)):
            lvl_radius = li/nLvls*radius
            ls,lw = lvl_styles[ll]
            if sigstyle is not None:#overrride
                ls = sigstyle
            circle = mpatches.Circle(X[xi,:2], lvl_radius, color=sigColors[xi], 
                ls=ls,lw=lw,fill=False,label=circleLabel if ii+ll == 0 else None)
            ax0.add_patch(circle)
    if circleLabel is not None:
        ax0.legend()

    if nodeLabels:
        for xi in subset:
            ax0.text(*X[xi,:2],str(node_is[xi]) if node_is is not None else str(xi))

    ax0.set(xticks=[],yticks=[])#,title='Optimal scales')

    #also plot midpt-centered sigmas with half radius
    if len(axes) > 1:

        ax1 = axes[1]
        ax1.scatter(*X[:,:2].T,s=15); ax1.axis('equal')
        for ei,(xi,xj) in enumerate(edge_list):
            if (xi not in subset or xj not in subset): continue
            radius = scalesScl*np.sqrt(optsigs[xi]*optsigs[xj])/2
            midpt = (X[xi,:2]+X[xj,:2])/2
            circle = mpatches.Circle(midpt, radius, color=circleColor, ls='-', fill=False,
                label=circleLabel if ei == 0 else None)
            ax1.add_patch(circle)
            # if ei == 0 and circleLabel is not None:
            #     ax1.legend([circle], [circleLabel])
            ax1.plot([X[xi,0],X[xj,0]],[X[xi,1],X[xj,1]],c='silver',alpha=.8,lw=1,zorder=-1)
        if nodeLabels:
            for xi in subset:
                ax1.text(*X[xi,:2],str(node_is[xi]) if node_is is not None else str(xi))

        ax1.set(xticks=[],yticks=[])
        if circleLabel is not None:
            ax1.legend()
    if axes is None:
        plt.show()


def getMultiScaleK(D2, optSigs, sig2scl=1., disc_pts=[], tol=1e-8):

    """ Produces a similarity matrix using a multi-scale Gaussian kernel.
    D2 is a square matrix of squared pairwise distances, and optSigs is a list
    of individual scales. sig2scl is the scalar multiple in the denominator of the
    Gaussian exponential. """

    assert D2.ndim == 2 and D2.shape[0] == D2.shape[1]
    N = D2.shape[0]

    K = np.eye(N)

    nonzero_sigs = optSigs.copy()
    nonzero_sigs[np.isclose(nonzero_sigs,0)] = 1

    #if exp(-p) < eps, set p to inf (exp val -> 0) to avoid underflow warnings
    eps = 2*np.finfo(D2.dtype).eps
    maxpow = -np.log(eps)

    for i in range(N-1):
        if i in disc_pts: continue

        try:
            assert D2[i,i+1:].min() > eps
        except:
            print(i)
            print(list(D2[i]))
            raise ValueError(f'Nearly identical points detected: {i} and {D2[i,i+1:].argmin()+i+1}.')
        try:
            assert optSigs[i] > eps
        except:
            print(i,optSigs[i],eps)
            print(list(optSigs))
            raise ValueError(f'Sigma nearly zero for non-disconnected point {i}.')
        
        power = D2[i,i+1:]/(sig2scl*nonzero_sigs[i]*nonzero_sigs[i+1:])
        # res = np.zeros(power.size)
        # mask = power > maxpow
        # res[mask] = 0
        # res[~mask] = np.exp(-power[~mask])
        res = np.zeros(power.size,dtype=D2.dtype)
        try:
            np.exp(-power, where=power < maxpow, out=res)#avoid underflow warnings
        except:
            print(optSigs)
            print(power)

        
        res[res < tol] = 0


        K[i,i+1:] = res #upper triang
        K[i+1:,i] = K[i,i+1:]#symmetrize


    return K

