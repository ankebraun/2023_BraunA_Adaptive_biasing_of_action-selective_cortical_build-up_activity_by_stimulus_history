#!/usr/bin/env python

import pylab as pl
import numpy as np
import column # Data set specific
import history,graphics,statistics,model # general
import sys,cPickle
import pdb
from threshold import u_v

import glm

__doc__ = """A number of very high level functions


Copyright (C) 2014 Ingo Fruend

This code reproduces the analyses in the paper

    Fruend, Wichmann, Macke (2014): Quantifying the effect of inter-trial dependence on perceptual decisions. J Vis, 14(7): 9.


    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    If you use the Software for your own research, cite the paper.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Level of logmessages -- 10 allows for INFO but not for DEBUG
#                         20 will suppress INFO
import logging
logging.root.level = 10
logging.BASIC_FORMAT = '%(message)s'

# pylab configuration
pl.rcParams['font.size'] = 9.

####################################### Loading data ##############################

def load_data_file ( filename, header=False, detection=True ):
    """Load data set from file and guess initial values
    
    :Parameters:
        *filename*  name of the data file
        *header*    is the first line a header that should be skipped?
        *detection* are the data detection data? For detection data, we
                    fit a threshold nonlinearity on the stimulus values
    """
    h = history.history_impulses ()
    print 'h',h
    dfile = open ( filename, 'r' )
    if header:
        dfile.readline ()
    cdata = np.fromfile ( dfile, sep=" " ).reshape ( (-1,5) )
    data = column.ColumnData (
            cdata, impulse_responses=h, threshold=detection, ground_truth=None )
    print 'data', data
#    print 'impulse_responses', h
    conditions = np.unique ( cdata[:,1] )
    w0 = np.zeros ( len(conditions)+1, 'd' )
    for c in xrange ( len(conditions) ):
        d = np.array ( data.getsummary ( c ) )
        print 'd', d
        w = history.get_w0 ( d )
        print 'w', w
        w0[0] += w[0]
        w0[1+c] = w[1]
    w0[0] /= len(conditions)
    print 'w', w
    print 'w0', w0
    
    plotinfo = {
            'colors': ['b','m','y','c','g','r','k','darkgreen','burlywood','chartreuse','pink','brown','yellowgreen',
                      'darkred','b','m','y','c','g','r','k','darkgreen','burlywood','chartreuse','pink','brown','yellowgreen','darkred'],
            'labels': ['session 2','session 3','session 4','session 5','session 6','7','8','9','10','11','12','13','14','15',
                       '16','17','18','19','20','21','22','23','24','25','26'],
            'conditions': xrange(len(conditions)),
            'indices': [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],[0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,16],[0,17],[0,18],[0,19],[0,20],[0,21],[0,22],[0,23],[0,24],[0,25],
                       [0,26],[0,27],[0,28]],
            'xmin': data.X[:,1].min(),
            'xmax': data.X[:,1].max()
            }

    data.detection = detection
    return data,w0,plotinfo

############################## analyze data ##############################

def search_for_start ( r, X, w0, applythreshold, hf0, pm=(.85,.93,1.,1.07,1.15), storeopt=False ):
    """Search for starting values

    :Parameters:
        *d*
            data set
        *w0*
            coarse starting values
        *pm*
            increments or decrements
    """
    print 'r', r
    print 'X', X
    print 'w0', w0
    print 'applythreshold', applythreshold
    print 'hf0', hf0

    logging.info ( "Initial preoptimization" )
    Mwh = model.history_model ( r, X[:,:hf0],
            applythreshold=applythreshold,
            w0=w0,
            lm=0.1, hf0=hf0, emiter=100 )
    Mnh = model.history_model ( r, X[:,:hf0],
            applythreshold=applythreshold,
            w0=w0,
            lm=0.1, hf0=hf0, emiter=100 )
    M0  = model.history_model ( r, X[:,:hf0],
            applythreshold=applythreshold,
            w0=w0,
            lm=0.1, hf0=hf0, emiter=100 )
                    


    Mwh.w = np.concatenate ( (Mwh.w,np.zeros(6,'d')) )
    Mwh.X = X
    print 'w0', w0
    print "Mwh.w:",Mwh.w
    print "Mnh.w:",Mnh.w
    print 'Mwh.X', Mwh.X
    print 'M0.w', M0.w
    nhind = 0
    whind = 0
    i = 1

    for al in pm:
        for lm in pm:
            logging.info ( "::::: Optimizing from starting value %d :::::" % (i,) )
            w0 = M0.w.copy()

            w0[1:hf0] *= al
            p0 = M0.pi.copy()
            p0[0] *= lm;
            p0[-1] = 1-p0[0]-p0[1]
            print 'w0', w0
   #         pdb.set_trace()            
            M_ = model.history_model ( r, X,
                    applythreshold=applythreshold,
                    w0=w0, p0=p0, nu0=M0.nu,
                    lm=0.1, hf0=hf0, verbose=True, emiter=300, storeopt=storeopt )
            print 'w0', w0
            print 'M_.w', M_.w
            print 'M_.pi', M_.pi
            print 'M_.q', M_.q  
            print 'M_.loglikelihood', M_.loglikelihood   
            print 'M_.hf0', M_.hf0                   
            print "X:",X

   #         print "X[:,:hf0]:",X[:,:hf0]
   #         print "Mwh.loglikelihood:",Mwh.loglikelihood
   #         print "Mnh.loglikelihood:",Mnh.loglikelihood
   #         print "M_.loglikelihood:",M_.loglikelihood
   #         if Mwh.loglikelihood < M_.loglikelihood:
   #             logging.info ( " *model chosen for history*" )
            Mwh = M_
            whind = i
            M_ = model.history_model ( r, X[:,:hf0],
                    applythreshold=applythreshold,
                    w0=w0, p0=p0, nu0=M0.nu,
                    lm=0.1, hf0=hf0, verbose=True, emiter=300, storeopt=storeopt )
            print "M_.loglikelihood:",M_.loglikelihood
            #if Mnh.loglikelihood < M_.loglikelihood:
            #    logging.info ( "  *model chosen for independent*" )
            #    Mnh = M_
            #    nhind = i
            i += 1
            print "Mwh.w:",Mwh.w
            print "Mnh.w:",Mnh.w
            print "al:",al
            print "lm:",lm
           
    logging.info ( "Mwh.w = %s\nMnh.w = %s" % (str(Mwh.w),str(Mnh.w)) )
    logging.info ( "Mwh.ll = %g\nMnh.ll = %s" % (Mwh.loglikelihood,Mnh.loglikelihood) )
    logging.info ( "Starting values:\n  with history: %d\n  without history: %d\n" % (whind,nhind) )
    return Mnh,Mwh
def analysis ( d, w0, nsamples=0, perm_collector=statistics.EvaluationCollector ):
    """Analyze a dataset

    :Parameters:
        *d*
            a history.DataSet instance (typically a subclass of history.DataSet)
        *w0*
            starting values for the first parameters in the model. The remaining parameters will
            start at 0. It seems to be a good idea to give starting values for the stimulus dependent
            parameters in the model and leave the parameters for history features at 0.
        *nsamples*
            number of samples for monte carlo procedures
        *perm_collector*
            the collector object for the permutation tests. The default one should do for all
            experiments with a design matrix in which the first column is 1 and the second column
            refers to the slope.

    :Example:
    >>> d,w,plotinfo = load_plaid ()
    >>> results = analysis ( d, w, 10 )
    >>> results.keys()
    ['model_nohist', 'model_w_hist', 'bootstrap', 'permutation_nh', 'permutation_wh']
    """
    if d.__dict__.has_key ( 'detection' ) and d.detection:
        print 'detection'
        dnh = d.__class__ ( d.fname, threshold=len(d.th_features)>0 )
    else:
        print 'no detection'
        print 'd.fname', d.fname
        print 'd.__class__( d.fname )',d.__class__( d.fname )
        print 'd.__class__',d.__class__
        dnh = d.__class__( d.fname ) # reload the data without history
    dwh = d
    if getattr ( d, 'audio', False ):
        easy,difficult = d.performance_filter ()
    else:
        easy,difficult = None, None
    logging.info ( "Fitting models" )


    if getattr ( d, 'p0', np.array(False) ).all() :
        print 'monkey'
        # This code will only run for monkeys
        M0 = model.history_model ( dnh.r, dnh.X, applythreshold=dnh.th_features, w0=w0, p0=d.p0, lm=0.1, hf0=dnh.hf0, emiter=40 )
        Mnh = model.history_model ( dnh.r, dnh.X, applythreshold=dnh.th_features, w0=M0.w, p0=M0.pi, lm=.1, hf0=dnh.hf0, verbose=True, emiter=300 )
        logging.info ( "likelihood for independent responses: %g" % (Mnh.loglikelihood,) )
        Mwh = model.history_model ( dwh.r, dwh.X, applythreshold=dwh.th_features, w0=M0.w, p0=M0.pi, lm=.1, hf0=dwh.hf0, verbose=True, emiter=300 )
        logging.info ( "likelihood for history model: %g" % (Mwh.loglikelihood,) )
    else:
        print 'no monkey'
        Mnh,Mwh = search_for_start ( d.r, d.X, w0, d.th_features, d.hf0, storeopt=True )
        print 'Mnh,Mwh',Mnh,Mwh
        logging.info ( "likelihood for independent responses: %g" % (Mnh.loglikelihood,) )
        logging.info ( "likelihood for history model: %g" % (Mwh.loglikelihood,) )

    print "nh",Mnh.w,Mnh.pi
    print "wh",Mwh.w,Mwh.pi
    print Mwh.applythreshold
    # Monte Carlo testing
    if nsamples>0:
        r_,X_ = dwh.permutation ()
        Mnh_perm,Mwh_perm = search_for_start ( r_, X_,
                w0, d.th_features, d.hf0 )

        logging.info ( "Permutation with history" )
        dnh.rng.set_state ( dwh.rng.get_state () ) # Set the states of the two random number generators to the same values to get the exact same sequence of random numbers
# get_state: A method specifying the query to call for getting the current random-generation state (e.g., random.getstate)
# set_state: A method specifying the function to call to restore the random-generation state using the value returned by get_state (e.g., random.setstate)
        perm_collector = statistics.EvaluationCollector (
                Mwh, easy, difficult )
        permutation_wh = pl.array ( statistics.mcsimulation (
            dwh.permutation, perm_collector,
            nsamples, Mwh_perm.w, Mwh_perm.pi,
            Mwh_perm.nu, verbose=logging.root.level<20,
            hf0=dwh.hf0, applythreshold=Mwh.applythreshold ) )

        logging.info ( "Permutation without history" )
        perm_collector = statistics.EvaluationCollector (
                Mnh, easy, difficult )
        permutation_nh   = pl.array ( statistics.mcsimulation (
            dnh.permutation, perm_collector,
            nsamples, Mnh_perm.w, Mnh_perm.pi, Mnh_perm.nu,
            verbose=logging.root.level<20,
            hf0=dnh.hf0, applythreshold=Mwh.applythreshold ) )

        logging.info ( "Bootstrap" )
        kcollector = statistics.Kernel_and_Slope_Collector (
                dwh.h, dwh.hf0, slopeindex=range(1,dwh.hf0) )

#        ksumcollector = statistics.Summed_Kernel_and_Slope_Collector (
#                dwh.h, dwh.hf0, slopeindex=range(1,dwh.hf0) )


        bootstrap   = pl.array ( statistics.mcsimulation (
            dwh.bootstrap, kcollector,
            nsamples, Mwh.w, Mwh.pi, Mwh.nu,
            verbose=logging.root.level<20,
            hf0=dwh.hf0, applythreshold=Mwh.applythreshold ) )
    else:
        permutation_wh  = None
        permutation_nh  = None
        bootstrap       = None

    print "nh after permutation",Mnh.w,Mnh.pi
    print "wh after permutation",Mwh.w,Mwh.pi
#    pdb.set_trace()
    results = {
            'model_nohist': Mnh,
            'model_w_hist': Mwh,
            'permutation_wh': permutation_wh,
            'permutation_nh': permutation_nh,
            'bootstrap': bootstrap
            }

    return results

############################## Display data ##############################

def plot ( d, results, infodict ):
    """plot all results"""

    fig = pl.figure ()
    ax  = graphics.canonical_axes ( fig )

    required_infokeys = ['labels','colors','indices','conditions','xmin','xmax']
    for k in required_infokeys:
        if not k in infodict.keys():
            raise ValueError, "Key %s was not in infodict" % (k,)

    pmfplot ( d, results, infodict, ax.pmf )
    nonlinearityplot ( d, results, infodict, ax.uv )
    permutationplot ( d, results, infodict, ax.likeli )
    kernelplot ( d, results, infodict, ax.history_rz, ax.history_perf )
    slopeplot ( d, results, infodict, ax.slopes )

############################## high level plotting routines -- called internally by plot()

def pmfplot ( d, results, infodict, ax, errors=True ):
    """Generate the pmf plot"""
    wfit_mean = [0 ,0]
    w0fit_mean = [0 ,0]    
    wfit_mean_nohist = [0 ,0]
    wfit_mean_whist = [0 ,0]      
    d_mean = []
    d1_mean = []
    d2_mean = []        
    for i,c in enumerate(infodict['conditions']):
        c = int(c)
        d_ = d.getsummary ( c )
        x = pl.mgrid[infodict['xmin']:infodict['xmax']:100j]
        if len(d.th_features)>0:
            # d_[:,0] = u_v ( d_[:,0], results['model_w_hist'].nu )
            # x = u_v ( x, results['model_w_hist'].nu )
            d_[:,0] = u_v ( d_[:,0], results['model_nohist'].nu )
            x = u_v ( x, results['model_nohist'].nu )

        if errors:
            graphics.plot_data_summary ( d_, ax,
                infodict['colors'][i], infodict['labels'][i] )
        else:
           # ax.plot ( d_[:,0], d_[:,1] / d_[:,2], '.', color=infodict['colors'][i], label = infodict['labels'][i])
 # label=infodict['labels'][i]
#            ax.plot ( d_[:,0], d_[:,1].sum(c) / d_[:,2].sum(c), '.', color=infodict['colors'][i], label = infodict['labels'][i]) # label=infodict['labels'][i]
            print "d_[:,0], d_[:,1] / d_[:,2]:",d_[:,0], d_[:,1] / d_[:,2]
#            d_mean.append(d_[:,0])
#            d1_mean.append(d_[:,1])
#            d2_mean.append(d_[:,2])

#        print "d_[:,0], d_[:,1].sum() / d_[:,2].sum():",d_[:,0], d_[:,1].sum() / d_[:,2].sum()
        # wfit  = results['model_w_hist'].w[infodict['indices'][c]]
#        print 'd_mean.append(d_[:,0])', d_mean.append(d_[:,0])
#        print 'np.mean(d_mean.append(d_[:,0]), axis = 0)', np.mean(d_mean.append(d_[:,0]), axis = 0)
        wfit  = results['model_nohist'].w[infodict['indices'][i]]
        wfit_whist = results['model_w_hist'].w[infodict['indices'][i]]
        
        w0fit = results['model_w_hist'].w0[infodict['indices'][i]]
        pfit  = results['model_w_hist'].pi
        p0fit  = results['model_nohist'].pi
        wfit_mean = wfit_mean + results['model_nohist'].w[infodict['indices'][i]]/ len (infodict['conditions'])
        w0fit_mean = w0fit_mean + results['model_w_hist'].w[infodict['indices'][i]]/ len (infodict['conditions'])     
        
        wfit_mean_nohist = wfit_mean_nohist + results['model_nohist'].w[infodict['indices'][i]]/ len (infodict['conditions'])
        wfit_mean_whist = wfit_mean_whist + results['model_w_hist'].w[infodict['indices'][i]]/ len (infodict['conditions'])        
        
        print "c:",c
        print "wfit:",wfit
        print "w0fit:",w0fit
        print "pfit:",pfit
        print "p0fit:",p0fit


        if not d.ground_truth is None:
            wgfit = d.ground_truth['w'][infodict['indices'][i]]
            pgfit = d.ground_truth['pi']
            gt = graphics.plot_pmf ( pgfit, wgfit, x, ax,
                    (np.array([255,240,240],'d')/255+infodict['colors'][i])/2. )
            pl.setp (gt, linestyle='--' )
 
        graphics.plot_pmf ( pfit, wfit_whist, x, ax, color = 'b')#, alpha=0.1 )#[.9,.9,.9], alpha=0.1 )
        graphics.plot_pmf ( p0fit, wfit,  x, ax, color = 'r') #infodict['colors'][i] )
    print "p0fit, wfit_mean:",p0fit, wfit_mean 
#    graphics.plot_pmf ( p0fit, wfit_mean,  x, ax, color = 'b',label ='mean')
#    graphics.plot_pmf ( pfit, w0fit_mean,  x, ax, color = 'r',label ='mean') 
#    graphics.label_axes (
#           # title="(A) psychometric function",
#            xlabel=r"transduced stimulus $u_\nu(s\tilde{z})$",
#            ylabel=r"probability for $r=1$",
#            nxticks=5,
#            ax=ax )

    print "wfit_mean:",wfit_mean


def nonlinearityplot ( d, results, infodict, ax ):
    """Plot with the nonlinearity"""
    xmin,xmax = infodict['xmin'],infodict['xmax']
    M = results['model_w_hist']
    for c in infodict['conditions']:
        d_ = d.getsummary ( c )

        graphics.plot_nonlinearity_summary ( d_, M.w, M.pi, ax, color=infodict['colors'][c], label=infodict['labels'][c] )
    graphics.plot_nonlinearity ( M.nu, np.mgrid[xmin:xmax:100j], ax, 'k' )
    if not d.ground_truth is None:
        nl = graphics.plot_nonlinearity ( d.ground_truth['nu'], np.mgrid[xmin:xmax:100j], ax, np.array([255,230,230],'d')/255 )
        pl.setp ( nl, linestyle='--' )
    graphics.label_axes ( title="(B) nonlinearity",
            xlabel=r"raw stimulus $s\tilde{z}$",
            ylabel=r"transduced stimulus $u_\nu(s\tilde{z})$",
            nxticks=5,
            ax=ax)

def permutationplot ( d, results, infodict, ax, noaic=False ):
    """permutation test"""
    l_obs = results['model_w_hist'].loglikelihood
    Caic  = results['model_nohist'].loglikelihood+len(results['model_w_hist'].w)-len(results['model_nohist'].w)
    out = [l_obs]
    print "l_obs=",l_obs
    print "Caic= ",Caic
    if not results['permutation_wh'] is None:
        hist, C95 = statistics.historytest ( results['permutation_wh'][:,0] )
        print "C95=  ",C95
        out.append ( C95 )
        out.append ( Caic )
        out.append ( np.mean ( results['permutation_wh'][:,0]<l_obs ) )
    else:
        hist = None
        C95  = None
        out.append ( None )
        out.append ( Caic )
        out.append ( None )
    if noaic:
        Caic=None
    graphics.montecarlo_test ( l_obs, hist, C95, Caic, ax, "likelihood" )
    graphics.label_axes ( title="(C) Permutation test",
            xlabel="log-likelihood",
            ax=ax )
    return out

def kernelplot ( d, results, infodict, ax1, ax2, legend='lower right' ):
    """Plot historykernels"""
    M = results['model_w_hist']
    bootstrap = results['bootstrap']

    # hr = d.gethistorykernel ( M.w[d.hf0:d.hf0+d.hlen], al )
    # hz = d.gethistorykernel ( M.w[d.hf0+d.hlen:],      al )
    C = statistics.Kernel_and_Slope_Collector ( d.h, d.hf0, range(1,d.hf0) )
    print d.hf0
    K = C(M)
    hr = K[:d.h.shape[0]]
    hz = K[d.h.shape[0]:-2]
    hz *= K[-2]
    hr *= K[-2]
    print "h_r[1]",hr[0]
    print "h_z[1]",hz[0]

    if not bootstrap is None:
        kernellen = (bootstrap.shape[1]-2)/2
        print kernellen
        al = bootstrap[:,-2]
        al.shape = (-1,1)
        bootstrap[:,:-2] *= al # Like that?
        print K[-2],pl.prctile(bootstrap[:,-2]),pl.mean(bootstrap[:,-2])

        hci = statistics.history_kernel_ci (
                bootstrap[:,kernellen:-2], bootstrap[:,:kernellen],
                hz, hr )
    else:
        hci = None
    print "hci:", hci
    kl = graphics.history_kernels ( hz, hr, hci, ax1,   "left/right", ground_truth = d.ground_truth )
    kl += graphics.history_kernels ( hz, hr, hci, ax2, "correct/incorrect", ground_truth = d.ground_truth )

    labely,labelh = same_y ( ax1, ax2 )

    graphics.label_axes (# title="(D) stimulus and response kernels",
            xlabel="lag",
            ylabel="Weight",
            legend=legend,
            ax=ax1 )
    graphics.label_axes ( title="(E) correct and incorrect kernels",
            xlabel="lag",
            ylabel="Weight",
            legend=legend,
            ax=ax2 )
    # pl.setp ( (ax1,ax2), ylim=(-6,6) )

    return kl

def slopeplot ( d, results, infodict, ax ):
    """slope results of the permutation test"""

    ax = graphics.prepare_axes ( ax, haveon=('bottom',) )

    h = np.histogram ( results['permutation_wh'][:,1] )
    graphics.montecarlo_test ( results['model_w_hist'].w[1], h, pl.prctile(results['permutation_wh'][:,1], 95), ax=ax, labeling='slope' )
    # ax.bar ( b[:-1], h, pl.diff(b), edgecolor=graphics.histogram_color, facecolor=graphics.histogram_color )
    # yrange = ax.get_ylim ()
    # ax.axvline ( results['model_w_hist'].w[1], ymin=yrange[0], ymax=yrange[1] )
    # ax.set_ylim ( yrange )
    # # ax.set_xlim ( .3,3 )
    # # ax.set_xticks ( [.5,1,1.5,2,2.5,3] )

    graphics.label_axes ( title="(F) slope effects",
            xlabel=r"slope",
            ax = ax )

def same_y ( *axes ):
    """Set all axes to the same ylimits"""
    lims = []
    for ax in axes:
        lims.append ( ax.get_ylim () )
    lims = pl.array(lims)
    ymin = abs(lims.min())
    ymax = abs(lims.max())
    yl = max(ymin,ymax)
    for ax in axes:
        ax.set_ylim ( -yl, yl )
    return -.8*yl,.1*yl
