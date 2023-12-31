#!/usr/bin/env python

# -*- coding: ascii -*-
__doc__ = """
This code was adapted from Fründ, Wichmann, Macke (2014): Quantifying the effect of inter-trial dependence on perceptual decisions. J Vis, 14(7): 9. 
See Copyright notice below

Copyright (C) 2014 Ingo Fründ

This code reproduces the analyses in the paper

    Fründ, Wichmann, Macke (2014): Quantifying the effect of inter-trial dependence on perceptual decisions. J Vis, 14(7): 9.


    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    If you use the Software for your own research, cite the paper.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os, sys

from intertrial_no_bootstrap_0_lags import util,column
import pylab as pl
import cPickle,os,sys
import pdb
from optparse import OptionParser

# Level of logmessages -- 10 allows for INFO but not for DEBUG
#                         20 will suppress INFO
import logging
logging.root.level = 10
logging.BASIC_FORMAT = '%(message)s'


############################## Parsing command line

usage = "analysis.py [options] <datafile>"

long_help = """
This is the workhorse script to generate the canonical figures.

Note that the data file should have a canonical structure with 5 columns

block    condition    stimulus    target    response

where each row contains the respective information for a single trial.

Typically, running this script will generate a folder for data backup called sim_backup. This folder contains all simulation results and can be used for more elaborate post-hoc plotting
"""

parser = OptionParser ( usage, epilog=long_help )

parser.add_option ( "-f", "--force",
        action="store_true",
        help="Force analysis even if backup files are found on disc" )
parser.add_option ( "-s", "--silent",
        action="store_true",
        help="Silent mode -- don't show any status messages" )
parser.add_option ( "-n", "--number-of-samples",
        default=0,
        type="int",
        help="number of samples for monte carlo procedures" )
parser.add_option ( "-r", "--hide-results",
        action="store_true",
        help="do not show the graphical results at the end" )
parser.add_option ( "-g", "--graphics-path",
        default="figures",
        help="path where the graphical output should be stored" )
parser.add_option ( "-t", "--detection",
        action="store_true",
        help="detection experiment: fit the threshold nonlinearity" )
parser.add_option ( "-e", "--header",
        action="store_true",
        help="Does the data file contain a header? If you choose this option, the header will be ignored!" )

opts,args = parser.parse_args ()

############################## Setting values in convenience module
if opts.silent:
    logging.root.level = 200

############################## Loading data
data,w0,plotinfo = util.load_data_file ( args[0], header=opts.header, detection=opts.detection )
print 'w0',w0
# Check for directories
if not os.path.exists ( "sim_backup" ):
    os.mkdir ( "sim_backup" )

if not os.path.exists ( opts.graphics_path ):
    os.mkdir ( opts.graphics_path )

############################## analyze data or read backup file
backup_file = os.path.join ( "sim_backup",os.path.basename(args[0])+"_0_lags.pcl" )
if os.path.exists ( backup_file ) and not opts.force:
    logging.info ( "Loading simulation results from %s" % (backup_file,) )
    results = cPickle.load ( open ( backup_file, 'r' ) )
#    logging.info ( "Read data from %d permutations and %d bootstrap repetitions" % \
#            (results['permutation_wh'].shape[0],results['bootstrap'].shape[0]) )
else:
    logging.info ( "Analyzing data" )
    results = util.analysis ( data, w0, opts.number_of_samples )
    print results['model_nohist'].pi
    logging.info ( "Storing results in %s" % (backup_file,) )
    cPickle.dump ( results, open ( backup_file, 'w' ) )

print results.keys()
print "nu=",results['model_w_hist'].nu

# plot
#util.plot ( data, results, plotinfo )

# store figure
#pl.savefig ( os.path.join ( opts.graphics_path, os.path.basename(args[0])+".pdf" ) )

#if not opts.hide_results:
#     pl.show()
