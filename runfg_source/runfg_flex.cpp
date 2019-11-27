/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <map>
#include <dai/alldai.h>  // Include main libDAI header file
#include <dai/jtree.h>
#include <dai/bp.h>
#include <dai/decmap.h>

#define MAX_STATES 1048576
#define MAX_WIDTH 16

using namespace dai;
using namespace std;

int main( int argc, char *argv[] ) 
{
    if ( argc != 4 && argc != 5 ) {
        cout << "Usage: " << argv[0] << " <filename.fg> <outfile> [maxprod] [maxstates]" << endl << endl;
        cout << "Reads factor graph <filename.fg> and runs" << endl;
        cout << "Max-product[maxprod=1] or sum-product[maxprod=0] Belief Propagation [maxstates=0] or Sum-product JunctionTree [maxstates > 0] set." << endl;
        cout << "Sum product JunctionTree if a junction tree is found with" << endl;
        cout << "total number of states less than maxstates." << endl << endl;
        return 1;
    } 
    ofstream outfile;    
    FactorGraph fg;
    fg.ReadFromFile(argv[1]);
    
    size_t maxstates = MAX_STATES;
    bool runJT = false;
    
    
    size_t sumprod = fromString<size_t>(argv[3]);
    if(argc == 5){
        maxstates = fromString<size_t>(argv[4]);
        if (maxstates <= 0){
            maxstates = MAX_STATES;
        }
        runJT = true;
    }
    
    // Set some constants
    size_t maxiter = 10000;
    Real   tol = 1e-9;
    size_t verb = 1;

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter",maxiter);  // Maximum number of iterations
    opts.set("tol",tol);          // Tolerance for convergence
    opts.set("verbose",verb);     // Verbosity (amount of output generated)
    size_t tWidth = -1;    
    if (runJT == true){
        // Bound treewidth for junctiontree

        BigInt tState;
        try {
            std::pair< size_t, BigInt > p = boundTreewidth(fg, &eliminationCost_MinFill, maxstates );
            //cout << "Running junction tree width " << p.first << " " << p.second << endl;
            tWidth = p.first;
            tState = p.second;
        } catch( Exception &e ) {
            if( e.getCode() == Exception::OUT_OF_MEMORY ) {
                runJT = false;
              //  cout << "Skipping junction tree (need more than " << maxstates << " states)." << endl;
            }
            else{
                throw;
            }
        }

        if (tWidth > MAX_WIDTH || tState > maxstates){
            cout << "Running BP maxwidth exceeded or maxstates exceeded" << endl;
            
            runJT = false;
        }
    } 
    
    if(runJT == true) {
        if (sumprod == 0){
            JTree jt;

            jt = JTree( fg, opts("updates",string("HUGIN"))("logdomain",true) );

            jt.init();
 
            jt.run();

            outfile.open(argv[2]);
            outfile << "Exact variable marginals twidth:" << tWidth << endl;
        
            for( size_t i = 0; i < fg.nrVars(); i++ ){ // iterate over all variables in fg
                outfile << jt.belief(fg.var(i)) << endl; // display the "belief" of jt for that variable
            }
            outfile.close();
        }
        else{
            JTree jt, jtmap;
            vector<size_t> jtmapstate;

            // Construct another JTree (junction tree) object that is used to calculate
                // the joint configuration of variables that has maximum probability (MAP state)
            jtmap = JTree( fg, opts("updates",string("HUGIN"))("inference",string("MAXPROD")) );
                // Initialize junction tree algorithm
            jtmap.init();
            // Run junction tree algorithm
            
            jtmap.run();
            // Calculate joint state of all variables that has maximum probability
            jtmapstate = jtmap.findMaximum();
            
            outfile.open(argv[2]);
            outfile << "Exact MAP state (log score = " << fg.logScore(jtmapstate) << "):twidth:" << tWidth << endl;
            for( size_t i = 0; i < jtmapstate.size(); i++ )
                outfile << fg.var(i) << "," << jtmapstate[i] << endl;
            outfile.close();
        }
    }
    else{
        if (sumprod == 0){
            BP bp(fg, opts("updates",string("SEQRND"))("logdomain",true));
                    // Initialize belief propagation algorithm
            bp.init();
            // Run belief propagation algorithm
        
            bp.run();
        
            outfile.open(argv[2]);
            outfile << "Approximate (loopy belief propagation) factor marginals twidth:" << tWidth << endl;
            for( size_t i = 0; i < bp.nrVars(); i++ ){ // iterate over all variables in fg
                outfile << bp.belief(bp.var(i)) << endl; // display the "belief" of jt for that variable
            }
            outfile.close();
        }
        else{
            BP mp(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
            // Initialize max-product algorithm
            mp.init();
            // Run max-product algorithm
            mp.run();
            // Calculate joint state of all variables that has maximum probability
            // based on the max-product result
            vector<size_t> mpstate = mp.findMaximum();

            // Report max-product MAP joint state]
            outfile.open(argv[2]);
            outfile << "Approximate (max-product) MAP state (log score = " << fg.logScore( mpstate ) << "):twidth:" << tWidth << endl;
            for( size_t i = 0; i < mpstate.size(); i++ )
                outfile << fg.var(i) << "," << mpstate[i] << endl;
            
            outfile.close();
        }
    }

    return 0;
}

