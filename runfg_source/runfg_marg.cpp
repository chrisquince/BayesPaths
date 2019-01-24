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
#define MAX_WIDTH 128

using namespace dai;
using namespace std;

int main( int argc, char *argv[] ) 
{
    if ( argc != 3 && argc != 4 ) {
        cout << "Usage: " << argv[0] << " <filename.fg> <outfile> [maxstates]" << endl << endl;
        cout << "Reads factor graph <filename.fg> and runs" << endl;
        cout << "Sum-product Belief Propagation [maxstates=0] or Sum-product JunctionTree [maxstates > 0] set." << endl;
        cout << "Sum product JunctionTree if a junction tree is found with" << endl;
        cout << "total number of states less than maxstates." << endl << endl;
        return 1;
    } 
    
    FactorGraph fg;
    fg.ReadFromFile(argv[1]);
    
    size_t maxstates = MAX_STATES;
    bool runJT = false;
        
    if(argc == 4){
        maxstates = fromString<size_t>(argv[3]);
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
    
    if (runJT == true){
        // Bound treewidth for junctiontree
        size_t tWidth;
        BigInt tState;
        try {
            std::pair< size_t, BigInt > p = boundTreewidth(fg, &eliminationCost_MinFill, maxstates );
            cout << "Running junction tree width " << p.first << " " << p.second << endl;
            tWidth = p.first;
            tState = p.second;
        } catch( Exception &e ) {
            if( e.getCode() == Exception::OUT_OF_MEMORY ) {
                runJT = false;
                cout << "Skipping junction tree (need more than " << maxstates << " states)." << endl;
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
        JTree jt;

        jt = JTree( fg, opts("updates",string("HUGIN"))("logdomain",true) );

        jt.init();
 
        jt.run();

        outfile.open(argv[2]);
        outfile << "Exact variable marginals:" << endl;
        
        for( size_t i = 0; i < fg.nrVars(); i++ ){ // iterate over all variables in fg
            outfile << jt.belief(fg.var(i)) << endl; // display the "belief" of jt for that variable
        }
        outfile.close();
    }
    else{
        BP bp(fg, opts("updates",string("SEQRND"))("logdomain",true));
                    // Initialize belief propagation algorithm
        bp.init();
        // Run belief propagation algorithm
        
        bp.run();
        
        outfile.open(argv[2]);
        outfile << "Approximate (loopy belief propagation) factor marginals:" << endl;
        for( size_t I = 0; I < fg.nrFactors(); I++ ){ // iterate over all factors in fg
            outfile << bp.belief(fg.factor(I).vars()) << endl; // display the belief of bp for the variables in that factor
        }
        outfile.close();
    }

    return 0;
}
