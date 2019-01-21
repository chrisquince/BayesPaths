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


int main( int argc, char *argv[] ) {
#if defined(DAI_WITH_BP) && defined(DAI_WITH_JTREE)
    if ( argc != 2 && argc != 3 ) {
        cout << "Usage: " << argv[0] << " <filename.fg> [maxstates]" << endl << endl;
        cout << "Reads factor graph <filename.fg> and runs" << endl;
        cout << "Sum-product Belief Propagation [maxstates=0] or Sum-product JunctionTree [maxstates > 0] set." << endl;
        cout << "Sum product JunctionTree if a junction tree is found with" << endl;
        cout << "total number of states less than maxstates." << endl << endl;
        return 1;
    } else {

        // Read FactorGraph from the file specified by the first command line argument
        FactorGraph fg;
        fg.ReadFromFile(argv[1]);

        for( size_t I = 0; I < fg.nrFactors(); I++ ){
            Factor factorI = fg.factor( I );
//            cout << "Normalise " << I << endl;
            factorI.normalize();
        }

        size_t maxstates = MAX_STATES;
        bool runJT = false;
        bool do_jt = true;
        if( argc == 3 ){
            maxstates = fromString<size_t>( argv[2] );
            if (maxstates <= 0){
                maxstates = MAX_STATES;
            }
            runJT = true;
        }
        if (runJT == true){
            do_jt = true;
        }
        else{
            do_jt = false;
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
                    do_jt = false;
                    cout << "Skipping junction tree (need more than " << maxstates << " states)." << endl;
                }
                else
                    throw;
            }

            if (tWidth > MAX_WIDTH || tState > maxstates){
                cout << "Running BP maxwidth exceeded" << endl;
                do_jt = false;
            } 
    }
    
            
    if( do_jt ) {
                JTree jt;
                // Construct another JTree (junction tree) object that is used to calculate
                // the joint configuration of variables that has maximum probability (MAP state)
                jt = JTree( fg, opts("updates",string("HUGIN"))("logdomain",true) );
                //jtmap = JTree( fg, opts("updates",string("HUGIN"))("inference",string("MAXPROD")) );
                // Initialize junction tree algorithm
                jt.init();
                // Run junction tree algorithm
                jt.run();
                // Calculate joint state of all variables that has maximum probability
                //jtmapstate = jtmap.findMaximum();
            

            cout << "Exact variable marginals:" << endl;
            for( size_t i = 0; i < fg.nrVars(); i++ ){ // iterate over all variables in fg
                cout << jt.belief(fg.var(i)) << endl; // display the "belief" of jt for that variable
            }
        }
        else{
            // Construct a BP (belief propagation) object from the FactorGraph fg
            // using the parameters specified by opts and two additional properties,
            // specifying the type of updates the BP algorithm should perform and
            // whether they should be done in the real or in the logdomain
            //
            // Note that inference is set to MAXPROD, which means that the object
            // will perform the max-product algorithm instead of the sum-product algorithm
            BP bp(fg, opts("updates",string("SEQRND"))("logdomain",true));
            // Initialize belief propagation algorithm
            bp.init();
            // Run belief propagation algorithm
            bp.run();

            cout << "Approximate (loopy belief propagation) factor marginals:" << endl;
            for( size_t I = 0; I < fg.nrFactors(); I++ ){ // iterate over all factors in fg
                cout << bp.belief(fg.factor(I).vars()) << endl; // display the belief of bp for the variables in that factor
            }
    }
    return 0;
}
#else
    cout << "libDAI was configured without BP or JunctionTree (this can be changed in include/dai/dai_config.h)." << endl;
#endif
}
