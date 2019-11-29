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
    if ( argc != 3 ) {
        cout << "Usage: " << argv[0] << " <filename.fg> <outfile>" << endl << endl;
        cout << "Reads factor graph <filename.fg> and runs" << endl;
        cout << "JunctionTree computes treewidth." << endl;
        return 1;
    } 

    size_t verb = 1;    
    ofstream outfile;    
    FactorGraph fg;
    Real   tol = 1e-9;
    size_t maxiter = 10000;
    
    fg.ReadFromFile(argv[1]);
    
    size_t maxstates = MAX_STATES;
    

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter",maxiter);  // Maximum number of iterations
    opts.set("tol",tol);          // Tolerance for convergence
    opts.set("verbose",verb);     // Verbosity (amount of output generated)
    size_t tWidth = -1;    
    BigInt tState;

    std::pair< size_t, BigInt > p = boundTreewidth(fg, &eliminationCost_MinFill, maxstates);

    tWidth = p.first;
    tState = p.second;
    
    outfile.open(argv[2]);
    
    outfile << "treewidth: " << tWidth << " states: " << tState << endl;
    
    outfile.close();
    
    return tWidth;
}

