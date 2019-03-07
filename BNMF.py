import sys, os
import argparse
import numpy as np

from BNMTF_ARD.code.models.bnmf_vb import bnmf_vb
from BNMTF_ARD.code.cross_validation.mask import compute_folds_attempts


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("cov_file", help="unitig cov file")

    args = parser.parse_args()

    import ipdb; ipdb.set_trace()
    
    covMap = {}
    with open(args.cov_file) as f:
        for line in f:
            line = line.rstrip()
            
            tokens = line.split(',')    
            
            unitig = tokens[0]
            
            covs = [float(x) for x in tokens[1:]]

            covMap[unitig] = covs
    
    unitigs = sorted(covMap.keys())
    V = len(unitigs)
    S = len(covMap[unitigs[0]])
    
    X = np.zeros((V,S))
    
    i = 0
    for unitig in unitigs:
        X[i,:] = np.asarray(covMap[unitig])
        i += 1
    
    no_folds = 10
    values_K = [1,2,3,4,6,8,10,15,20,30,40]

    output_folder = "./BNMTF_ARD/"
    output_file = output_folder+'nmf_vb_ard.txt'
    
    metrics = ['MSE', 'R^2', 'Rp']

    ''' Model settings. '''
    iterations = 200

    init_UV = 'random'
    ARD = True

    lambdaU, lambdaV = 0.1, 0.1
    alphatau, betatau = 1., 1.
    alpha0, beta0 = 1., 1.
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaU':lambdaU, 'lambdaV':lambdaV }


    ''' Load in data. '''
    R = X
    M = np.ones(X.shape)
    I, J = M.shape


    ''' Generate matrices M - one list of M's for each value of K. '''
    M_attempts = 1000
    all_Ms_training_and_test = [
        compute_folds_attempts(I=I,J=J,no_folds=no_folds,attempts=M_attempts,M=M)
        for K in values_K
    ]
    
    print "Dummy"
    
if __name__ == "__main__":
    main(sys.argv[1:])
