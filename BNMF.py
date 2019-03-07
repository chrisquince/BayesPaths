import sys, os
import argparse



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

            covMap[initig] = covs
    
    unitigs = sorted(covMap.keys())
    V = len(unitigs)
    S = len(covMap[unitigs[0]])
    
    X = np.zeros((V,S))
    
    i = 0
    for unitig in unitigs:
        X[i,:] = np.asarray(covMap[unitig])
        i += 1
        
    print "Dummy"
    
if __name__ == "__main__":
    main(sys.argv[1:])