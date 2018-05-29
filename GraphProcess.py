import argparse
import sys

class GraphProcess():
    """ Class for processing assembly graph"""    
    
    def __init__(self):
    
    

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")

    parser.add_argument("cog_file", help="unitig cog assignments")
    
    parser.add_argument("core_cogs", help="list of core cogs")
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main(sys.argv[1:])