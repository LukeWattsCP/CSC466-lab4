import sys
import numpy as np
import pandas as pd



def main():
    
    if len(sys.argv) == 4:
        filepath = sys.argv[1]
        epsilon = sys.argv[2]
        numpoints = sys.argv[3]
    else:
        print("Invalid arguments, please see README")
        return

    data = pd.read_csv(filepath, index_col=False) # the points
    print(data)

    visited = pd.DataFrame({'visited':([0] * len(data))})
    print(visited)



if __name__ == "__main__":
    main()