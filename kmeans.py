import sys
import pandas as pd
import numpy as np
from scipy.spatial import distance



def kmeanspp(data, k):
    n = 0
    csdata = data.copy(deep = True)
    centroids = []
    center = tuple(csdata.mean())

    points = [tuple(x) for x in csdata.to_numpy()]

    cdst = [distance.euclidean(p, center) for p in points]
    csdata['cdst'] = cdst
    n += 1
    m0idx = csdata['cdst'].idxmax()
    m0 = tuple(csdata.loc[m0idx][0:len(csdata.columns)-n])
    csdata = csdata.drop([m0idx]).reset_index(drop = True)
    del points[m0idx]
    centroids.append(m0)

    if n < k:
        m0dst = [distance.euclidean(p, m0) for p in points]
        # print(m0dst)
        csdata[str('m'+str(n-1)+'dst')] = m0dst
        n += 1
        m1idx = csdata[str('m'+str(n-2)+'dst')].idxmax()
        m1 = tuple(csdata.loc[m1idx][0:len(csdata.columns)-n])
        csdata = csdata.drop([m1idx]).reset_index(drop = True)
        # print(m1)
        print(csdata)
        del points[m1idx]
        centroids.append(m1)

    while n < k:
        sums = []
        for i in range(0,len(csdata)):
            mi = tuple(csdata.iloc[i][0:len(csdata.columns)-2])
            # print(mi)
            midstsum = sum([distance.euclidean(mj, mi) for mj in centroids])
            sums.append(midstsum)
        maxsum = max(sums)
        maxsumidx = sums.index(maxsum)

        print(maxsum)
        print(maxsumidx)
        n += 1
        mi = tuple(csdata.iloc[maxsumidx][0:len(csdata.columns)-2])
        csdata = csdata.drop([maxsumidx]).reset_index(drop = True)
        print(mi)
        del points[maxsumidx]
        centroids.append(mi)

    print(data)
    print(csdata)
    print(centroids)

def main():
    n = len(sys.argv)
    filepath = None
    k = 0


    if n != 3:
        print("args error")
    else:
        filepath = sys.argv[1]
        k = int(sys.argv[2])

    data = pd.read_csv(filepath)

    data = data.rename(columns={x:y for x,y in zip(data.columns,range(0,len(data.columns)))})
    print(data)
    kmeanspp(data,k)




if __name__ == "__main__":
    main()