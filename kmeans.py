import sys
import pandas as pd
import numpy as np
from scipy.spatial import distance # shouldn't need to pip this as long as you have numpy
import random



def kmeanspp(data, k):
    n = 0
    csdata = data.copy(deep = True) # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    centroids = []
    center = tuple(csdata.mean()) # calculating center of entire dataset

    points = [tuple(x) for x in csdata.to_numpy()] # converting points to tuples for distance calculations

    # using imported distance calculator, will have to write our own for final submission
    cdst = [distance.euclidean(p, center) for p in points] # distance of all points from center stored in an array
    csdata['cdst'] = cdst # added as column
    n += 1 # n is number of centroids selected, lazy counter
    m0idx = csdata['cdst'].idxmax() # index of furthest distance from center, first centroid
    m0 = tuple(csdata.loc[m0idx][0:len(csdata.columns)-n]) # first centroid as tuple
    csdata = csdata.drop([m0idx]).reset_index(drop = True) # remove centroid from data table for centroid selection
    del points[m0idx] # remove the centroid from selectable list of points
    centroids.append(m0) # added to list

    if n < k: # 2nd centroid selection
        m0dst = [distance.euclidean(p, m0) for p in points] # distance from first selected centriod for all other points
        # print(m0dst)
        csdata[str('m'+str(n-1)+'dst')] = m0dst # added as column, not really necessary but nice for debugging
        n += 1
        m1idx = csdata[str('m'+str(n-2)+'dst')].idxmax() # index of 2nd centroid
        m1 = tuple(csdata.loc[m1idx][0:len(csdata.columns)-n]) # 2nd centroid as tuple
        csdata = csdata.drop([m1idx]).reset_index(drop = True) # removed from df
        # print(m1)
        print(csdata)
        del points[m1idx] # removed from points list
        centroids.append(m1) # added to centroids list

    while n < k: # all centroids after 2nd up to kth
        sums = []
        for i in range(0,len(csdata)): # finds sum of distances from the centroids for each point, point furthest from all centroids using sum is next centroid
            mi = tuple(csdata.iloc[i][0:len(csdata.columns)-2])
            # print(mi)
            midstsum = sum([distance.euclidean(mj, mi) for mj in centroids]) # sum of distances from centroids for a point
            sums.append(midstsum) 
        maxsum = max(sums) # greastest sum of distances 
        maxsumidx = sums.index(maxsum) # index of point with greatest sum / next centroid
 
        print(maxsum)
        print(maxsumidx)
        n += 1
        mi = tuple(csdata.iloc[maxsumidx][0:len(csdata.columns)-2]) # next centroid as tuple
        csdata = csdata.drop([maxsumidx]).reset_index(drop = True) # removed from df
        print(mi)
        del points[maxsumidx] # removed from list of points
        centroids.append(mi) # added to list of centroids

    print(data) # original data frame, will be used for actual clustering
    print(csdata) # copy used for centroid selection, centroids should be removed
    print(centroids) # list of centroids selected

def hybrid_kpp(data,k):
    n = 0
    csdata = data.copy(
        deep=True)  # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    centroids = []
    center = tuple(csdata.mean())  # calculating center of entire dataset

    points = [tuple(x) for x in csdata.to_numpy()]  # converting points to tuples for distance calculations

    # using imported distance calculator, will have to write our own for final submission
    m0 = random.sample(points, 1)[0]
    # import pdb; pdb.set_trace()
    m0indx = points.index(m0)
    # points.remove(m0)
    del points[m0indx]
    csdata = csdata.drop([m0indx]).reset_index(drop=True)  # removed from df
    centroids.append(m0)
    n += 1
    # import pdb; pdb.set_trace()
    if n < k:  # 2nd centroid selection
        # import pdb; pdb.set_trace()
        m0dst = [distance.euclidean(p, m0) for p in
                 points]  # distance from first selected centriod for all other points
        # print(m0dst)
        csdata[str('m' + str(n - 1) + 'dst')] = m0dst  # added as column, not really necessary but nice for debugging
        n += 1
        m1idx = csdata[str('m' + str(n - 2) + 'dst')].idxmax()  # index of 2nd centroid
        m1 = tuple(csdata.loc[m1idx][0:len(csdata.columns) - n + 1])  # 2nd centroid as tuple
        csdata = csdata.drop([m1idx]).reset_index(drop=True)  # removed from df
        # print(m1)
        print(csdata)
        del points[m1idx]  # removed from points list
        centroids.append(m1)  # added to centroids list

    while n < k:  # all centroids after 2nd up to kth
        sums = []
        for i in range(0, len(
                csdata)):  # finds sum of distances from the centroids for each point, point furthest from all centroids using sum is next centroid
            mi = tuple(csdata.iloc[i][0:len(csdata.columns) - 1])
            # print(mi)
            midstsum = sum(
                [distance.euclidean(mj, mi) for mj in centroids])  # sum of distances from centroids for a point
            sums.append(midstsum)
        maxsum = max(sums)  # greastest sum of distances
        maxsumidx = sums.index(maxsum)  # index of point with greatest sum / next centroid

        print(maxsum)
        print(maxsumidx)
        n += 1
        mi = tuple(csdata.iloc[maxsumidx][0:len(csdata.columns) - 1])  # next centroid as tuple
        csdata = csdata.drop([maxsumidx]).reset_index(drop=True)  # removed from df
        print(mi)
        del points[maxsumidx]  # removed from list of points
        centroids.append(mi)  # added to list of centroids

    print(data)  # original data frame, will be used for actual clustering
    print(csdata)  # copy used for centroid selection, centroids should be removed
    print(centroids)  # list of centroids selected

def main():
    n = len(sys.argv)
    filepath = None
    k = 0 # number of clusters desired


    if n != 3:
        print("args error")
    else:
        filepath = sys.argv[1]
        k = int(sys.argv[2])

    data = pd.read_csv(filepath) # the points

    data = data.rename(columns={x:y for x,y in zip(data.columns,range(0,len(data.columns)))}) # rename columns with dimension value 
    print(data)
    import pdb; pdb.set_trace()
    hybrid_kpp(data,k)
    # kmeanspp(data,k)




if __name__ == "__main__":
    main()