import sys
import math
import pandas as pd
import numpy as np
from scipy.spatial import distance # shouldn't need to pip this as long as you have numpy, will replace with my own distance function
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
        # print(csdata)
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
 
        # print(maxsum)
        # print(maxsumidx)
        n += 1
        mi = tuple(csdata.iloc[maxsumidx][0:len(csdata.columns)-2]) # next centroid as tuple
        csdata = csdata.drop([maxsumidx]).reset_index(drop = True) # removed from df
        # print(mi)
        del points[maxsumidx] # removed from list of points
        centroids.append(mi) # added to list of centroids

    # print(data) # original data frame, will be used for actual clustering
    # print(csdata) # copy used for centroid selection, centroids should be removed
    # print(centroids) # list of centroids selected
    
    return centroids

def hybrid_kpp(data,k):
    n = 0
    csdata = data.copy(
        deep=True)  # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    centroids = []
    center = tuple(csdata.mean())  # calculating center of entire dataset

    points = [tuple(x) for x in csdata.to_numpy()]  # converting points to tuples for distance calculations

    # using imported distance calculator, will have to write our own for final submission
    m0 = random.sample(points, 1)[0]
    m0indx = points.index(m0)
    # points.remove(m0)
    del points[m0indx]
    csdata = csdata.drop([m0indx]).reset_index(drop=True)  # removed from df
    centroids.append(m0)
    n += 1
    if n < k:  # 2nd centroid selection
        m0dst = [distance.euclidean(p, m0) for p in points]  # distance from first selected centriod for all other points
        # print(m0dst)
        csdata[str('m' + str(n - 1) + 'dst')] = m0dst  # added as column, not really necessary but nice for debugging
        n += 1
        m1idx = csdata[str('m' + str(n - 2) + 'dst')].idxmax()  # index of 2nd centroid
        m1 = tuple(csdata.loc[m1idx][0:len(csdata.columns) - n + 1])  # 2nd centroid as tuple
        csdata = csdata.drop([m1idx]).reset_index(drop=True)  # removed from df
        # print(m1)
        # print(csdata)
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

        # print(maxsum)
        # print(maxsumidx)
        n += 1
        mi = tuple(csdata.iloc[maxsumidx][0:len(csdata.columns) - 1])  # next centroid as tuple
        csdata = csdata.drop([maxsumidx]).reset_index(drop=True)  # removed from df
        # print(mi)
        del points[maxsumidx]  # removed from list of points
        centroids.append(mi)  # added to list of centroids

    # print(data)  # original data frame, will be used for actual clustering
    # print(csdata)  # copy used for centroid selection, centroids should be removed
    # print(centroids)  # list of centroids selected

    return centroids


def main():
    n = len(sys.argv)
    filepath = None
    k = 0 # number of clusters desired

    if n != 3:
        print("args error")
        return
    else:
        filepath = sys.argv[1]
        k = int(sys.argv[2])

    data = pd.read_csv(filepath, index_col=False) # the points

    r = open(filepath)
    restAttrs = r.readline().split(',')
    for ix, a in enumerate(restAttrs):
        if int(a) != 1:
            data = data.drop(data.columns[ix], axis=1)
    print(data)



    data = data.rename(columns={x:y for x,y in zip(data.columns,range(0,len(data.columns)))}) # rename columns with dimension value 

    dimensions = len(data.columns)



    # centroids = hybrid_kpp(data,k)
    centroids = kmeanspp(data,k)
    dfcent = pd.DataFrame(centroids)
    
    
    
    points = [tuple(x) for x in data.to_numpy()]

    means = []
    loopcounter = 0

    rehist = dfcent.copy(deep=True)
    prevSmallest = None
    prevSSE = None
    
    while 1 == 1: # loop for assigning points to centroids, will include termination points as breaks

        # p2cmin = distance.cdist(points,centroids, metric='euclidean').min(axis=1)
        distp2c = distance.cdist(points,centroids, metric='euclidean')
        # print(p2cmin)
        dfp2c = pd.DataFrame(distp2c)
        # print(dfp2c)

        smallest = dfp2c.idxmin(axis=1)
        # print(smallest)
        dfp2c['centroid'] = smallest
        data['centroid'] = smallest
        data.sort_values(by=['centroid'],inplace = True)

        # print(data)

        ccount = data.groupby(['centroid']).count()
        csum = data.groupby(['centroid']).sum()
        cdiv = csum/ccount

        # print(ccount)
        # print(csum)
        # print(cdiv)

        # for ci in range(0, len(centroids)):
        rehist = pd.concat([rehist,cdiv], axis=1)
        # print(rehist)
        cdiff = abs(dfcent - cdiv)
        # print(cdiff)



        data['centroidVal'] = data.apply(lambda r: centroidValue(r,centroids), axis = 1)
        cv = pd.DataFrame(data['centroidVal'].to_list(), index=data.index)

        # data = pd.concat([data,cv], axis=1)
        data = data.drop(['centroidVal'],axis = 1)
        # print(cv)

        currSSE = (data - cv).pow(2)
        # print(SSE)
        currSSE = currSSE.sum(axis = 1)
        currSSE['centroid'] = data['centroid']
        currSSE = currSSE.groupby(['centroid']).sum()
        # print(SSE)
        # print(data)
        




        dfcent = cdiv
        # print(dfcent)
        centroids = [tuple(c) for c in dfcent.to_numpy()]


        if (loopcounter > 0) and (smallest.eq(prevSmallest.values).mean() > .99) : #if minimum reassignment of points between clusters
            loopcounter += 1
            # print(smallest)
            # print(prevSmallest)
            # print(smallest.eq(prevSmallest.values).mean())
            break
            

        if ((cdiff == 0).all(axis = 0)).all() == True: #if centroids recalculation produces minimal change
            loopcounter += 1
            # print(smallest.eq(prevSmallest.values).mean())
            break

        if (loopcounter > 0): #if minimum reassignment of points between clusters
                SSEdiff = prevSSE - currSSE
                # SSEsum = 
                # print(SSE)
                # print(prevSSE)
                # print(SSEdiff)
                if 1 == 2:
                    loopcounter += 1

                    break



        

        prevSSE = currSSE
        prevSmallest = smallest
        loopcounter += 1

    # print(data)
    # print(dfp2c)
    for c in centroids:
        print("Cluster " + str(centroids.index(c))+":")
        print("Center: " + str(c))
        print("Max Dist. to Center: " + str(dfp2c.groupby(['centroid']).max()[centroids.index(c)][centroids.index(c)]) )
        print("Min Dist. to Center: " + str(dfp2c.groupby(['centroid']).min()[centroids.index(c)][centroids.index(c)]) )
        print("Avg Dist. to Center: " + str(dfp2c.groupby(['centroid']).mean()[centroids.index(c)][centroids.index(c)]) )
        print(str(dfp2c.groupby(['centroid']).count()[centroids.index(c)][centroids.index(c)])+" Points:")
        centPoints = data.loc[data['centroid'] == centroids.index(c)]
        centPoints = centPoints.drop(['centroid'], axis = 1).sort_index()
        print( centPoints.to_string() )

    print("End")




def centroidValue(row, centroids):
    for c in centroids:
        if row['centroid'] == centroids.index(c):
            return c




if __name__ == "__main__":
    main()