import sys
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import random

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def kmeanspp(data, k):
    n = 0
    csdata = data.copy(deep = True) # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    centroids = []
    
    cent2 =  csdata.mean()
    # print(cent2)
    # print(csdata)
    cdst = csdata.subtract(cent2, axis = 1)
    cdst = cdst.pow(2).sum(axis = 1).apply(np.sqrt)


    points = [tuple(x) for x in csdata.to_numpy()] # converting points to list of tuples 

    # # using imported distance calculator, will have to write our own for final submission
    # cdst = [distance.euclidean(p, center) for p in points] # distance of all points from center stored in an array
    cdst = cdst.tolist()


    csdata['cdst'] = cdst # added as column
    n += 1 # n is number of centroids selected, lazy counter
    m0idx = csdata['cdst'].idxmax() # index of furthest distance from center, first centroid
    m0 = csdata.loc[m0idx][0:len(csdata.columns)-1]
    csdata = csdata.drop([m0idx]).reset_index(drop = True) # remove centroid from data table for centroid selection
    del points[m0idx] # remove the centroid from selectable list of points
    centroids.append(tuple(m0)) # added to list
    csdata = csdata.drop('cdst', axis = 1).reset_index(drop=True)


    if n < k: # 2nd centroid selection

        # print(m0)
        m0dst = csdata.subtract(m0, axis = 1)
        m0dst = m0dst.pow(2).sum(axis = 1).apply(np.sqrt)
        m0dst = m0dst.tolist()


        # m0dst = [distance.euclidean(p, m0) for p in points] # distance from first selected centriod for all other points
        # print(m0dst)


        csdata[str('m0dst')] = m0dst # added as column, not really necessary but nice for debugging
        n += 1
        m1idx = csdata[str('m0dst')].idxmax() # index of 2nd centroid
        m1 = csdata.loc[m1idx][0:len(csdata.columns)-1] # 2nd centroid as tuple
        csdata = csdata.drop([m1idx]).reset_index(drop = True) # removed from df
        # print(m1)
        # print(csdata)
        del points[m1idx] # removed from points list
        centroids.append(tuple(m1)) # added to centroids list
        csdata = csdata.drop('m0dst', axis = 1).reset_index(drop=True)

        # print(csdata)
        # print(centroids)


    while n < k: # all centroids after 2nd up to kth
        dfcentroid = pd.DataFrame(centroids)
        sums = csdata.apply(lambda x : euclideanSumDF(dfcentroid, x ),axis = 1)
        maxsumidx = sums.idxmax()
 
        n += 1
        mi = tuple(csdata.iloc[maxsumidx][0:len(csdata.columns)]) # next centroid as tuple
        csdata = csdata.drop([maxsumidx]).reset_index(drop = True) # removed from df
        del points[maxsumidx] # removed from list of points
        centroids.append(mi) # added to list of centroids
    
    return centroids

def kmeanshybrid(data,k):
    n = 0
    csdata = data.copy(deep=True)  # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    centroids = []

    points = [tuple(x) for x in csdata.to_numpy()]  # converting points to tuples 

    m0 = random.sample(points, 1)[0]
    m0indx = points.index(m0)
    del points[m0indx]
    csdata = csdata.drop([m0indx]).reset_index(drop=True)  # removed from df
    centroids.append(m0)
    n += 1


    if n < k: # 2nd centroid selection

        # print(m0)
        m0dst = csdata.subtract(m0, axis = 1)
        m0dst = m0dst.pow(2).sum(axis = 1).apply(np.sqrt)
        m0dst = m0dst.tolist()


        # m0dst = [distance.euclidean(p, m0) for p in points] # distance from first selected centriod for all other points
        # print(m0dst)

        csdata[str('m0dst')] = m0dst # added as column, not really necessary but nice for debugging
        n += 1
        m1idx = csdata[str('m0dst')].idxmax() # index of 2nd centroid
        m1 = csdata.loc[m1idx][0:len(csdata.columns)-1] # 2nd centroid as tuple
        csdata = csdata.drop([m1idx]).reset_index(drop = True) # removed from df
        # print(m1)
        # print(csdata)
        del points[m1idx] # removed from points list
        centroids.append(tuple(m1)) # added to centroids list
        csdata = csdata.drop('m0dst', axis = 1).reset_index(drop=True)

        # print(csdata)
        # print(centroids)


    while n < k: # all centroids after 2nd up to kth
        dfcentroid = pd.DataFrame(centroids)
        sums = csdata.apply(lambda x : euclideanSumDF(dfcentroid, x ),axis = 1)
        maxsumidx = sums.idxmax()
 
        n += 1
        mi = tuple(csdata.iloc[maxsumidx][0:len(csdata.columns)]) # next centroid as tuple
        csdata = csdata.drop([maxsumidx]).reset_index(drop = True) # removed from df
        del points[maxsumidx] # removed from list of points
        centroids.append(mi) # added to list of centroids
    
    return centroids


def kmeansvanilla(data, k):
    n = 0
    csdata = data.copy(deep=True)  # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    centroids = []

    while n < k:
        points = [tuple(x) for x in csdata.to_numpy()]  # converting points to tuples 
        m0 = random.sample(points, 1)[0]
        m0indx = points.index(m0)
        del points[m0indx]
        csdata = csdata.drop([m0indx]).reset_index(drop=True)  # removed from df
        centroids.append(m0)
        n += 1
    # print(centroids)

    return centroids


def main():
    n = len(sys.argv)
    filepath = None
    k = 0 # number of clusters desired

    if n != 4:
        print("Invalid argument(s), please see README")
        return
    else:
        filepath = sys.argv[1]
        k = int(sys.argv[2])
        mode = int(sys.argv[3])

    data = pd.read_csv(filepath, index_col=False) # the points

    r = open(filepath)
    restAttrs = r.readline().split(',')
    for ix, a in enumerate(restAttrs):
        if int(a) != 1:
            data = data.drop(data.columns[ix], axis=1)
    # print(data)



    data = data.rename(columns={x:y for x,y in zip(data.columns,range(0,len(data.columns)))}) # rename columns with dimension value 

    dimensions = len(data.columns)



    if mode == 0:
        centroids = kmeansvanilla(data, k)
    elif mode == 1:
        centroids = kmeanshybrid(data,k)
    elif mode == 2:
        centroids = kmeanspp(data,k)
    else:
        print("Invalid argument(s), please see README")
        return





    dfcent = pd.DataFrame(centroids)
    # points = [tuple(x) for x in data.to_numpy()]

    means = []
    loopcounter = 0

    rehist = dfcent.copy(deep=True)
    prevSmallest = None
    prevSSE = None
    
    while 1 == 1: # loop for assigning points to centroids, will include termination points as breaks



        dfp2c = data.apply(lambda x : euclideanDF(dfcent, x ),axis = 1)


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
    clusterList = []
    print("Clustering finished, number of centroid recalculations performed: " + str(loopcounter))
    for c in centroids:
        print("Cluster " + str(centroids.index(c))+":")
        print("Center: " + str(c))
        print("Max Dist. to Center: " + str(dfp2c.groupby(['centroid']).max()[centroids.index(c)][centroids.index(c)]) )
        print("Min Dist. to Center: " + str(dfp2c.groupby(['centroid']).min()[centroids.index(c)][centroids.index(c)]) )
        print("Avg Dist. to Center: " + str(dfp2c.groupby(['centroid']).mean()[centroids.index(c)][centroids.index(c)]) )
        print(str(dfp2c.groupby(['centroid']).count()[centroids.index(c)][centroids.index(c)])+" Points:")
        centPoints = data.loc[data['centroid'] == centroids.index(c)]
        centPoints = centPoints.drop(['centroid'], axis = 1).sort_index()
        clusterList.append(centPoints)
        print( centPoints.to_string() )


    # data.columns = ['x','y', 'z', 'centroid']
<<<<<<< HEAD
    # print(data)
    #
    # # 2D
=======
    # data.columns = ['x','y', 'centroid']

    # print(data)

    # 2D
>>>>>>> law
    # groups = data.groupby('centroid')
    # fig, ax = plt.subplots()
    # for name, group in groups:
    #     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    # ax.legend()

    # plt.show()

    # # 3D
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(data['x'], data['y'], data['z'], c=data['centroid'])
    # plt.show()
    # data = data.sort_index()
    # setosa = data[0:50]
    # setosa = setosa.drop('centroid', axis =1)
    # setosa.columns = ['a','b','c','d']
    # testsetosa = clusterList[1]
    # testsetosa.columns = ['a','b','c','d']

    # sj = pd.merge(setosa, testsetosa, how='inner', on = ['a','b','c','d'])

    # print(len(sj.drop_duplicates(subset=['a','b','c', 'd'], keep=False)))
    # print(len(set(setosa).intersection(clusterList[1])))
    # print(len(pd.merge(setosa, clusterList[1], how='outer'))/len(clusterList[1]))


    # veris = data[50:100]
    # veris = veris.drop('centroid', axis =1)

    # print(veris)
    # print(clusterList[0])
    # print(veris.eq(clusterList[0]).mean())

    # virgin = data[100:]
    # virgin = virgin.drop('centroid', axis =1)
    # print(virgin)
    # print(clusterList[2])
    # print(virgin.eq(clusterList[2]).mean())



    # print("End")




def centroidValue(row, centroids):
    for c in centroids:
        if row['centroid'] == centroids.index(c):
            return c

def euclideanSumDF(df1, df2):
    r = df1 - df2
    r = r.pow(2).sum(axis = 1).apply(np.sqrt)
    r = r.sum()
    return r

def euclideanDF(df1, df2):
    r = df1 - df2
    r = r.pow(2).sum(axis = 1).apply(np.sqrt)
    return r
    



if __name__ == "__main__":
    main()