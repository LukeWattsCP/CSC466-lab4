import sys
import numpy as np
import pandas as pd
from pandas.io.formats.format import return_docstring



def coreCluster(x,data,core,distArray,clusterCount,epsilon,numpoints):
    if pd.isnull(core.loc[x.name]['cluster']):
        if core['cluster'].isnull().all():
            clusterCount = 0
        else:
            clusterCount = core['cluster'].max() + 1
        x.at['cluster'] = clusterCount
        data.at[x.name, 'cluster'] = clusterCount
        core.at[x.name, 'cluster'] = clusterCount

        x.at['visited'] = 1
        data.at[x.name,'visited'] = 1
        core.at[x.name,'visited'] = 1

        densityConnected(x,data,core,distArray,clusterCount,epsilon,numpoints)
    # print(x['cluster'])
    return

def densityConnected(x,data,core,distArray,clusterCount,epsilon,numpoints): # maybe there's a way to vectorize this? no idea tbh
    for idx in x['neighbors']:
        data.at[idx,'cluster'] = clusterCount
        # x['cluster']
        if idx in core.index:
            p = core.loc[idx]
            core.at[idx,'cluster'] = clusterCount
            p.at['cluster'] = clusterCount

            # print(p)
            if p['visited'] == 0:
                data.at[idx,'visited'] = 1
                core.at[idx,'visited'] = 1
                p.at['visited'] = 1
                densityConnected(p,data,core,distArray,clusterCount,epsilon,numpoints)

            

    return
def main():
    
    if len(sys.argv) == 4:
        filepath = sys.argv[1]
        epsilon = float(sys.argv[2])
        numpoints = int(sys.argv[3])
    else:
        print("Invalid arguments, please see README")
        return

    data = pd.read_csv(filepath, index_col=False) # the points
    r = open(filepath)
    restAttrs = r.readline().split(',')
    for ix, a in enumerate(restAttrs):
        if int(a) != 1:
            data = data.drop(data.columns[ix], axis=1)
    data = data.rename(columns={x:y for x,y in zip(data.columns,range(0,len(data.columns)))}) # rename columns with dimension value 

    # print(data)

    distArray = data.apply(lambda r: euclideanDF(r, data), axis = 1)
    # distArray.values[[np.arange(distArray.shape[0])]*2] = -1
    # s = pd.Series(data=[1,2,3],index=['a','b','c'])
    np.fill_diagonal(distArray.values, -1)
    print(distArray)
    classification = distArray[(distArray[:] <= epsilon) & (distArray[:] != -1)].count()
    classification = classification.apply(lambda r: 'c' if r >= numpoints else 'n')
    classification = pd.DataFrame({'type':classification.values})
    # print(classification)
    visited = pd.DataFrame({'visited':([0] * len(data))})
    # print(visited)


    data = pd.concat([data,classification,visited], axis = 1)
    # print(data)

    core = (data.loc[data['type'] == 'c']).copy(deep = True)
    # .drop(columns = ['type'])
    core['neighbors'] = np.empty((len(core), 0)).tolist()

    # print(distArray)

    neighbors = distArray.apply(lambda r: findNeighbors(r,epsilon))
    # print(neighbors)
    core['neighbors'] = neighbors
    core['cluster'] = np.nan
    data['cluster'] = np.nan

    clusterCount = -1
    core.apply(lambda x: coreCluster(x,data,core,distArray,clusterCount,epsilon,numpoints), axis = 1)
    print(str(data.to_string()))
    print(str(core.to_string()))


    print('END')


def euclideanDF(df1, df2):
    r = df1 - df2
    r = r.pow(2).sum(axis = 1).apply(np.sqrt)
    return r

def findNeighbors(df1, e):
    r = df1.index[(df1 <= e) & (df1 != -1)].tolist()

    return r
    


if __name__ == "__main__":
    main()