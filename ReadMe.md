```bash
virtualenv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
#hclustering.py
```bash
python3 hcluster.py <csvfile> <distance_method> <alpha>
<csvfile> = dataset path that we want to use 
<distance_method> -> 1 = singly link
                  -> 2 = complete link
                  -> 3 = average link
<alpha> the threshold value (this could be optional)

OUTPUT: 
<csvfile>_output.json --> gives the whole entire dendrogram (hiearchy)

if alpha != 0:
    <csvfile>_output_alpha.txt --> this gives information on how many clusters we will have after we cut off at that threshold off the dendrogram. Also list what each cluster contain and its height and etc.
    
    Console Output: The evaluation of all the clusters that were cut off from the alpha value
        -> Points, Center, Max Distance To Center, Min Distance To Center, Average Distance To Center

Note: If alpha is not provided, it would just take the threshold as 0 and the evaluation would just be a singular cluster (entire hiearchical) consisted of all points.
```