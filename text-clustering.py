
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics


# In[ ]:


docs = []


# In[ ]:


data = list()
ilist = list()
jlist = list()
with open('train.dat', 'r') as fr:
    for lineno, line in enumerate(fr):
        line = line.strip()
        parts = line.split()
        for i in range(int(len(parts)/2)):
            # 1237 1 1390 1 1391 5 ...
            jlist.append(int(parts[i*2]))
            data.append(int(parts[i*2+1]))
            ilist.append(lineno)

cm = csr_matrix((data, (ilist, jlist)), dtype = np.float)
cm.toarray().shape


# In[ ]:


X_initial = np.array(cm)
X_initial


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=False)
#transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
tfidf = transformer.fit_transform(cm)


# In[ ]:


tfidf.shape


# In[ ]:


cmArr = tfidf.toarray()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import TruncatedSVD\nsvd = TruncatedSVD(n_components=150)\n#svd.fit(cm.toarray())\n#print(svd.explained_variance_ratio_.sum())')


# In[ ]:


from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
#normalizer = Normalizer(copy=False)
#lsa = make_pipeline(svd, normalizer)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X = svd.fit_transform(cmArr)')


# In[ ]:


X.shape


# In[ ]:


explained_variance = svd.explained_variance_ratio_.sum()
print(explained_variance)


# In[ ]:


#Create a set of np.random K centroids for each feature of the given dataset
def randCent(dataSet,K):
    n=np.shape(dataSet)[1]
    centroids=np.mat(np.zeros((K,n)))
    for j in range(n):
        minValue=min(dataSet[:,j])
        maxValue=max(dataSet[:,j])
        rangeValues=float(maxValue-minValue)
        #Make sure centroids stay within the range of data
        centroids[:,j]=minValue+rangeValues*np.random.rand(K,1)
    # np.matrix
    return centroids


# In[ ]:


# euclidean distance measure
def distanceMeasure(vecOne, vecTwo):
    return np.sqrt(np.sum(np.power(vecOne-vecTwo,2)))


# In[ ]:


# K means clustering method
def kMeans(dataSet,K,distMethods=distanceMeasure,createCent=randCent):
    m=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((m,2)))
    # np.matrix
    centroids=createCent(dataSet,K)
    clusterChanged=True
    
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=np.inf; minIndex=-2
            for j in range(K):
                distJI=distMethods(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0] != minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        #update all the centroids by taking the np.mean value of relevant data
        for cent in range(K):
            ptsInClust=dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:]=np.mean(ptsInClust,axis=0)
    # (np.matrix, np.matrix)
    return centroids,clusterAssment


# In[ ]:


#bisecting K-means method
def bisectingKMeans(dataSet,K,numIterations):
    m,n=dataSet.shape
    clusterInformation = np.mat(np.zeros((m,2)))
    centroidList=[]
    minSSE = np.inf
    
    #At the first place, regard the whole dataset as a cluster and find the best clusters
    for i in range(numIterations):
        # (np.matrix, np.matrix)
        centroid,clusterAssment=kMeans(dataSet, 2)
        SSE=np.sum(clusterAssment,axis=0)[0,1]
        if SSE<minSSE:
            minSSE=SSE
            tempCentroid=centroid
            tempCluster=clusterAssment
    centroidList.append(tempCentroid[0].tolist()[0])
    centroidList.append(tempCentroid[1].tolist()[0])
    clusterInformation=tempCluster
    minSSE=np.inf 
    
    while len(centroidList)<K:
        maxIndex=-2
        maxSSE=-1
        #Choose the cluster with Maximum SSE to split
        for j in range(len(centroidList)):
            SSE=np.sum(clusterInformation[np.nonzero(clusterInformation[:,0]==j)[0]])
            if SSE>maxSSE:
                maxIndex=j
                maxSSE=SSE
                
        #Choose the clusters with minimum total SSE to store into the centroidList
        for k in range(numIterations):
            pointsInCluster=dataSet[np.nonzero(clusterInformation[:,0]==maxIndex)[0]]
            # (np.matrix, np.matrix)
            centroid,clusterAssment=kMeans(pointsInCluster, 2)
            SSE=np.sum(clusterAssment[:,1],axis=0)
            if SSE<minSSE:
                minSSE=SSE
                tempCentroid=centroid.copy()
                tempCluster=clusterAssment.copy()
        #Update the index
        tempCluster[np.nonzero(tempCluster[:,0]==1)[0],0]=len(centroidList)
        tempCluster[np.nonzero(tempCluster[:,0]==0)[0],0]=maxIndex
        
        #update the information of index and SSE
        clusterInformation[np.nonzero(clusterInformation[:,0]==maxIndex)[0],:]=tempCluster
        #update the centrolist
        centroidList[maxIndex]=tempCentroid[0].tolist()[0]
        centroidList.append(tempCentroid[1].tolist()[0])
    # (List[List[float]], np.matrix)
    return centroidList,clusterInformation


# In[ ]:


c_list, c_info = bisectingKMeans(X,7,20)


# In[ ]:


#type(c_list), len(c_list), type(c_list[0]), len(c_list[0]), c_list[0][0]
c_list[0][0]


# In[ ]:


with open('clustering_v7.0.txt', 'w') as fw:
    for v in c_info[:, 0]+1:
        print(int(v.A[0][0]), file=fw)


# In[ ]:


# Using silhouette score as the metric
valueK = list()
silhouetteScore = list()
labels = list()
for k in range(3, 22, 2):
    c_list, c_info = bisectingKMeans(X,k,10)
    for v in c_info[:, 0]+1:
        labels.append((int(v.A[0][0])))

    valueK.append(k)
    silhouetteScore.append(metrics.silhouette_score(X, labels))


# In[ ]:


# Plotting silhoutte score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(valuek, silhouetteScore)
plt.xlabel('Number of Clusters k')
plt.ylabel('Silhouette Score')

