# TEXT CLUSTERING USING BISECTING K-MEANS ALGORITHM

Clustering is the task of grouping a set of similar objects than to those in other groups. While it is unsupervised, its application results in taking data that can categorize groups of related data, and forming clusters that represent data that have similarities.
Document clustering (or text clustering) is the application of cluster analysis to textual documents. It has applications in automatic document organization, topic modeling, keyword extraction, and information retrieval.

The objectives of this assignment were as described below:
• Implement the Bisecting K-Means algorithm.
• Deal with text data (news records) in document-term sparse matrix format.
• Design a proximity function for text data.
• Think about the Curse of Dimensionality.
• Think about best metrics for evaluating clustering solutions.

### Approach:

❖ **Data Processing:** The input data consisted of 8580 text records(documents) provided in sparse format. The same data was read into a Compressed Sparse Row (CSR) Matrix. By doing so, the memory required for data storage can be minimized contributing to the efficiency of arithmetic operations, row slicing and matrix vector products.

❖ **Tf-idf Term Weighting:** In a large text corpus, some words will be very present but carry little meaningful information about the actual meaning of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms. TFIDF, stands for term frequency–inverse document frequency, which is a numerical statistic that is intended to reflect how important a word is to a document. It is proven that TFIDF scaling can enhance clustering. Therefore, implemented Tf-idf scaling to the sparse data to increase proportionality to the number of times a word appears in the document and offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.

❖ **Dimensionality Reduction:** The high number of features in the matrix contributed significantly to the curse of dimensionality. Therefore, linear dimensionality reduction was performed using truncated singular value decomposition (SVD). The method works with sparse matrices efficiently as it does not center the data before computing the SVD. In particular, truncated SVD works well on tf-idf matrices and in that context, it is known as latent semantic analysis (LSA).

❖ **K-Means Algorithm:** The Κ-means clustering algorithm uses iterative refinement to produce a result. The algorithm inputs are the number of clusters Κ and the data set. The data set is a collection of features for each data point. The algorithm starts with initial estimates for the Κ centroids which then iterates between two steps, Data Assignment and Centroid update. In data assignment step, each data point is assigned to its nearest centroid, based on the squared Euclidean distance measure. In every iteration, the centroids are recomputed by taking the mean of all the data points assigned to that centroid’s cluster. The algorithm iterated until a stopping condition is met, that is, no data points change clusters, the sum of the distances is minimized, or a hard-coded number of iterations is reached. The algorithm is guaranteed to converge to a result. The result may not be necessarily the best possible outcome which means assessing more than one run of the algorithm with randomized initial centroids may generate a better outcome.

❖ **Bisecting K-Means Algorithm:** Like any algorithm, there are disadvantages to the k-means as well. Firstly, there are nested loops, which means it is quadratic complex. Moreover, the distance metric used within the nested loop is done unconditionally, arising tractability concerns. Lastly, the algorithm has the tendency of collapsing on a local minima instead of a global minima due to poor initialization. A potential remedy for this, is to do a post-processing of the created clustered model, where the Cluster to be split is picked on which Cluster has the highest sum of squared errors (SSE). This will hopefully yield for an SSE between the two resultant child clusters that is less than the SSE for their parent cluster. This forms the basis for the algorithm known as the Bisecting K-Means algorithm(pseudocode) which is shown as below. 
    • Pick a cluster to split. 
    • Find 2 sub-clusters using the basic k-Means algorithm. 
    • Repeat step 2, the bisecting step, for ITER times and take the split that produces lowest SSE. 
    • Repeat steps 1, 2 and 3 until the desired number of clusters is reached.
    
❖ **Silhouette Score:** Silhouette score metric is used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and therefore, provides a way to assess number of clusters visually. Same was plotted on the y-axis with values for k on the x-axis increasing from 3 to 21 in steps of 2 for the given dataset. The Silhouette Coefficient is defined for each sample and is composed of two scores:
    • The mean distance between a sample and all other points in the same cluster.
    • The mean distance between a sample and all other points in the nearest cluster.

### Conclusion:

Text clustering was performed successfully implementing K-Means and Bisecting K-Means algorithm. Same was evaluated using Silhouette Metric. The data was dealt using document-term sparse matrix and curse of dimensionality was cured using singular value decomposition.


