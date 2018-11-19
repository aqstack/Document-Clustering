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

❖ Data Processing: The input data consisted of 8580 text records(documents) provided in sparse format. The same data was read into a Compressed Sparse Row (CSR) Matrix. By doing so, the memory required for data storage can be minimized contributing to the efficiency of arithmetic operations, row slicing and matrix vector products.

❖ Tf-idf Term Weighting: In a large text corpus, some words will be very present but carry little meaningful information about the actual meaning of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms. TFIDF, stands for term frequency–inverse document frequency, which is a numerical statistic that is intended to reflect how important a word is to a document. It is proven that TFIDF scaling can enhance clustering. Therefore, implemented Tf-idf scaling to the sparse data to increase proportionality to the number of times a word appears in the document and offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.

❖ Dimensionality Reduction: The high number of features in the matrix contributed significantly to the curse of dimensionality. Therefore, linear dimensionality reduction was performed using truncated singular value decomposition (SVD). The method works with sparse matrices efficiently as it does not center the data before computing the SVD. In particular, truncated SVD works well on tf-idf matrices and in that context, it is known as latent semantic analysis (LSA).
