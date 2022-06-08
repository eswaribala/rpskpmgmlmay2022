# Hierarchical Clustering

# Importing the dataset
dataset = read.csv('G:/Local disk/MachineLearning/Data_Processing/Hierarchical-Clustering/Hierarchical_Clustering/Mall_Customers.csv')
dataset = dataset[4:5]
#dataset=head(dataset)
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(d = dist(dataset, method = 'euclidean'), method = 'complete')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting Hierarchical Clustering to the dataset
hc = hclust(d = dist(dataset, method = 'euclidean'), method = 'complete')
y_hc = cutree(hc, 5)
# 
# # Visualising the clusters
# library(cluster)
# clusplot(dataset,
#          y_hc,
#          lines = 0,
#          shade = TRUE,
#          color = TRUE,
#          labels= 2,
#          plotchar = FALSE,
#          span = TRUE,
#          main = paste('Clusters of customers'),
#          xlab = 'Annual Income',
#          ylab = 'Spending Score')