import csv
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster, datasets
from matplotlib import pyplot as plt
import random
from copy import deepcopy

#read data
f = open('dataset.csv')
csv_f = csv.reader(f)
datal = []
for row in csv_f:
	rowint = [float(i) for i in row]	
	datal.append(rowint)
data = np.array(datal)

plt.figure(1)
#perform hierarchial clustering
#save the clustering results in variable clust
clust = linkage(data, method='ward', metric='euclidean')
#create a dendrogram from the result of the hierarchial clustering
#rotate the x axis labels, set the font size for x axis labels
dendrogram(clust, leaf_rotation = 90, leaf_font_size = 8) 

plt.title("Hierarchial Clustering Dendrogram")
plt.xlabel("sample index")
plt.ylabel("distance")

print data.shape[0]
#declare cluster centers (list of 2) assume (k=3)
#initialize the cluster centers randomly
cen = []
cen.append( data[random.randint(0,data.shape[0])] )
cen.append( data[random.randint(0,data.shape[0])] )
cen.append( data[random.randint(0,data.shape[0])] )
cent = np.array(cen)
print cent

plt.figure(2)

#introduce labels for each feature
names = ['Feature A', 'Feature B']

#draw the initial guess
plt.scatter(data[:,0], data[:,1], s=8, c='g')
plt.scatter(cent[:,0], cent[:,1], marker='*', s=150, c='black')
plt.title("Clusters with Initial Centroids")
plt.xlabel(names[0])
plt.ylabel(names[1])

#define a function for Eudlicldean Distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
	
#create a list of 0's to store old value of centroids
cent_old = np.zeros(cent.shape) 
#create a list of 0's to use as index of cluster labels 0,1,2
cluster = np.zeros(len(data)) 
#distance between new centroids and old centroids 
distdiff = dist(cent, cent_old, None) 

#initial iteration 
iteration = 0
#plot clusters
plotite = [5, 10, 100]

# repeat until cluster assignment stop changing
while distdiff != 0 and iteration < 100:
	# assign each point to its closest cluster
	for m in range(len(data)):
		#compute distances to each cluster centroid
		distances = dist(data[m], cent)
		#find the closest distance and store the index
		clust = np.argmin(distances)
		cluster[m] = clust
	# store old centroids
	cent_old = deepcopy(cent)
	# Finding the new centroids by taking the average value
	for i in range(3):
		points = []
		for j in range(len(data)):
			if cluster[j] == i:
				points.append(data[j])
		cent[i] = np.mean(points, axis=0)
	distdiff = dist(cent, cent_old, None)
	iteration += 1
	print 'The ', iteration, 'th iterations'
	#print cent
	if iteration in plotite:
		colors = ['r', 'g', 'b']
		fig, ax = plt.subplots()
		for i in range(3):
			points = []
			for j in range(len(data)):
				if cluster[j] == i:
					points.append(data[j])        
			points = np.array(points)
        		ax.scatter(points[:, 0], points[:, 1], s=8, c=colors[i])
		ax.scatter(cent[:, 0], cent[:, 1], marker='*', s=150, c='black')
		plt.title("Iterations of Cluster: %i" %iteration)
		plt.xlabel(names[0])
		plt.ylabel(names[1])

print cent

if iteration not in plotite:
	colors = ['r', 'g', 'b']
	fig, ax = plt.subplots()
	for i in range(3):
		points = []
		for j in range(len(data)):
			if cluster[j] == i:
				points.append(data[j])        

		points = np.array(points)
        	ax.scatter(points[:, 0], points[:, 1], s=8, c=colors[i])
	ax.scatter(cent[:, 0], cent[:, 1], marker='*', s=150, c='black')
	plt.title('Final Clusters with Centroids')
	plt.xlabel(names[0])
	plt.ylabel(names[1])

plt.show()


