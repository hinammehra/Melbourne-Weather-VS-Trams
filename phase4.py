# Final Investigation
# This file has code about clustering, correlation, and route level scatter plots

import math
from sklearn import cluster
from scipy.stats.stats import pearsonr
from phase2 import *

# draws route level scatter plots
def route_plots(merged_df, tram_metric_name, weather_metric_name, route_name):
	merged_df.plot(kind='scatter', x=tram_metric_name, y=weather_metric_name)	
	plt.grid()
	plt.savefig(route_name)
	plt.close()

# Applies tram and weather threshold
# Returns a tuple of number of rows affected, mean of weather column, and max of weather column
def filter(tram_metric, weather_metric, tram_route, tram_threshold, weather_threshold, weather_name):
	df = merge(tram_metric, weather_metric)
	rows = df[weather_name][(df[tram_route]<=tram_threshold) & (df[weather_name]>weather_threshold)]
	return (len(rows),round(rows.mean(),1), round(rows.max(),1))

# Filters tram performance attributes with given threshold
# returns a dataframe containing route level rows and weather rows with threshold applied
def tram_filter(tram_metric,tram_route,tram_threshold,weather_metric,weather_name,tram_metric_name):
	df = merge(tram_metric, weather_metric)
	df = df[df[tram_route]<=tram_threshold]
	df.rename(columns={tram_route: tram_metric_name}, inplace=True)
	return df[[tram_metric_name,weather_name]]

tram_routes_name = list(route_punctuality.columns.values)[2:]

# creates a dictionary
# Group names from 0..k are keys with an array of tram routes as its values
# centroids is a key, with a dictionary of centroids divided by group as its values
def results(groups, k):
	analyse_dict = {}
	labels = groups[0]
	centroids = groups[1]
	route_cluster = []
	for j in range(len(tram_routes_name)):
		route_cluster.append([tram_routes_name[j],labels[j]])
	centroid_dict = {}
	for i in range(k):
		centroid_dict[i] = np.round(centroids[i],1)
	analyse_dict['centroids'] = centroid_dict

	for x in range(len(route_cluster)):
		if route_cluster[x][1] not in analyse_dict.keys():
			analyse_dict[route_cluster[x][1]] = []
			analyse_dict[route_cluster[x][1]].append(route_cluster[x][0])
		else:
			analyse_dict[route_cluster[x][1]].append(route_cluster[x][0])
	return analyse_dict

# performs k-means clustering on the given numpy array
def clustering(route_vals, k):
	data = np.array(route_vals)
	kmeans = cluster.KMeans(n_clusters=k)
	kmeans.fit(data)
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	return results([labels, centroids], k)

# accepts number of clusters 
# creates an array of filtered tupless
def analyse(k):
	punctuality_th = 77.0
	delivery_th = 98.0
	rain_th = 0.0
	temp_th = 25.0
	punctuality_rainfall_a = []
	punctuality_temp_a = []
	delivery_rainfall_a = []
	delivery_temp_a = []
	for i in tram_routes_name:
		punctuality_rainfall_a.append(filter(route_punctuality, rainfall, i, punctuality_th, rain_th, 'Rainfall'))
		punctuality_temp_a.append(filter(route_punctuality, temperature, i, punctuality_th, temp_th, 'Maximum temperature'))
		delivery_rainfall_a.append(filter(route_delivery, rainfall, i, delivery_th, rain_th, 'Rainfall')) 
		delivery_temp_a.append(filter(route_delivery, temperature, i, delivery_th, temp_th, 'Maximum temperature')) 

# calculates average of an array when nan values exists
def avg(array):
	sum_ = 0
	num = len(array)
	for i in array:
		if not math.isnan(i):
			sum_ += i
		else:
			num -= 1
	return sum_/num

# calculates pearson's correlation
def correlation():
	punctuality_rainfall_corr = []
	punctuality_temp_corr = []
	delivery_rainfall_corr = []
	delivery_temp_corr = []
	for i in tram_routes_name:
		punctuality_rainfall_corr.append(pearsonr(route_punctuality[i],rainfall['Rainfall'])[0]) # no relation
		punctuality_temp_corr.append(pearsonr(route_punctuality[i],temperature['Maximum temperature'])[0]) #96 -0.342654369193
		delivery_temp_corr.append(pearsonr(route_delivery[i],temperature['Maximum temperature'])[0]) #96 -0.290976756897
		delivery_rainfall_corr.append(pearsonr(route_delivery[i],rainfall['Rainfall'])[0])
	punctuality_rainfall_max_corr = min(punctuality_rainfall_corr)
	punctuality_rainfall_mean_corr = avg(punctuality_rainfall_corr)
	punctuality_temp_max_corr = min(punctuality_temp_corr)
	punctuality_temp_mean_corr = avg(punctuality_temp_corr)
	delivery_temp_max_corr = min(delivery_temp_corr)
	delivery_temp_mean_corr = avg(delivery_temp_corr)
	delivery_rainfall_max_corr = min(delivery_rainfall_corr)
	delivery_rainfall_mean_corr = avg(delivery_rainfall_corr)

def route_scatter_plots():
	punctuality_th = 77.0
	delivery_th = 98.0
	for i in tram_routes_name:
		#poor_punctuality = tram_filter(route_punctuality,i,punctuality_th,rainfall,'Rainfall', 'Punctuality')
		#route_plots(poor_punctuality,'Punctuality','Rainfall',i)
		#poor_punctuality = tram_filter(route_punctuality,i,punctuality_th,temperature,'Maximum temperature', 'Punctuality')
		#route_plots(poor_punctuality,'Punctuality','Maximum temperature',i)
		#poor_delivery = tram_filter(route_delivery,i,delivery_th,rainfall,'Rainfall','Delivery')
		#route_plots(poor_delivery,'Delivery','Rainfall',i)
		poor_delivery = tram_filter(route_delivery,i,delivery_th,temperature,'Maximum temperature','Delivery')
		route_plots(poor_delivery,'Delivery','Maximum temperature',i)



route_scatter_plots();
#correlation()
#analyse(6)