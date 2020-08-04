# Initial Investigation
# This file has code about pre-processing, and network level scatter plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Removes columns 'Quality', 'Station number', 'Product Code', 'Year', 'Days of Accumulation'
def remove_weather_columns(df):
	df = df.drop(df.columns[[0, 1, 2, 6, 7]], axis=1)
	return df

# fills missing data with 0
def fill_weather_data(df):
	df.fillna(0, inplace=True)
	return None

# replaces tram's Month column with weather's Month column
# Transform's data from percentages to whole numbers
def clean_tram_columns(df, weather):
	df.drop(df.columns[[0]], axis=1, inplace=True)
	df['Month'] = weather['Month']
	df.ix[:,2:] *= 100
	df = df.round(1)
	return df

# fills missing values in tram data with mean of the entire network
def fill_tram_data(df, type):
	df.rename(columns = lambda x: str(x), inplace=True)
	df['12'].fillna(value=network[type], inplace=True)
	df['11'].fillna(value=network[type], inplace=True)
	return None

# Weather Data cleaned/filled
rainfall = pd.read_csv('rainfall.csv')
rainfall = remove_weather_columns(rainfall)
fill_weather_data(rainfall)
rainfall.rename(columns={'Rainfall amount (millimetres)': 'Rainfall'}, inplace=True)


temperature = pd.read_csv('temperature.csv')
temperature = remove_weather_columns(temperature)
temperature.rename(columns={'Maximum temperature (Degree C)': 'Maximum temperature'}, inplace=True)

# Tram Data cleaned/filled
network = pd.read_excel('Trams.xlsx')
network = clean_tram_columns(network, temperature)
network.rename(columns={'% services on-time over length of route': 'Punctuality', '% timetable delivered': 'Delivery'}, inplace=True)


route_punctuality = pd.read_excel('Trams.xlsx', 1)
route_delivery = pd.read_excel('Trams.xlsx', 2)

route_punctuality = clean_tram_columns(route_punctuality, temperature)
route_delivery = clean_tram_columns(route_delivery, temperature)

fill_tram_data(route_punctuality, "Punctuality")
fill_tram_data(route_delivery, "Delivery")


#Initial Investigation
poor_punctuality = network[(network["Punctuality"] <= 77.0)]
poor_delivery = network[(network["Delivery"] <= 98.0)]

high_rainfall = rainfall[(rainfall["Rainfall"] >= 1.0)]
high_temperature = temperature[(temperature["Maximum temperature"] >= 25.0)]

# merges two dataframes 
def merge(tram_metric, weather_metric):
	df = pd.merge(tram_metric, weather_metric, on=['Month', 'Day'])
	return df

# draws a scatter plot
def scatter_plot(tram_metric, weather_metric, tram_metric_name, weather_metric_name, hex):
	df = merge(tram_metric, weather_metric)
	df.plot(kind='scatter', x=tram_metric_name, y=weather_metric_name, color=hex)
	plt.grid()
	plot_name = tram_metric_name + " - " + weather_metric_name + hex
	plt.savefig(plot_name)

# Phase 2 Visualtions

# scatter_plot(poor_punctuality, rainfall, 'Punctuality', 'Rainfall','#180283')
# scatter_plot(poor_punctuality, temperature, 'Punctuality', 'Maximum temperature','#180283')
# scatter_plot(poor_delivery, rainfall, 'Delivery', 'Rainfall','#180283')
# scatter_plot(poor_delivery, temperature, 'Delivery', 'Maximum temperature','#180283')
# scatter_plot(network, high_rainfall, 'Punctuality', 'Rainfall', '#7C2851')
# scatter_plot(network, high_temperature, 'Punctuality', 'Maximum temperature', '#7C2851')
# scatter_plot(network, high_temperature, 'Delivery', 'Maximum temperature', '#7C2851')
# scatter_plot(network, high_rainfall, 'Delivery', 'Rainfall', '#7C2851')



