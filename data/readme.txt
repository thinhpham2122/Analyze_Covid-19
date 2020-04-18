Two data sets are included:

- DataSet1.csv : Contains features for each county in GA and NY.
- DataSet2.csv : Contains a time series of COVID-19 Cases and Deaths in each county in GA and NY.

About data set features:

- DataSet1_variables.csv:
	- fips_county: 			Unique county identifier within that state.
	- county:				Name of the county.
	- fips_state: 			Unique state identifier.
	- state:				Name of the state.
	-state and county together (abcde: ab for state (GA = 11, NY=36), cde: county) 
	- age_group:			Age range for data on that row. NOTE: There are three columns per county to 
							allow for three age ranges (0-19, 20-49, 50 up).
	- population_total:		Total population for county/age.
	- population_male:		Total male population for county/age.
	- population_female:	Total female population for county/age.
	- poverty_estimate:		Total population in poverty for county (only for ALL age_group).
	- poverty_percentage:	Percent of population in poverty for county (only for ALL age_group).
	- income_median:		Median household income for county (only for ALL age_group).

- DataSet2_time_series.csv:
	- fips_county: 			Unique county identifier within that state.
	- county:				Name of the county.
	- fips_state: 			Unique state identifier.
	- state:				Name of the state.
	- type:					Type of time series, Cases or Deaths, due to COVID-19.
	- 1/22/2020-4/13/2020:	Number of Cases/Deaths in county over time.

How to use the data sets:

Make sure when data points are read in from both files, counties numbers match. 