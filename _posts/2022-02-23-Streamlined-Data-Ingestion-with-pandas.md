---
layout: post
title: Streamlined Data Ingestion with pandas
date: 2022-02-22
tags: datacamp, python, dataingestion
categories: datacamp python dataingestion
comments: true
---

Data ingestion practice

## Lesson 1 : *Importing Data from Flat Files*
* Pandas
* Modifying flat file imports

* Import a subset of columns
```python
# Create list of columns to use
cols = ["zipcode", "agi_stub", "mars1", "MARS2", "NUMDEP"]

# Create dataframe from csv using only selected columns
data = pd.read_csv("vt_tax_data_2016.csv", usecols=cols)

# View counts of dependents and tax returns by income level
print(data.groupby("agi_stub").sum())

```

* Import a file in chunks
```python
# Create dataframe of next 500 rows with labeled columns
vt_data_next500 = pd.read_csv("vt_tax_data_2016.csv", 
                              nrows=500,
                              skiprows=500,
                              header=None,
                              names=list(vt_data_first500))

# View the Vermont dataframes to confirm they're different
print(vt_data_first500.head())
print(vt_data_next500.head())


```

* Specify Data Types
```python
# Create dict specifying data types for agi_stub and zipcode
data_types = {"agi_stub": "category",
              "zipcode": str}

# Load csv using dtype to set correct data types
data = pd.read_csv("vt_tax_data_2016.csv", dtype=data_types)

# Print data types of resulting frame
print(data.dtypes.head())

```
* Customizing Missing data values
```python
# Create dict specifying that 0s in zipcode are NA values
null_values = {"zipcode": 0}

# Load csv using na_values keyword argument
data = pd.read_csv("vt_tax_data_2016.csv", 
                   na_values=null_values)

# View rows with NA ZIP codes
print(data[data.zipcode.isna()])


```
* Line with Errors
```python
try:
  # Set warn_bad_lines to issue warnings about bad records
  data = pd.read_csv("vt_tax_data_2016_corrupt.csv", 
                     error_bad_lines=False, 
                     warn_bad_lines=True)
  
  # View first 5 records
  print(data.head())
  
except pd.io.common.CParserError:
    print("Your data contained rows that could not be parsed.")

```


## Lesson 2 : Importing Data From Excel Files
* read spreadsheet
```python
# Load pandas as pd
import pandas as pd

# Read spreadsheet and assign it to survey_responses
survey_responses = pd.read_excel('fcc_survey.xlsx')

# View the head of the dataframe
print(survey_responses.head())

```
* Load a portion of a spreadsheet
```python
# Create string of lettered columns to load
col_string = "AD, AW:BA"

# Load data with skiprows and usecols set
survey_responses = pd.read_excel("fcc_survey_headers.xlsx", 
                                 skiprows=2, 
                                 usecols=col_string)

# View the names of the columns selected
print(survey_responses.columns)
```

* Getting data from multiple worksheets
```python
# Create df from second worksheet by referencing its position
responses_2017 = pd.read_excel("fcc_survey.xlsx",
                               sheet_name=1)

# Graph where people would like to get a developer job
job_prefs = responses_2017.groupby("JobPref").JobPref.count()
job_prefs.plot.barh()
plt.show()

```

```python

# Create df from second worksheet by referencing its name
responses_2017 = pd.read_excel("fcc_survey.xlsx",
                               sheet_name="2017")

# Graph where people would like to get a developer job
job_prefs = responses_2017.groupby("JobPref").JobPref.count()
job_prefs.plot.barh()
plt.show()


```

```python
# Load both the 2016 and 2017 sheets by name
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name=['2016', '2017'])

# View the data type of all_survey_data
print(type(all_survey_data))


---
# Load all sheets in the Excel file
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name=[0, '2017'])

# View the sheet names in all_survey_data
print(all_survey_data.keys())


---
# Load all sheets in the Excel file
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name=None)

# View the sheet names in all_survey_data
print(all_survey_data.keys())

```

* Work with multiple spreadsheets
> Workbooks meant primarily for human readers, not machines, may store data about a single subject across multiple sheets. For example, a file may have a different sheet of transactions for each region or year in which a business operated.

> pandas has been imported as pd. All sheets have been read into the ordered dictionary responses, where sheet names are keys and dataframes are values, so you can get dataframes with the values() method.

```python
# Create an empty dataframe
all_responses = pd.DataFrame()

# Set up for loop to iterate through values in responses
for df in responses.values():
  # Print the number of rows being added
  print("Adding {} rows".format(df.shape[0]))
  # Append df to all_responses, assign result
  all_responses = all_responses.append(df)

# Graph employment statuses in sample
counts = all_responses.groupby("EmploymentStatus").EmploymentStatus.count()
counts.plot.barh()
plt.show()


```


### *Modifying imports: true/false data*
```python
# Load the data
survey_data = pd.read_excel("fcc_survey_subset.xlsx")

# Count NA values in each column
print(survey_data.isna().sum())

# Set dtype to load appropriate column(s) as Boolean data
survey_data = pd.read_excel("fcc_survey_subset.xlsx",
                            dtype={"HasDebt": bool})

# View financial burdens by Boolean group
print(survey_data.groupby("HasDebt").sum())

       HasFinancialDependents  HasHomeMortgage HasStudentDebt
HasDebt                                                         
False               112.0              0.0             0.0
True                205.0            151.0           281.0
```

* *Set custom true/false values*
```python

# Load file with Yes as a True value and No as a False value
survey_subset = pd.read_excel("fcc_survey_yn_data.xlsx",
                              dtype={"HasDebt": bool,
                              "AttendedBootCampYesNo": bool},
                              true_values=["Yes"],
                              false_values=["No"])

# View the data
print(survey_subset.head())

```


### *Modifying imports: parsing dates*
* Date and Time Data
* *Parse simple dates*
```python
# Load file, with Part1StartTime parsed as datetime data
survey_data = pd.read_excel("fcc_survey.xlsx",
                            parse_dates=["Part1StartTime"])

# Print first few values of Part1StartTime
print(survey_data.Part1StartTime.head())


```

* Get datetimes from multiple columns
> Sometimes, datetime data is split across columns. A dataset might have a date and a time column, or a date may be split into year, month, and day columns.
> A column in this version of the survey data has been split so that dates are in one column,

```python
# Create dict of columns to combine into new datetime column
datetime_cols = {"Part2Start": ["Part2StartDate", 
                                "Part2StartTime"]}

# Load file, supplying the dict to parse_dates
survey_data = pd.read_excel("fcc_survey_dts.xlsx",
                            parse_dates=datetime_cols)

# View summary statistics about Part2Start
print(survey_data.Part2Start.describe())


```

* *Parse non-standard date formats*
```python
# Parse datetimes and assign result back to Part2EndTime
survey_data[“Part2EndTime”] = pd.to_datetime(survey_data[“Part2EndTime”], 
                                             format=“%m%d%Y %H:%M:%S”)
```


## Lesson 3: Database
* `sqlalchemy`’s `create_engine()`
* `sqlite:///filename.db`

```python
# Import sqlalchemy's create_engine() function
from sqlalchemy import create_engine

# Create the database engine
engine = create_engine("sqlite:///data.db")

# View the tables in the database
print(engine.table_names())
---

# Import sqlalchemy's create_engine() function
from sqlalchemy import create_engine

# Create the database engine
engine = create_engine("sqlite:///data.db")

# View the tables in the database
print(engine.table_names())

---
# Create the database engine
engine = create_engine("sqlite:///data.db")

# Create a SQL query to load the entire weather table
query = """
SELECT * 
  FROM weather;
"""

# Load weather with the SQL query
weather = pd.read_sql(query, engine)

# View the first few rows of data
print(weather.head())


```


```python
# Create database engine for data.db
engine = create_engine('sqlite:///data.db')

# Write query to get date, tmax, and tmin from weather
query = """
SELECT date, 
       tmax, 
       tmin
  FROM weather;
"""

# Make a dataframe by passing query and engine to read_sql()
temperatures = pd.read_sql(query, engine)

# View the resulting dataframe
print(temperatures)

```


## Lesson 4 : *Importing JSON Data and Working with APIs*
* Json
* `split` format
```python
try:
    # Load the JSON with orient specified
    df = pd.read_json("dhs_report_reformatted.json",
                      orient="split")
    
    # Plot total population in shelters over time
    df["date_of_census"] = pd.to_datetime(df["date_of_census"])
    df.plot(x="date_of_census", 
            y="total_individuals_in_shelter")
    plt.show()
    
except ValueError:
    print("pandas could not parse the JSON.")

```

* Get data from an API
```python
api_url = "https://api.yelp.com/v3/businesses/search"

# Get data about NYC cafes from the Yelp API
response = requests.get(api_url, 
                        headers=headers, 
                        params=params)

# Extract JSON data from the response
data = response.json()

# Load data to a dataframe
cafes = pd.DataFrame(data["businesses"])

# View the data's dtypes
print(cafes.dtypes)
---


# Create dictionary to query API for cafes in NYC
parameters = {"term": "cafe",
              "location": "NYC"}

# Query the Yelp API with headers and params set
response = requests.get(api_url, 
                        headers=headers, 
                        params=parameters)

# Extract JSON data from response
data = response.json()

# Load "businesses" values to a dataframe and print head
cafes = pd.DataFrame(data["businesses"])
print(cafes.head())

```

* *Set request headers*
> Many APIs require users provide an API key, obtained by registering for the service. Keys typically are passed in the request header, rather than as parameters.

```python
# Create dictionary that passes Authorization and key string
headers = {"Authorization": "Bearer {}".format(api_key)}

# Query the Yelp API with headers and params set
response = requests.get(api_url, 
                        headers=headers, 
                        params=params)

# Extract JSON data from response
data = response.json()

# Load "businesses" values to a dataframe and print names
cafes = pd.DataFrame(data["businesses"])
print(cafes.name)

```

* *Working with nested JSONs*
* Flatten nested JSONs
```python

# Load json_normalize()
from pandas.io.json import json_normalize

# Isolate the JSON data from the API response
data = response.json()

# Flatten business data into a dataframe, replace separator
cafes = json_normalize(data["businesses"],
                       sep="_")

# View data
print(cafes.head())

```

* *Handle deeply nested data*
```python
# Flatten businesses records and set underscore separators
flat_cafes = json_normalize(data["businesses"],
                            sep="_")

# View the data
print(flat_cafes.head())

---

# Specify record path to get categories data
flat_cafes = json_normalize(data["businesses"],
                            sep="_",
                            record_path="categories")

# View the data
print(flat_cafes.head())
---
# Load other business attributes and set meta prefix
flat_cafes = json_normalize(data["businesses"],
                            sep="_",
                            record_path="categories",
                            meta=["name", 
                                  "alias",  
                                  "rating",
                                  ["coordinates", "latitude"], 
                                  ["coordinates", "longitude"]],
                            meta_prefix="biz_")

# View the data
print(flat_cafes.head())


```


