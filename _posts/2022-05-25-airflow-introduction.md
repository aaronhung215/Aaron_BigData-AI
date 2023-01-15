---
layout: post
title: Airflow Introduction
date: 2022-05-25
tags: airflow, python
categories: airflow python
comments: true
---

Airflow introduction

## Lesson 1 : Intro to Airflow
### introduction
* Data Engineering
	*  Taking any action involving data and turning it into a reliable, repeatable, and maintainable process.
* Workflow
	* A set of steps to accomplish a given data engineering task
* Airflow
	* Creation
	* Scheduling
	* Monitoring
	* Can implement programs from any language, but workows are wrien in Python 
	* Implements workows as DAGs: Directed Acyclic Graphs


* DAG code example
```
etl _ dag = DAG( dag_ id='etl _pipeline' , default _ args={"start _ date": "2020-01-08"} )
```
* Running a simple airflow task
` airflow run <dag_ id> <task _ id> <start _ date>   `
` airflow run example-etl download-file 2020-01-10 `

### Airflow 
* DAG, or Directed Acyclic Graph:
	* Directed, there is an inherent ow representing dependencies between components. 
	* Acyclic, does not loop / cycle / repeat. Graph, the actual set of components. 
	* Seen in Airfow, Apache Spark, Luigi
* DAG in Airflow
	* Are written in Python (but can use components written in other languages). 
	* Are made up of components (typically tasks) to be executed, such as operators, sensors, etc. 
	* Contain dependencies defined explicitly or implicitly. 
		* ie, Copy the file to the server before trying to import it to the database service.

* Define a DAG
```python
# Import the DAG object
from airflow.models import DAG

# Define the default_args dictionary
default_args = {
  'owner': 'dsmith',
  'start_date': datetime(2020, 1, 14),
  'retries': 2
}

# Instantiate the DAG object
etl_dag = DAG('example_etl', default_args=default_args)

```

* DAGs on the command line
` airflow list_dags #to show all recognized DAGs. `

### Airflow Web UI
![](https://i.imgur.com/NamSCn9.png)



![](https://i.imgur.com/AQxgVsB.png)

* Task : generate_random_number

![](https://i.imgur.com/mXxdXzj.png)


* Logs

![](https://i.imgur.com/44G1Txz.png)



## Lesson 2 : *Implementing Airflow DAGs*
### Airflow Operator
1. Represent a single task in a workflow. 
2. Run independently (usually). 
3. Generally do not share information. 
4. Various operators to perform different tasks.

* BashOperator
```python
BashOperator( 
	task_id='bash_example' , 
	bash_command='echo "Example!"' , 
	dag=ml_dag
)
```
	* Executes a given Bash command or script. 
	* Runs the command in a temporary directory. 
	* Can specify environment variables for the command.
Example:
```python
# Import the BashOperator
from airflow.operators.bash_operator import BashOperator

# Define the BashOperator 
cleanup = BashOperator(
    task_id='cleanup_task',
    # Define the bash_command
    bash_command='cleanup.sh',
    # Add the task to the dag
    dag=analytics_dag
)
# Define a second operator to run the `consolidate_data.sh` script
consolidate = BashOperator(
    task_id='consolidate_task',
    bash_command='consolidate_data.sh',
    dag=analytics_dag)

# Define a final operator to execute the `push_data.sh` script
push_data = BashOperator(
    task_id='pushdata_task',
    bash_command='push_data.sh',
    dag=analytics_dag)


```
* Operator gotchas
	* Not guaranteed to run in the same location / environment. 
	* May require extensive use of Environment variables. 
	* Can be difficult to run tasks with elevated privileges.

### Tasks
* Instances of operators
* usually assigned to a variable in Python
```python
example_task = BashOperator(task_id='bash_example',
                          bash_command='echo "Example!"',
                          dag=dag)
```
* Referred to by the task_id within the Airflow tools
* Task dependencies
	* upstream *<<*
	* downstream *>>*

![](https://i.imgur.com/dQvg5Oo.png)


* Example 
```python
# Define a new pull_sales task
pull_sales = BashOperator(
    task_id='pullsales_task',
    bash_command='wget https://salestracking/latestinfo?json',
    dag=analytics_dag
)

# Set pull_sales to run prior to cleanup
pull_sales >> cleanup

# Configure consolidate to run after cleanup
consolidate << cleanup

# Set push_data to run last
consolidate >> push_data


```

### additional operator
* Python Operator
```python

from airflow.operators.python_operator import PythonOperator
def printtime():
	print("This goes in the logs!")

python_task = PythonOperator(
    task_id='simple_print',
    python_callable=printme,
    dag=example_dag
)
```

* Arguments
```python 

def sleep(length_of_time):
  time.sleep(length_of_time)

sleep_task = PythonOperator(
    task_id='sleep',
    python_callable=sleep,
    op_kwargs={'length_of_time': 5}
    dag=example_dag
)

```

```python
def pull_file(URL, savepath):
    r = requests.get(URL)
    with open(savepath, 'wb') as f:
        f.write(r.content)   
    # Use the print method for logging
    print(f"File pulled from {URL} and saved to {savepath}")

from airflow.operators.python_operator import PythonOperator

# Create the task
pull_file_task = PythonOperator(
    task_id='pull_file',
    # Add the callable
    python_callable=pull_file,
    # Define the arguments
    op_kwargs={'URL':'http://dataserver/sales.json', 'savepath':'latestsales.json'},
    dag=process_sales_dag
)

# Add another Python task
parse_file_task = PythonOperator(
    task_id='parse_file',
    # Set the function to call
    python_callable=parse_file,
    # Add the arguments
    op_kwargs={'inputfile':'latestsales.json', 'outputfile':'parsedfile.json'},
    # Add the DAG
    dag=process_sales_dag
)
    

```

* Email Operator
```python
from airflow.operators.email_operator import
 EmailOperatoremail_task = EmailOperator(
    task_id='email_sales_report',
    to='sales_manager@example.com',
    subject='Automated Sales Report',
    html_content='Attached is the latest sales report',
    files='latest_sales.xlsx',
    dag=example_dag
)


# Import the Operator
from airflow.operators.email_operator import EmailOperator

# Define the task
email_manager_task = EmailOperator(
    task_id='email_manager',
    to='manager@datacamp.com',
    subject='Latest sales JSON',
    html_content='Attached is the latest sales JSON file as requested.',
    files='parsedfile.json',
    dag=process_sales_dag
)

# Set the order of tasks
pull_file_task >> parse_file_task >> email_manager_task

```


### Scheduling
* State
	* ` running `
	* ` failed `
	* ` success`

![](https://i.imgur.com/vjwp6St.png)



* `start_date`
* `end_date`
* `max_tries`
*  ` schedule_interval`
* cron syntax
![](https://i.imgur.com/lJR0Csp.png)


![](https://i.imgur.com/Croyc56.png)


![](https://i.imgur.com/V4E7qHC.png)


```python
# Update the scheduling arguments as defined
default_args = {
  ‘owner’: ‘Engineering’,
  ‘start_date’: datetime(2019, 11, 1),
  ‘email’: [‘airflowresults@datacamp.com’],
  ‘email_on_failure’: False,
  ‘email_on_retry’: False,
  ‘retries’: 3,
  ‘retry_delay’: timedelta(minutes=20)
}

dag = DAG(‘update_dataflows’, default_args=default_args, schedule_interval=’30 12 * * 3’)
```

## Lesson 3 : Maintaining and monitoring Airflow workflows
### Airflow sensors
	* An operator that waits for a certain condition to be true 
		* Creation of a file 
		* Upload of a database record 
		* Certain response from a web request 
	* Can dene how ofen to check for the condition to be true 
	* Are assigned to tasks

* argument
	* `mode`
		* `poke`
		* `reschedule`
	* ` poke_interval`
	* `timeout`
* File sensor
* `ExternalTaskSensor` : wait for a task in another DAG to complete
* `HttpSensor` : Request a web URL and check for content
* `SqlSensor` : Runs a SQL query to check for content
* Why sensor
	* uncertain when it will be true
	* If  failure  not immediately desired
	* To add task repetition without loops 
### Executors
* SequentialExecutor
	* Runs one task at a time
	* for debugging, not for  production
* LocalExecutor
	* Run on a single system
	* Treats task as processes
	* *Prallelism* defined by the user
	* Can utilize all resources of a given host system
* CeleryExecutor
	* Uses a Celery backend as task manger
	* Multiple worker systems can be defined
	* Is significantly more difficult to setup & configure
	* Extremely powerful method for organization with extensive workflows

![](https://i.imgur.com/9fDL3bY.png)


![](https://i.imgur.com/I7NIgxU.png)


### *Debugging and troubleshooting in Airflow*
* DAG won’t run on schedule
* DAG won’t load
	* `airflow.cfg`
* Syntax errors
	* `airflow list_dags`
	* `python3 dagfile.py`


### *SLAs and reporting in Airflow*


![](https://i.imgur.com/mef1tzy.png)


```python
# Import the timedelta object
from datetime import timedelta

# Create the dictionary entry
default_args = {
  'start_date': datetime(2020, 2, 20),
  'sla': timedelta(minutes=30)
}

# Add to the DAG
test_dag = DAG('test_workflow', default_args=default_args, schedule_interval='@None')


# Import the timedelta object
from datetime import timedelta

test_dag = DAG('test_workflow', start_date=datetime(2020,2,20), schedule_interval='@None')

# Create the task with the SLA
task1 = BashOperator(task_id='first_task',
                     sla=timedelta(hours=3),
                     bash_command='initialize_data.sh',
                     dag=test_dag)


```

![](https://i.imgur.com/LCH5HwO.png)



```python
# Define the email task
email_report = EmailOperator(
        task_id='email_report',
        to='airflow@datacamp.com',
        subject='Airflow Monthly Report',
        html_content="""Attached is your monthly workflow report - please refer to it for more detail""",
        files=['monthly_report.pdf'],
        dag=report_dag
)

# Set the email task to run after the report is generated
email_report << generate_report

```


## Lesson 4 : Building production pipelines in Airflow
### Working with templates

```python
templated _ command=""" echo "Reading {{ params.filename }}" """ t1 = BashOperator(task _ id='template _ task' , bash _ command=templated _ command, params={'filename': 'file1.txt'} dag=example _ dag)

```


### Creating a production pipeline
* Running DAGs & Tasks
` airflow run <dag_id> <task_id> <date>`

* To run a full DAG
` airflow trigger_ga -e <date> <dag_id> `


> Conclusion
> 
![](https://i.imgur.com/A72Stv9.png)


