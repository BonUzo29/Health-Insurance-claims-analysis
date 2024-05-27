# Health-Insurance-claims-analysis

### Project Overview

This project focuses on detecting fraudulent, legitimate, and suspicious health insurance claims using machine learning models. The project leverages PySpark to handle large datasets and build scalable machine learning pipelines. It includes generating simulated data, preprocessing, training multiple models, and making predictions on new data.

### Features

- **Simulated Data Generation:** Generates a large dataset with random values for various health insurance claim attributes.
- **Data Preprocessing:** Encodes categorical variables and assembles features into a single vector.
- **Machine Learning Models:** Implements Random Forest, Logistic Regression, and Decision Tree classifiers.
- **Model Training:** Trains models on the generated data and saves the trained models for future use.
- **Predictions:** Makes predictions on new data and outputs the results.
- **Scalability:** Utilizes PySpark for efficient processing of large datasets.

### Project Structure

- `standalone-ml-spark-esemble-claims.py`: Generates simulated health insurance claim data and saves it as a CSV file, preprocesses the data, trains machine learning models, and saves the trained models.
- `printout_models_performance_metrics.py`: Loads the trained models and makes predictions on new data.
- `rf_model/`: Directory to store Random Forest Classifier model.
- `log_reg_model/`: Directory to store Logistic Regression model.
- `dt_model/`: Directory to store Decision Tree Classifier.

#### Data Generation

The `standalone-ml-spark-esemble-claims.py` script creates a large dataset with 1,000,000 rows, including attributes like claim amount, provider type, patient age, claim type, procedure complexity, diagnosis code, treatment cost, service duration, and out-of-network indicator. Each attribute is generated with random values to simulate real-world scenarios.

#### Data Preprocessing

It then preprocesses the data by encoding categorical variables using StringIndexer and assembling all features into a single vector using VectorAssembler. It then splits the data into training and test sets.
Model Training

Three machine learning models are trained:

- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier

These models are trained using the training dataset, and the trained models are saved for future use.

#### Making Predictions

The `printout_models_performance_metrics.py` script loads the trained models and makes predictions on new data. The results include predictions from each model and are saved to a CSV file.


### Getting Started
##### Prerequisites

- Python 3.6+
- Apache Spark
- PySpark


### Installation

Clone the repository:

    git clone https://github.com/BonUzo29/Health-Insurance-claims-analysis.git
    
    cd health-insurance-claims-analysis


## Build PySpark Image using this `docker-compose.yaml` file
```
version: "3"
services:
  pyspark-elyra:
    command:
      - start-notebook.sh
    container_name: pyspark-elyra
    entrypoint:
      - tini
      - -g
      - --
    environment:
      - PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/spark/bin
      - DEBIAN_FRONTEND=noninteractive
      - CONDA_DIR=/opt/conda
      - SHELL=/bin/bash
      - NB_USER=jovyan
      - NB_UID=1000
      - NB_GID=100
      - LC_ALL=en_US.UTF-8
      - LANG=en_US.UTF-8
      - LANGUAGE=en_US.UTF-8
      - HOME=/home/jovyan
      - XDG_CACHE_HOME=/home/jovyan/.cache/
      - APACHE_SPARK_VERSION=3.0.2
      - HADOOP_VERSION=2.7
      - JUPYTER_ENABLE_LAB=yes
      - SPARK_HOME=/usr/local/spark
      - 'SPARK_OPTS=--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info'
    image: ruslanmv/pyspark-elyra:3.0.2
    logging:
      driver: json-file
      options: {}
    networks:
      - pyspark_network
    ports:
      - "8888:8888/tcp"
    stdin_open: true
    tty: true
    user: "1000"
    volumes:
      - /home/project_directory:/home/jovyan/work
    working_dir: /home/jovyan
networks:
  pyspark_network:
    driver: bridge
```


### This contains our spark script mounted on the docker volume we are creating with this command.

```
      docker run --name pyspark-elyra -it -p 8888:8888 \
      -v /C:/project/Bon_Classifier/final_algorithm:/home/jovyan/work \
      -v /C:/project/Bon_Classifier/final_algorithm/spark-processed-claims.py:/home/jovyan/work/spark-processed-claims.py \
      -e KAFKA_ADVERTISED_HOST_NAME=host.docker.internal \
      -e KAFKA_ADVERTISED_PORT=9092 \
      -d ruslanmv/pyspark-elyra:3.0.2
```


### Usage 

After building the docker image of PySpark, we will now use this command to start the script to generate simulated health insurance claim data, preprocess the data and train machine learning models, load the trained models and make predictions on new data:


    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1 ./standalone-ml-spark-esemble-claims.py

<br>
<br>

## Apache Spark output showing new streaming data from spark.
      
     id	claim_amount	provider_type	patient_age	claim_type	procedure_complexity	diagnosis_code	treatment_cost	service_duration	out_of_network
      0	9979.02	Clinic	63	Outpatient	High	A001	2800.28	4	Yes
      1	4167.91	Pharmacy	45	Inpatient	Medium	A001	3425.86	9	No
      2	1334.57	Hospital	23	Outpatient	High	B002	4714.47	5	Yes
      3	6693	Hospital	44	Outpatient	High	A001	359.63	7	No
      4	2576.81	Pharmacy	35	Inpatient	High	A001	1565.87	5	Yes
      5	5702.05	Clinic	40	Inpatient	Medium	B002	3666.44	5	No
      6	3947.9	Clinic	36	Inpatient	High	A001	4395.55	4	Yes
      7	1900.82	Hospital	40	Outpatient	Low	A001	3194.54	7	Yes
      8	2872.69	Pharmacy	23	Outpatient	Medium	B002	4025.98	3	No
      ...
      ....
      ...
      .........
<br>
<br>
<br>

## Apache Spark output after applying 3 separate algorthims used to predict the streaming data.
      
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+
      |id |claim_amount|provider_type|patient_age|claim_type|procedure_complexity|diagnosis_code|treatment_cost|service_duration|out_of_network|rf_prediction|log_reg_prediction|dt_prediction|
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+
      |29 |7898.32     |Hospital     |21         |Inpatient |Low                 |B002          |4988.81       |8               |No            |1.0          |1.0               |1.0          |
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+
      |id |claim_amount|provider_type|patient_age|claim_type|procedure_complexity|diagnosis_code|treatment_cost|service_duration|out_of_network|rf_prediction|log_reg_prediction|dt_prediction|
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+
      |23 |1959.75     |Clinic       |33         |Outpatient|Medium              |B002          |249.01        |6               |Yes           |0.0          |0.0               |0.0          |
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+
      ----+----------------+--------------+-------------+------------------+-------------+
      |id |claim_amount|provider_type|patient_age|claim_type|procedure_complexity|diagnosis_code|treatment_cost|service_duration|out_of_network|rf_prediction|log_reg_prediction|dt_prediction|
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+
      |19 |7676.73     |Pharmacy     |25         |Outpatient|Medium              |B002          |975.01        |1               |Yes           |1.0          |1.0               |1.0          |
      +---+------------+-------------+-----------+----------+--------------------+--------------+--------------+----------------+--------------+-------------+------------------+-------------+


## Our main script that generates the simulated health insurance data is this. 
The first long code below performs the following conditions for predicting the labels using each algorithm:

For the Random Forest (RF) Classifier:
        It considers a claim as fraudulent if either the claim amount is greater than 5000 or the provider type is a hospital.
        It labels a claim as suspicious if the patient age is over 65 and the claim type is "Inpatient".
        Otherwise, it labels the claim as legitimate.

For the Logistic Regression Classifier:
        It applies similar conditions as the RF classifier but uses logistic regression for classification.

For the Decision Tree Classifier:
        It employs the same conditions as the RF classifier but uses a decision tree for classification.

These conditions below are defined in the following section of the code when the simulated data is being generated:

#### Define labels: 0 for legitimate, 1 for fraudulent, 2 for suspicious
```
data = data.withColumn('label', when((data['claim_amount'] > 5000) | (data['provider_type'] == 'Hospital'), 1)
                      .when((data['patient_age'] > 65) & (data['claim_type'] == 'Inpatient'), 2)
                      .otherwise(0))
```
Below is the full script:

```
from pyspark.sql.functions import round, col, rand, randn, when, monotonically_increasing_id
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create SparkSession
spark = SparkSession.builder \
    .appName("HealthInsuranceClaimsDetection") \
    .getOrCreate()

# Set log level to OFF
spark.sparkContext.setLogLevel("OFF")

# Generate simulated data with selected columns
num_rows = 1000000
batch_size = 1

# Define selected columns
selected_columns = [
    'claim_amount',
    'provider_type',
    'patient_age',
    'claim_type',
    'procedure_complexity',
    'diagnosis_code',
    'treatment_cost',
    'service_duration',
    'out_of_network'
]

# Generate data with random values for each column
data = spark.range(num_rows)
data = data.withColumn('claim_amount', round(rand() * 10000, 2))
data = data.withColumn('provider_type', when(rand() < 0.3, 'Hospital').when(rand() < 0.6, 'Clinic').otherwise('Pharmacy'))
data = data.withColumn('patient_age', (randn() * 10 + 40).cast('int'))
data = data.withColumn('claim_type', when(rand() < 0.5, 'Inpatient').otherwise('Outpatient'))
data = data.withColumn('procedure_complexity', when(rand() < 0.3, 'Low').when(rand() < 0.6, 'Medium').otherwise('High'))
data = data.withColumn('diagnosis_code', when(rand() < 0.5, 'A001').otherwise('B002'))
data = data.withColumn('treatment_cost', round(rand() * 5000, 2))
data = data.withColumn('service_duration', (rand() * 10).cast('int'))
data = data.withColumn('out_of_network', when(rand() < 0.5, 'Yes').otherwise('No'))

# Add a unique identifier column
data = data.withColumn("id", monotonically_increasing_id())

# Save generated data to CSV file
data.toPandas().to_csv("generated_data.csv", index=False)

# Define labels: 0 for legitimate, 1 for fraudulent, 2 for suspicious
data = data.withColumn('label', when((data['claim_amount'] > 5000) | (data['provider_type'] == 'Hospital'), 1)
                      .when((data['patient_age'] > 65) & (data['claim_type'] == 'Inpatient'), 2)
                      .otherwise(0))

# Split data into training and test sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

# StringIndexer for categorical variables
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in ['provider_type', 'claim_type', 'procedure_complexity', 'diagnosis_code', 'out_of_network']]

# Define VectorAssembler
vector_assembler = VectorAssembler(inputCols=[col+"_index" if col in ['provider_type', 'claim_type', 'procedure_complexity', 'diagnosis_code', 'out_of_network'] else col for col in selected_columns], outputCol="features")

# Define Machine Learning Models
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
log_reg_classifier = LogisticRegression(featuresCol="features", labelCol="label")
dt_classifier = DecisionTreeClassifier(featuresCol="features", labelCol="label")

# Create ML pipelines
rf_pipeline = Pipeline(stages=indexers + [vector_assembler, rf_classifier])
log_reg_pipeline = Pipeline(stages=indexers + [vector_assembler, log_reg_classifier])
dt_pipeline = Pipeline(stages=indexers + [vector_assembler, dt_classifier])

# Train models
rf_model = rf_pipeline.fit(train_data)
log_reg_model = log_reg_pipeline.fit(train_data)
dt_model = dt_pipeline.fit(train_data)

# Save trained models
rf_model.write().overwrite().save("rf_model")
log_reg_model.write().overwrite().save("log_reg_model")
dt_model.write().overwrite().save("dt_model")


# Make predictions
rf_predictions = rf_model.transform(test_data).withColumnRenamed("prediction", "rf_prediction")
log_reg_predictions = log_reg_model.transform(test_data).withColumnRenamed("prediction", "log_reg_prediction")
dt_predictions = dt_model.transform(test_data).withColumnRenamed("prediction", "dt_prediction")

# Select relevant columns
predictions = rf_predictions.select(selected_columns + ["id", "rf_prediction"])
predictions = predictions.join(log_reg_predictions.select("id", "log_reg_prediction"), on="id", how="inner")
predictions = predictions.join(dt_predictions.select("id", "dt_prediction"), on="id", how="inner")

# Save predicted data to CSV file
predictions.toPandas().to_csv("predicted_data.csv", index=False)

# Print predictions to console in batches of 5
total_rows = predictions.count()
current_row = 0
while current_row < total_rows:
    predictions_batch = predictions.filter(predictions.id >= current_row).limit(batch_size)
    predictions_batch.show(truncate=False)
    current_row += batch_size


```
The code above performs the following tasks:

1. Setting up SparkSession: It creates a SparkSession with the application name "HealthInsuranceClaimsDetection".

2. Generating Simulated Data: It generates simulated data with randomly generated values for each column, such as claim amount, provider type, patient age, claim type, procedure complexity, diagnosis code, treatment cost, service duration, and out-of-network indicator. The generated data is stored in a DataFrame.

3. Defining Labels: It defines labels for the data based on certain conditions. A claim is labeled as fraudulent if the claim amount exceeds 5000 or if the provider type is a hospital. It labels a claim as suspicious if the patient age is over 65 and the claim type is "Inpatient". Otherwise, it labels the claim as legitimate.

4. Splitting Data: It splits the data into training and test sets.

5. Encoding Categorical Variables: It uses StringIndexer to encode categorical variables into numerical indices.

6. Defining Vector Assembler: It defines a VectorAssembler to assemble all features into a single vector.

7. Defining Machine Learning Models: It defines Random Forest, Logistic Regression, and Decision Tree classifiers.

8. Creating ML Pipelines: It creates ML pipelines for each classifier, including the encoding of categorical variables and assembling features.

9. Training Models: It trains the Random Forest, Logistic Regression, and Decision Tree models using the training data.

10. Saving Trained Models: It saves the trained models to disk.

11. Making Predictions: It makes predictions on the test data using the trained models.

12. Selecting Relevant Columns: It selects relevant columns along with predictions from each model.

13. Saving Predicted Data: It saves the predicted data to a CSV file.

14. Printing Predictions: It prints predictions to the console in batches of 5 rows.

    

## We can then use this script (also added to this our GitHub Repository) to evaluate the performance of our saved model.

```
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql.functions import when, col
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# Create SparkSession
spark = SparkSession.builder \
    .appName("ModelEvaluation") \
    .getOrCreate()

# Set log level to OFF
spark.sparkContext.setLogLevel("OFF")

# Load saved models
rf_model = PipelineModel.load("rf_model")
log_reg_model = PipelineModel.load("log_reg_model")
dt_model = PipelineModel.load("dt_model")

# Load test data
test_data = spark.read.csv("generated_data.csv", header=True, inferSchema=True)

# Define labels: 0 for legitimate, 1 for fraudulent, 2 for suspicious
test_data = test_data.withColumn('label', 
                    when((test_data['claim_amount'] > 5000) | (test_data['provider_type'] == 'Hospital'), 1)
                    .when((test_data['patient_age'] > 65) & (test_data['claim_type'] == 'Inpatient'), 2)
                    .otherwise(0))

# Make predictions
rf_predictions = rf_model.transform(test_data)
log_reg_predictions = log_reg_model.transform(test_data)
dt_predictions = dt_model.transform(test_data)

# Evaluate models
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rf_precision = evaluator.evaluate(rf_predictions)
log_reg_precision = evaluator.evaluate(log_reg_predictions)
dt_precision = evaluator.evaluate(dt_predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
rf_recall = evaluator.evaluate(rf_predictions)
log_reg_recall = evaluator.evaluate(log_reg_predictions)
dt_recall = evaluator.evaluate(dt_predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
rf_f1 = evaluator.evaluate(rf_predictions)
log_reg_f1 = evaluator.evaluate(log_reg_predictions)
dt_f1 = evaluator.evaluate(dt_predictions)

# Print performance metrics
print("Random Forest Metrics:")
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)
print()

print("Logistic Regression Metrics:")
print("Precision:", log_reg_precision)
print("Recall:", log_reg_recall)
print("F1-score:", log_reg_f1)
print()

print("Decision Tree Metrics:")
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1-score:", dt_f1)
print()



```

When we run the script above, we get this output

      Random Forest Metrics:
      Precision: 0.99898696080771
      Recall: 0.9989840000000001
      F1-score: 0.9989237508481799
      
      Logistic Regression Metrics:
      Precision: 0.7859431760562169
      Recall: 0.787532
      F1-score: 0.7866015780591822
      
      Decision Tree Metrics:
      Precision: 0.9904867057003954
      Recall: 0.9912270000000001
      F1-score: 0.9907957619832551

The `Precision`, `Recall` and `F1-score` values shows the evaluation metrics for three different machine learning models: Random Forest, Logistic Regression, and Decision Tree. These metrics are important indicators of how well the models are performing on our test data. 

So, specifically, each metric represents:

### Precision: 
Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. In the context of our problem, it represents the ability of the model to correctly identify positive cases (e.g., fraudulent or suspicious claims) out of all the cases it predicted as positive. A higher precision indicates fewer false positives.

### Recall: 
Recall, also known as sensitivity, is the ratio of correctly predicted positive observations to the total actual positive observations. It measures the ability of the model to correctly identify positive cases out of all the actual positive cases. A higher recall indicates fewer false negatives.

### F1-score: 
The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall, considering both false positives and false negatives. It's a useful metric when one wants to compare models across different precision-recall trade-offs.

### Interpreting the values we got for Precision, Recall and F1-score results:

### Random Forest: 
      Random Forest Metrics:
      Precision: 0.99898696080771
      Recall: 0.9989840000000001
      F1-score: 0.9989237508481799
This model has very high precision, recall, and F1-score, indicating that it performs exceptionally well on our test data. It can effectively identify positive cases while minimizing false positives and false negatives.

### Logistic Regression: 
      Logistic Regression Metrics:
      Precision: 0.7859431760562169
      Recall: 0.787532
      F1-score: 0.7866015780591822
The precision, recall, and F1-score for this model are lower compared to Random Forest but still reasonable. It might not perform as well as Random Forest, but it's still a decent model for our task.

### Decision Tree: 
      Decision Tree Metrics:
      Precision: 0.9904867057003954
      Recall: 0.9912270000000001
      F1-score: 0.9907957619832551
Similar to Random Forest, Decision Tree also shows high precision, recall, and F1-score. It performs well on our test data, although the scores are slightly lower than those of Random Forest.
