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

