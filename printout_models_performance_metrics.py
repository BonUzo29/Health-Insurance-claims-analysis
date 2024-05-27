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

# Define a function to plot ROC curve
def plot_roc_curve(predictions, title):
    results = predictions.select(['probability', 'label']).rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
    metrics = BinaryClassificationMetrics(results)
    fpr, tpr = metrics.roc()
    roc_auc = metrics.areaUnderROC

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve for each model
plot_roc_curve(rf_predictions, "Random Forest ROC Curve")
plot_roc_curve(log_reg_predictions, "Logistic Regression ROC Curve")
plot_roc_curve(dt_predictions, "Decision Tree ROC Curve")
