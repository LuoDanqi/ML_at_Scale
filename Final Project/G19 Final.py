# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Predicting Flight Departure Delays Using Scalable Machine Learning Algorithms
# MAGIC **Team 19**
# MAGIC - Daniel Luo (luodanqi123@ischool.berkeley.edu)
# MAGIC - Don Moon (dongukmoon@ischool.berkeley.edu) 
# MAGIC - Kenneth Pong (kpong@ischool.berkeley.edu) 
# MAGIC - Taeil Goh (taeil.goh@ischool.berkeley.edu) 
# MAGIC 
# MAGIC [Presentation link](https://docs.google.com/presentation/d/1BGA3MSoOQM1jVaHImx6lZ7xfY8-69WpjcDjq60-5Mi0/edit?usp=sharing)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Executive Summary
# MAGIC 
# MAGIC With the net welfare impact of flight delays in the United States alone estimated in the tens of billions of dollars, mitigating the inefficiency losses to those delays has become critical.  One way to address that inefficiency is to predict when they will occur, so that airlines and passengers alike can adapt their plans accordingly to reduce losses.  To that end, we took a dataset of flights from 2015 through 2019, and started to develop features relating to weather, seasonality, and delays.  Following that, we created flight cancellation models initially using data from 2015 through 2018 as a training set based on four algorithms implemented in Spark: Logistic Regression (LR), Decision Trees (DT), Random Forest (RF), and Gradient-Boosted trees (GBT).  The performance of these models were evaluated on the basis of the area under receiver operating characteristic curve (AUROC), as well as the area under precision-recall curve (AUPRC). Over the course of model development we were able to lower the training time for all algorithms from around 2800-3200 seconds down to about 600-800 seconds using an eight-node Databricks cluster with a runtime version of 7.0 ML, and i3.xlarge workers with 4 cores and 30 GB memory.  As well, the largest improvement in model performance was observed by incorporating an inbound arrival delay feature.  By the specified metrics, our model development ultimately showed that GBT yielded the best results at each stage, with a final AUROC of `0.8516` and an AUPRC of `0.9048`.  RF came in a close second, with final AUROC and AUPRC scores of `0.8170` and `0.8657`, respectively.
# MAGIC 
# MAGIC - weather data: `ELEVATION`, `WND_ANGLE`, `WND_SPEED`, `CIG_HEIGHT`, `VIS_DIST`, `ORG_TMP_F`, `ORG_DEW`, `ORG_SLP_1`
# MAGIC - seasonaility: `DAY_OF_WEEK`, `DEP_TIME`, `MONTH`
# MAGIC - delays: `LATE_AIRCRAFT_DELAY`, `CARRIER_DELAY`, `SECURITY_DELAY`
# MAGIC 
# MAGIC Here is the summary of various models and results. 
# MAGIC 
# MAGIC | Model | Inputs Description | Performance (AUROC) | Performance (AUPRC) | Runtime (secs) | 
# MAGIC |------|------|------|------|
# MAGIC | baseline (`LR`) | weather data | `0.6201` | `0.2781` | `2792` |  
# MAGIC | baseline (`DT`) | weather data | `0.4134` | `0.1673` | `3047` | 
# MAGIC | baseline (`RF`) | weather data | `0.6225` | `0.2889` | `3208` | 
# MAGIC | baseline (`GBT`) | weather data | `0.6463` | `0.3106` | `2894` |
# MAGIC | better model (`GBT`) | plus seasonaility | `0.7177` | `0.3858` | `3205` | 
# MAGIC | better model (`GBT`) | plus inbound arrival delay | `0.8324` | `0.8889` | `583` | 
# MAGIC | better model (`GBT`) | max 30 iterations | `0.8388` | `0.8937` | `565` | 
# MAGIC | better model (`GBT`) | plus median delays | `0.8436` | `0.8973` | `692` | 
# MAGIC | better model (`GBT`) | plus avg. delays without median delays | `0.8514` | `0.9040` | `638` |
# MAGIC | best model (`LR`) | limit avg. delays to recent 1 year delay | `0.5` | `0.6179` | `757` | 
# MAGIC | best model (`RF`) | limit avg. delays to recent 1 year delay | `0.8170` | `0.8657` | `801` | 
# MAGIC | best model (`GBT`) | limit avg. delays to recent 1 year delay | `0.8516`| **`0.9048`** | `631` | 
# MAGIC 
# MAGIC 
# MAGIC ##### Confusion matrix statistics  for the best model (GBT + limit avg. delays to recent 1 year delay). 
# MAGIC - Test accuracy= 0.778429
# MAGIC - f1 = 0.77931
# MAGIC - precision = 0.700964
# MAGIC - **false positive = 0.193322**
# MAGIC - true positive (recall) = 0.732752
# MAGIC 
# MAGIC NOTE: We have used MulticlassClassificationEvaluator which seem to show lower numbers than expected. But due to the last minute request for the confusion matrix, we did not have a chance to create confusion matrix from the scratch. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Introduction 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Background
# MAGIC 	
# MAGIC  According to the Bureau of Transportation Statistics’ aggregate data on flights in the United States, airline flight departure delays have become a widespread problem, accounting for about 22% of all flights.  [Peterson et al (2013)](https://www.ingentaconnect.com/content/lse/jtep/2013/00000047/00000001/art00007?token=006019fbf642d264168263c4a6f644a4621766a3340253675496e58654624736a42312074217d6b3b2450db8b4842e1d) establishes that these delays represent a significant cause of inefficiency, with such delays causing effects such as lost labour productivity, opportunity costs, and reductions in consumer spending on travel and tourism goods and services.  Given a 30 percent reduction in flight delays, Peterson argues that US net welfare would increase by $38.5 billion.  Given the scale of such numbers, the importance of addressing the effects of airline delays becomes clear from a business perspective, as addressing flight delays would go a long way toward improving economic efficiency.  One way to go about this is to predict delays - if airline companies and passengers can find out ahead of a flight’s appointed departure time that a flight is likely to be delayed, then they may mitigate their losses by adapting their behaviour and circumstances in anticipation of the coming disruption.  Thus, this project concerns itself with developing a statistical model to predict airline delays based on airline and weather data.
# MAGIC  
# MAGIC #### Question Formulation
# MAGIC 
# MAGIC  In order to direct our efforts, we must start with identifying the target audience of the model we are developing, as well as the stakeholders.  The model is being developed primarily with airport operating authorities in mind as the intended recipient, as the ability to predict flight delays allows them to optimize their efforts to compensate for deviations from plan.  Of course, anyone impacted by downstream actions resulting from predicted delays are considered stakeholders in this model, albeit of secondary importance.  This group is represented largely by the airline companies themselves, as well as the passengers.  In view of the intended recipient and stakeholders, the question we are addressing is: “Given airline and weather data two hours prior to departure, how can we maximize the utility from that data in making flight delay predictions for airport, airlines, and passengers?”.
# MAGIC 
# MAGIC  Based on the available data, we have two options for the outcome variable in flight prediction.  We can either predict 1) whether a flight is delayed for 15 minutes or not via the `DEP_DEL15` feature, or 2) the length of flight delay by bucket  via the `DEP_DELAY_GROUP` feature.  While knowing the estimated length of the delay would undoubtedly provide more valuable actionable information to the passengers and airline companies than simply whether it will be delayed 15+ minutes at all, we have decided for the purposes of this project to focus on the former prediction.  
# MAGIC 
# MAGIC The reasons are twofold: Firstly, as discussed in the proceeding Metrics section, there is a very high cost to over-predicting the length of delays, and the complexity required to achieve a satisfactory level of accuracy is substantial.  The consequence of that complexity is that the model would generalize poorly to unseen data outside of the development process (i.e. the model will be overfitted to the data).  Secondly, we want to encourage the recipients of these predictions to behave conservatively -  given that such a system may modify the behaviour of people exposed to this information, we want to minimize the probability of passengers taking actions as a result of this system causing them to miss flights.  In this case, this means providing less detail.
# MAGIC  
# MAGIC #### Business Perspective
# MAGIC 
# MAGIC  The ideal result of this flight delay prediction system takes the form of a streaming context, in which all available data are presented to the algorithm two hours before departure time in order to yield a real-time prediction of whether the flight will be delayed.  However, due to time constraints, the first iteration of the system presented in this paper is concerned solely with batch prediction of delays in a pre-compiled dataset.  The intention is to use the model’s prediction on static datasets as a proof of concept, which can be used to refine algorithm selection, feature selection, and feature engineering considerations.  Once those refinements are identified and implemented, future models will be implemented in a streaming context for real-time flight delay predictions.  Currently, though, real-time predictive capability is out of scope.  Still, out of consideration for the future development of real-time capabilities, the model developed here will restrict itself to data that are available two hours before departure.
# MAGIC 
# MAGIC #### High level data sets
# MAGIC 	
# MAGIC  The input data for this project come from two primary sources, as provided as part of the parameters for this project: Firstly, for flights, we draw on the US Department of Transportation's TranStats data collection of on-time performance data. This dataset encompasses flights from 2015 through 2019 inclusive. For the purposes of this dataset, the training data are derived from the flight information from 2015 through 2018 inclusive, while the held-out data against which the models will be evaluated come from the 2019 flight data. A second key dataset input comes from the National Oceanic and Atmospheric Administration (NOAA)’s repository for weather data.  This dataset includes variables such as temperature, precipitation, atmospheric pressure, and visibility, among others.  This data also includes the weather stations from which those measurements were taken, as well as the timestamp of those measurements.  The former in particular is used in order to join the data with the weather stations.  The vintage of this weather data mtaches that of the flight data (2015-2019).  From a geographical standpoint, this project also incorporates location data on the airports (which also comes from [the Bureau of Transportation Statistics](https://www.transtats.bts.gov/Fields.asp?Table_ID=288)), as well as the weather stations (which comes from [the US Air Force’s Air Weather Service Master Station Catalog](http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html)).
# MAGIC 
# MAGIC #### Metrics
# MAGIC 
# MAGIC  In terms of the metrics that we can use to gauge the accuracy of the models being developed, there are several main choices: 
# MAGIC  
# MAGIC  *Simple accuracy* (the % of accurate predictions)<br>
# MAGIC  *Area Under the Receiver Operating Characteristic Curve* (the AUROC)<br>
# MAGIC  *Area Under the Precision-Recall Curve* (the AUPRC)
# MAGIC  
# MAGIC  To choose the proper metric, we first consider the relative importance of Type I errors (false positives) versus Type II errors (false negatives).  In the context of a flight delay prediction system, the relative effect of the possible errors becomes clear when evaluating the consequences from the perspective of a passenger: If a passenger is informed that a flight is not delayed when it actually is (false negative), the failure means that the passenger will arrive at their gate at the appointed time, only to be inconvenienced by a longer than expected wait.  However, if that same passenger is informed a flight is delayed when it is actually on time (false positive), then the passenger may alter their itinerary in response, which may cause them to miss their flight altogether.  Obviously, merely being inconvenienced for a flight is significantly less costly than missing it altogether.  The false positive is equally undesirable from the perspective of the business - because airline operating costs are largely fixed for each flight, and the failure of a passenger to take their assigned flight means they take up space in a subsequent flight, a passenger missing their flight essentially doubles the seat allocation required to deliver them to their destination.
# MAGIC 
# MAGIC Given the importance of the delayed flights class over the non-delayed one, our choice of metric should quantify both the models’ performance in maximizing accuracy amongst the delayed set, as well as accurately predicting as many delayed flights as possible overall.  Hence, the metrics used to evaluate the performance of the models will be AUROC and AUPRC.  The reason is that AUROC is a straightforward and well-known metric, but is somewhat hampered in the case of this scenario because [it does not distinguish between the relative costs of false positive and false negatives](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).  By comparison, AUPRC is a more accurate metric [owing to its focus on precision and recall](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
# MAGIC ) (i.e. how the model handles the positive examples).   However, AUPRC is trickier to interpret because it focuses specifically on one subset of the data, which makes AUROC a good complimentary metric to aid in results interpretation despite its limitations.
# MAGIC 
# MAGIC More information on the AUPRC can be found [here](https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/).
# MAGIC 
# MAGIC 
# MAGIC #### SOTA solutions
# MAGIC 
# MAGIC  The problem space of machine learning predictions for flight delays has been tackled with a number of techniques.  According to [Chakrabarty (2019)](https://arxiv.org/ftp/arxiv/papers/1903/1903.06740.pdf), approaches run the gamut of ML algorithms, including gradient boosting classifiers, decision trees, random forests, Adaboost, k-Nearest Neighbours, linear regression (weighted and unweighted), naive bayes, logistic regression, support vector machines, recurrent neural networks, and more.  [Shao, Prabowo et al (2019)](https://arxiv.org/pdf/1911.01605.pdf) quantified the performance of several different algorithms (light GBM, linear regression, multilayer perceptron - MLP, and support vector regressors - SVR), albeit with root mean squared error as the choice of metric.  Their analysis examined the performance of these algorithms when applied to flight, weather, and ATC data in varying combinations, and found that in all of those combinations, Light GBM yielded a noticeably better score than that of the other techniques.  The positive results based on Light GBM were echoed by [Ye et al (2020)](https://www.researchgate.net/publication/340360352_A_Methodology_for_Predicting_Aggregate_Flight_Departure_Delays_in_Airports_Based_on_Supervised_Learning), who found that in terms of mean square error (MAE) and mean absolute error (MSE), the technique yields better results than linear regression, Support Vector Machines (SVM), and Extremely Randomized Trees (ExtraTR).  
# MAGIC 
# MAGIC On the other end of the flight lifecycle, [Kuhn and Jamadagni (2017)](http://cs229.stanford.edu/proj2017/final-reports/5243248.pdf) developed a model predicting arrival delays which achieved an F1 Score of 0.91 with a logistic regression model using only three variables - departure delay time, taxi delay, and origin airport.  Obviously, we cannot use departure delays for the purposes of this model, but the importance score of the features highlights one important fact - that prior delays are the largest indicator of future delays.
# MAGIC 
# MAGIC For Spark-specific applications of machine learning models in predicting airline delays, [Yanying et al (2019)](https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050919X00186/1-s2.0-S1877050919320241/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGEaCXVzLWVhc3QtMSJIMEYCIQDuovURyOL0YozCOefi8KKMqSApSlgBtGoEsLk1tt4PYAIhANbvnpc5KnxaFSiisvb7lMYUqUfJWDplJGEBgBtQWy3TKrQDCCoQAxoMMDU5MDAzNTQ2ODY1IgxcfNshvqz2oinaVwsqkQN5iey82DUQxDedBX1hYyzZiBjEWkrzP%2BaRhFevzxliGdJY5snKG5yJEwJHTckgC8BiX%2BatOplG7iQqLX7GVMsh8DEJWGVq9I0kx3W0GA%2FAwfonWOCP1erfhHPWyRkJnyTUeYbui%2F6kjruWxNYU4ud7M8ijCCC%2BbBkBH%2F9L2p76VTV7glmqfKy9u3ijRDzuW3ygwvjQPKxb8T7bN5oTk3ahhnui99miNfwFCThU69HvCUNWw52nIkQ61Z7%2BSgY%2Bd%2F%2BuzQ6%2B773L4WIUTZBdDsvgfuJh1YveLOZiQt%2B3QlE6of%2BmajnsqwPBIpaQI5j%2FwpHYYKD%2BjDfegvdK8tCzz4p90im%2FYBhldnMcN9AzQvS7bV6mKIOI4k9NAeWNPKHruLtrfLqmwYO5%2Ft1HoTDKdwvNc9SNOOrz9NHNHXvLdVM2%2FcdDmevjKMdlxKgbPYZ%2FLBJz2EsvEW%2BtUI8w9454leJVALZ1awa9tTvqrj%2F9gUhJAHZ0gXTNECqMJfvMEUljUMGooRojeMOAAsZI1HROe6TAHjDtk4r5BTrqAUCf2M3GWVh7NiT8qS0eeMiSTcxcfleJASkfjdYW83BqI%2B%2Bb%2B3SQpYPIbL6Ria9VndQwvm5byD69cw8dOsYmRa3r4FUTr2jrQeN6mxERr4N63iZdwxkdfYIWOCJrBQgb%2FXq9hv%2BkcnsyplBk02OBSk%2BTMaNwwlUgE6ElYHPADqfGB7iTOKb2aqIQq304DN89CHny%2FMv3yxVxJeNtcC3qISlyIFYUkKXhBnel3XKIi9A%2BPwKnbkLmHzO1U6uFt9VNPmrnOocu4OZ5cqP8aioSx2KYzoV3%2BmSewFpBedcxh%2B8mOCtnL61LHLvRuA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200730T095113Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4ACPTQEU%2F20200730%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0ba50eb24cc07dba8e2bfb1e252eed75f6afdbf73d69581a3bbfdcdd5e809b25&hash=6b2425d394323d299be9ba7708c290768f5da418dcf427b720982e9e9e2a6882&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050919320241&tid=spdf-fccadcae-a5fe-431e-a775-fefbd14b6689&sid=432d03e595cbf84d508a6d2870cb05e66630gxrqa&type=client) employed logistic regression, SVM, Naive Bayes, and Decision Trees on Spark, using varying numbers of nodes for training purposes.  
# MAGIC 
# MAGIC For auxiliary sources, [Henriques and Feiteira (2018)](https://reader.elsevier.com/reader/sd/pii/S1877050918317319?token=41B696200E1CF218B24A7686ED1FB1979EDACF7C2D66BC2110D11BC6ABC7CEE0F3AD5D79E792E3C8F4A180FABC9B20FC) - predicted delays using DTs, Random Forests, and multilayer perceptrons on ATL airport in particular, scored by ROC, F, and accuracy.  Although the scope of their work was limited to only one airport, it still found that weather features were less influential compared to delay propagation information.  Finally, [Gopalakrishnan and Balakrishnan (2017)](http://www.mit.edu/~hamsa/pubs/GopalakrishnanBalakrishnanATM2017.pdf) made use of a rather specialized algorithm known as Markov Jump Linear System (MJLS).  Their work also compiled median delays by airport alongside prediction errors provided at multiple timeframes prior to flight departure.  However, we were ultimately unable to pursue investigation into MJLS for want of a library to implement it with.
# MAGIC 
# MAGIC To summarize, the main algorithms, metrics, and best results observed of the papers reviewed were as follows:
# MAGIC 
# MAGIC | Authors | Main Techniques Applied | Metric(s) | Best Results (Model Used) |
# MAGIC |------|------|------|------|
# MAGIC |Chakrabarty (2019) | Gradient Boosting Classifier, SMOTE (for imbalance correction) | Training/validation accuracy, Precision, Recall, F1 | 86.68% Training accuracy, 85.73% validation accuracy, 0.88 Precision, 0.86 Recall, 0.85 F-1 (GBC)|
# MAGIC |Gopalakrishnan and Balakrishnan (2017) | Neural Networks, Classification and Regression Trees, Linear Regression, Markov Jump Linear System | Mean error | 4.7 minutes mean error for 2 hour time horizon (MJLS) |
# MAGIC |Henriques and Feiteira (2018) | Decision Tree, Random Forest, Multilayer Perceptron, SMOTE and undersampling for imbalance correction | Area Under Receiver Operating Characteristic Curve, F1 Score, Accuracy | 0.56 ROC, 0.79 F1, 85.63% Accuracy (MLP with SMOTE, outliers removed, excluding dep_delay and real_dep_time) |
# MAGIC |Kuhn and Jamadagni (2017) | Decision Tree, Logistic Regression, Neural Network | Precision, Recall, F1 | 0.93 Precision, 0.88 Recall, 0.91 F1 on delayed arrival predictions (DT) |
# MAGIC |Shao, Prabowo et al (2019) | Light GBM, Logistic Regression, MLP, Support Vector Regression, Multilayer Perceptron | Root Mean Squared Error | 0.37 RMSE (LGBM) |
# MAGIC |Yanying et al (2019) | Logistic Regression, Support Vector Machine, Naive Bayes, Decision Tree | Area Under Precision-Recall Curve, Area Under Receiver Characteristic Curve | 0.439 PR, 0.558 AUC (DT) |
# MAGIC |Ye et al (2020) | Light GBM, Extra Trees, Support Vector Machines, Linear Regression |Mean Square Error, Mean Absolute Error | 92.74 MSE, 6.58 minutes MAE (Light GBM) |

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

from pyspark.sql import types
import pyspark.sql.functions as F

from sklearn import metrics

#MLlib
from pyspark.ml.feature import RFormula 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import VectorAssembler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

sqlContext = SQLContext(sc)

DATA_PATH = 'dbfs:/mnt/mids-w261/data/datasets_final_project/'
MASTER_CORD_PATH = '/FileStore/tables/407131831_T_MASTER_CORD.csv'

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### EDA of Airline Data Set
# MAGIC 
# MAGIC Based on a preliminary exploration of the attributes available in the Airlines dataset, we identified seven main categories out of `109` features, summarized as follows:
# MAGIC 
# MAGIC   1. Time (`Year`, `Quarter`, `Month`, `FL_DATE`, `DEP_TIME`...) 
# MAGIC   2. Carriers Information
# MAGIC   3. Aiport and Geographical Information (`ORIGIN_AIRPORT`, `DEST_AIRPORT`)
# MAGIC   4. __Departure Delay Information__ (`DEP_DELAY`, `DEP_DELAY_NEW`, `DEP_DEL15`, `DEP_DELAY_GROUP`, `CANCELLED`... )
# MAGIC   5. Arrival Delay Information and other Delay information (`WEATHER_DELAY`, `NAS_DELAY`..)
# MAGIC   6. Aircraft Attributes (`TAXI`, `WHEELS`..) - assuming those data will not have before departure
# MAGIC   7. Route Information (e.g. `Distance`, `DIVERTED`.. if applicable)
# MAGIC   
# MAGIC Among the Departure Delay information features, we found several possible outcome variables:
# MAGIC 
# MAGIC   1. for baseline, we can focus on `DEP_DELAY_15`
# MAGIC   2. we can use `DEP_DELAY_NEW`
# MAGIC   3. `Cancelled` (Not Considered). 
# MAGIC   4. `DEP_DELAY`
# MAGIC   5. `WEATHER_DELAY`, `NAS_DELAY` should be out intermediate outcome variables
# MAGIC   6. `DEP_DELAY_GROUP` has 14 category values
# MAGIC   7. `Diverted` could also be intergreted 
# MAGIC   
# MAGIC ##### Initial EDA of Airline Data
# MAGIC For the initial EDA, we focused on the first 6 month of 2015 airline data with only ALT and ORD airports (__See the EDA Reference notebook for full details__). Here is the summary of the steps we took:
# MAGIC   1. Load the airline data and Displayed the first 1000 rows of airline_6m data as well as the basic statistics
# MAGIC   2. Checked the time component (seasonality) and carrier component of the flights
# MAGIC   3. Checked the origin and destination airport information
# MAGIC   4. Checked the departure delay and arrival delay distribution
# MAGIC   
# MAGIC Initial Findings from the airline dataset:
# MAGIC   1. For 6m EDA dataset, there is only two airport ATL, ORD. for train data set, 2015 - 2018, there are 322 airports.
# MAGIC   2. Based on the 6m EDA dataset, the flight counts has no significant seasonality, probabily because those two ariports are fairly large.
# MAGIC   3. Depature flights happen mainly between 5 a.m. to 9 p.m.
# MAGIC   4. The amount of flights has significant differences between Carriers
# MAGIC   5. The flights numbers to different destination airports are not evenly distributed,i.e. LGA has about 9700 flights and COD only has 2 flights in the first 6 months of 2015.
# MAGIC   6. For Departure delays: 
# MAGIC      6.1. "DEP_DEL15" is a Binary variable 0 - not delayed 15 mins; 1 - delyed more than 15 mins.
# MAGIC      6.2. "DEP_DELAY" is a flight delay in minutes, which is extremely skewed.  
# MAGIC      6.3. "DEP_DELAY_GROUP" is categorical variable which bins the "DEP_DELAY" in 15 minutes; and we can also see that the groups are also skewed. Majority of deleyed flights are in group 1 to 3.
# MAGIC      6.4. "DEP_DELAY_NEW" is also flight delay in minutes but it removed the negative values.
# MAGIC   7. "DEP_DEL15" and "DEP_DELAY_GROUP" will be used as our outcome variable and check relationship with other features
# MAGIC   8. The histogram of ARR DELAY and DEP DELAY are almost identical
# MAGIC   9. All other delays are also extremely skewed, we probably can take a mean instead of average as input variable.
# MAGIC   
# MAGIC Once we fleshed out our question formulation, we settled on using the `DEP_DEL15` attribute as the outcome variable.  For more information on the rationale behind choosing this atttribute over `DEP_DELAY_GROUP`, please see the Question Formulation section of the main report.  At this point, we ruled out using the aircraft-related attributes group to build our models, as that we could not be certain that those variables will be available two hours prior to departure.  

# COMMAND ----------

# load airlines data
parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data_6m/part-*.parquet"]
airlines_6m = spark.read.option("header", "true").parquet(*parquet_path)
#print(f"{len(airlines_6m.columns)} columns")
# print()
# print(airlines_6m.columns)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Departure Delay vs. Other variables in the full dataset
# MAGIC At this point, we started looking for __relationships between the departure delay attributes and data from the other six main attribute categories__ in the full set of airlines data. Here is the summary of EDA steps we took:
# MAGIC   1. Load the full dataset and transformed them  
# MAGIC   2. Checked the total delay rate for the full training set (2015-2018)
# MAGIC   3. Random sampled out from the training dataset for furthur EDA
# MAGIC   4. Checked the correlationships between the `DEP_DEL15` and `DEP_DELAY_GROUP` with other category variables
# MAGIC   5. Built the correlation heatmap for internal correlations

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1: Loading and Transformation
# MAGIC 
# MAGIC **Transformation 1**: Dropping cancelled flights
# MAGIC 
# MAGIC *Rationale*: Cancelled flights represent a different set of circumstances compared to flight delays, with different implications from a business perspective, as it represents a net loss in the total number of flights conducted rather than a mere deviation from the expected schedule.  Because of this, we have elected to remove flights marked as cancelled from both the training and test datasets.
# MAGIC 
# MAGIC **Transformation 2**: Inputing a value of zero for missing values in NAS, Weather, Security, and Carrier delay features
# MAGIC 
# MAGIC *Rationale*: In cases where delay attributes were missing values outright, it is reasonable to infer that there was no delay owing to those causes.  Therefore, null values in the `NAS_DELAY`, `WEATHER DELAY`, `SECURITY_DELAY` and `CARRIER DELAY` attributes had a value of zero attached to them to reflect that.
# MAGIC 
# MAGIC **Transformation 3**: Extraction of hour value from `DEP_TIME` feature
# MAGIC 
# MAGIC *Rationale*: In order to associate flight information with the hourly weather information, we extracted the hour of day attribute from both feature sets.  In addition, to account for the fact that we are making our predictions two hours prior to the scheduled departure, we applied a 2 hour subtraction to the timeframe of the weather predictions, because we would necessarily be making delay predictions based on weather data available two hours prior to departure.

# COMMAND ----------

def load_airline_data_eda():

    def transform_airline_data(parquet_path):
        """
        read from s3 parquets
        transform given airline dataset to
        - for delay fields, replace null value by zero
        """
        
        airlines = spark.read.option("header", "true").parquet(*parquet_path)
          
        # Seperate DEP_TIME into Hour for join (We might want to use TimeStamp values with date to join for 2 hours before)
        airlines = airlines.withColumn("DEP_HOUR", F.bround('DEP_TIME', -2)/100)

        # drop cancelled flight for baseline 
        airlines = airlines.where(F.col('cancelled') == 0)

        # replace null value by zeros
        airlines = airlines.withColumn('NAS_DELAY', F.when(airlines['NAS_DELAY'].isNull(), 0).otherwise(airlines['NAS_DELAY']) )
        airlines = airlines.withColumn('WEATHER_DELAY', F.when(airlines['WEATHER_DELAY'].isNull(), 0).otherwise(airlines['WEATHER_DELAY']) )
        airlines = airlines.withColumn('SECURITY_DELAY', F.when(airlines['SECURITY_DELAY'].isNull(), 0).otherwise(airlines['SECURITY_DELAY']) )
        airlines = airlines.withColumn('LATE_AIRCRAFT_DELAY', F.when(airlines['LATE_AIRCRAFT_DELAY'].isNull(), 0).otherwise(airlines['LATE_AIRCRAFT_DELAY']) )
        airlines = airlines.withColumn('CARRIER_DELAY', F.when(airlines['CARRIER_DELAY'].isNull(), 0).otherwise(airlines['CARRIER_DELAY']) )
        airlines = airlines.withColumn('ARR_DELAY', F.when(airlines['ARR_DELAY'].isNull(), 0).otherwise(airlines['ARR_DELAY']) )
        airlines = airlines.withColumn('ARR_DELAY_NEW', F.when(airlines['ARR_DELAY_NEW'].isNull(), 0).otherwise(airlines['ARR_DELAY_NEW']) )
        
        return airlines
  
    # load training airlines data
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2015.parquet", 
                    "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2016.parquet",
                    "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2017.parquet",
                    "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2018.parquet"]
    airline_train = transform_airline_data(parquet_path)
    
    # load test airlines data
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2019.parquet"] 
    airline_test = transform_airline_data(parquet_path)
    
    # load sample airlines data
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data_6m/part-*.parquet"] 
    airline_6m = transform_airline_data(parquet_path)
      
    return airline_train, airline_test, airline_6m

airline_train, airline_test, airlines_6m = load_airline_data_eda()
display(airline_train.sample(False, 0.00001))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2: Total Delay Rate
# MAGIC 
# MAGIC Based on the training set, which encompasses flight data from 2015 to 2018, we calculated that there was a 22% global probability that any given flight would be delayed for more than 15 minutes.

# COMMAND ----------

flights_with_delay = airline_train.where(F.col('dep_del15') == 1).count()
flights_with_nodelay = airline_train.where(F.col('dep_del15') == 0).count()
print("flights with delay:", flights_with_delay)
print("flights with no delay:", flights_with_nodelay)
print(f"likihood of flight delay: {flights_with_delay/flights_with_nodelay:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 3: Random Sampling from Training Set
# MAGIC In order to run some analysis into the training data set, we took a random sample from that dataset to perform our EDA upon.

# COMMAND ----------

# Making a sample set of the full training set to conduct EDA 
sample_train = airline_train.sample(False, 0.001, 42)
print(sample_train.count())
sample_df = sample_train.toPandas()

# Get the related features
airline_columns_tokeep = ['DEP_DEL15', 'DEP_DELAY_GROUP', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK','DEP_HOUR', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'WEATHER_DELAY', 'NAS_DELAY', 'CARRIER_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY',  'ARR_DELAY', 'DISTANCE', 'DIV_AIRPORT_LANDINGS']
sample_df2 = sample_df[airline_columns_tokeep]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 4: Correlations between the DEP_DEL15 and DEP_DELAY_GROUP with other category variables
# MAGIC  When we plotted the categorical data on the subset chosen for analysis, we noticed that the data told us a different story from the six month subset.  For example, seasonality now became visible in the broader dataset, with the summer months (June through August) and December having higher a higher probability of flight delay than in other months.  We also noticed that late-night flights (with a departure time between 11PM to 3AM) were prone to significantly higher delay probabilities than at other times of day.  Finally, plotting the probability of delay against airport IDs showed us that there was significant variation in the probability of flight delay between airports.
# MAGIC 
# MAGIC  We then repeated the exercise on the continuous data type features.  From this, we learned that all of the individual delay features (weather, NAS, carrier, and security) are significantly correlated with the `DEP_DEL15` and `DEP_DELAY_GROUP` attributes.  As well, the arrival delay attribute also shows large differences between `DEP_DEL15` = 0 and `DEP_DEL15` = 1.  This makes logical sense, because when a flight's departure is delayed, its arrival will normally be delayed as well (this can be alleviated with higher flight speeds, but that's only an option in certain cases).  From this, we got the idea that delays in the prior arrival of a flight might be useful in predicting delays in subsequent departures.  Finally, the distance and diversion attributes did not seem to have any significant impact on the delay attributes.

# COMMAND ----------

# Check the correlations between time-related features and DEL15
# plot the delay rates versus each time measure 
plt.figure(figsize = (25,5))
plt.subplot(1,5,1)
sample_df2.groupby("QUARTER")["DEP_DEL15"].mean().plot.bar()
plt.subplot(1,5,2)
sample_df2.groupby("MONTH")["DEP_DEL15"].mean().plot.bar()
plt.subplot(1,5,3)
sample_df2.groupby("DAY_OF_MONTH")["DEP_DEL15"].mean().plot.bar()
plt.subplot(1,5,4)
sample_df2.groupby("DAY_OF_WEEK")["DEP_DEL15"].mean().plot.bar()
plt.subplot(1,5,5)
sample_df2.groupby("DEP_HOUR")["DEP_DEL15"].mean().plot.bar()

# COMMAND ----------

# plot the delay rates versus each time measure 
plt.figure(figsize = (25,10))
plt.subplot(2,1,1)
sample_df2.groupby("ORIGIN_AIRPORT_ID")["DEP_DEL15"].mean().plot.bar()
plt.subplot(2,1,2)
sample_df2.groupby("DEST_AIRPORT_ID")["DEP_DEL15"].mean().plot.bar()

# COMMAND ----------

fig, ax_grid = plt.subplots(2, 4, figsize=(20,10))
y = sample_df2['DEP_DEL15']
FIELDS = sample_df2.columns
for idx, feature in enumerate(FIELDS[9:]):
    x = sample_df2[feature]
    sns.boxplot(x, y, ax=ax_grid[idx//4][idx%4], orient='h', linewidth=.5)
    ax_grid[idx//4][idx%4].invert_yaxis()
fig.suptitle("Individual Features vs. Outcome (DEP_DEL15)", fontsize=15, y=0.9)
display(plt.show())

# COMMAND ----------

fig, ax_grid = plt.subplots(2, 4, figsize=(20,10))
y = sample_df2['DEP_DELAY_GROUP']
FIELDS = sample_df2.columns
for idx, feature in enumerate(FIELDS[9:]):
    x = sample_df2[feature]
    sns.boxplot(x, y, ax=ax_grid[idx//4][idx%4], orient='h', linewidth=.5)
    ax_grid[idx//4][idx%4].invert_yaxis()
fig.suptitle("Individual Features vs. Outcome (DEP_DELAY_GROUP)", fontsize=15, y=0.9)
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 5: Correlation Heatmap
# MAGIC Based on the correlation heat map that we ran on the initial feature selection, we observed that `DEP_DEL15` was correlated with the hour of departure, as well as the other delay attributes (weather, NAS, carrier, late aircraft, and arrival), which was expected.  We also saw that there were inner correlations between arrival delay and the other delay categories, as well as departure hour.

# COMMAND ----------

# plot heatmap for correlations matrix 
corr = sample_df2[FIELDS[:]].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.5)
plt.title("Correlations between the selected features.")
display(plt.show())

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ##### Step 6: Reduction of variable list to only essential variables
# MAGIC 
# MAGIC *Rationale*: The given datasets contain a lot of extraneous features not used in the model development process - hence, to minimize the size of the files used for computation and further joining, we have dropped all variables from the flight table, retaining only the following variables: `DEP_DEL15`, `WEATHER_DELAY`, `NAS_DELAY`, `CARRIER_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`, `MONTH`, `DAY_OF_WEEK`, `ORIGIN_AIRPORT_ID`, `DEST_AIRPORT_ID`, `DEP_TIME`, `FL_DATE`, `DEP_HOUR`, `DAY_OF_WEEK`, `DAY_OF_MONTH`

# COMMAND ----------

AIRLINE_COLUMNS_TOKEEP = ['DEP_DEL15', 'WEATHER_DELAY', 'NAS_DELAY',  'CARRIER_DELAY', 'OP_CARRIER', 'SECURITY_DELAY', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'LATE_AIRCRAFT_DELAY', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_TIME', 'FL_DATE', "DEP_HOUR", "DAY_OF_WEEK", "DAY_OF_MONTH", 'TAIL_NUM', 'CRS_ARR_TIME', 'ARR_DELAY_NEW', 'MONTH', 'YEAR']

def load_airline_data(use_small_set_only = False):
    """
    load ariline data and transform 
    """

    def transform_airline_data(parquet_path):
        """
        read from s3 parquets
        transform given airline dataset to
        - for delay fields, replace null value by zero
        """
        
        airlines = spark.read.option("header", "true").parquet(*parquet_path)
          
        # Seperate DEP_TIME into Hour for join (We might want to use TimeStamp values with date to join for 2 hours before)
        airlines = airlines.withColumn("DEP_HOUR", F.bround('DEP_TIME', -2)/100)

        # drop cancelled flight for baseline 
        airlines = airlines.where(F.col('cancelled') == 0).select(AIRLINE_COLUMNS_TOKEEP)

        # replace null value by zeros
        airlines = airlines.withColumn('NAS_DELAY', F.when(airlines['NAS_DELAY'].isNull(), 0).otherwise(airlines['NAS_DELAY']) )
        airlines = airlines.withColumn('WEATHER_DELAY', F.when(airlines['WEATHER_DELAY'].isNull(), 0).otherwise(airlines['WEATHER_DELAY']) )
        airlines = airlines.withColumn('SECURITY_DELAY', F.when(airlines['SECURITY_DELAY'].isNull(), 0).otherwise(airlines['SECURITY_DELAY']) )
        airlines = airlines.withColumn('LATE_AIRCRAFT_DELAY', F.when(airlines['LATE_AIRCRAFT_DELAY'].isNull(), 0).otherwise(airlines['LATE_AIRCRAFT_DELAY']) )
        airlines = airlines.withColumn('CARRIER_DELAY', F.when(airlines['CARRIER_DELAY'].isNull(), 0).otherwise(airlines['CARRIER_DELAY']) )
        airlines = airlines.withColumn('ARR_DELAY_NEW', F.when(airlines['ARR_DELAY_NEW'].isNull(), 0).otherwise(airlines['ARR_DELAY_NEW']) )

        return airlines
  
    airline_train, airline_test = None, None
    if not use_small_set_only:
        # load airlines data
        parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2015.parquet", 
                        "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2016.parquet",
                        "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2017.parquet",
                        "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2018.parquet"]
        airline_train = transform_airline_data(parquet_path)

        airline_parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2019.parquet"] 
        airline_test = transform_airline_data(airline_parquet_path)
    
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data_6m/part-*.parquet"] 
    airline_6m = transform_airline_data(parquet_path)
      
    return airline_train, airline_test, airline_6m

# COMMAND ----------

# MAGIC %md 
# MAGIC ### EDA Weather dataset
# MAGIC 
# MAGIC After examining the airline dataset, we turned to conducting EDA on the weather data.  For the purposes of this stage, we used the full 2015 weather station data.  This file contained 116 million records, encompassing reports from 12,794 stations.  We saw that this dataset included several attributes that might be useful for joining it to the airlines dataset - namely, time, latitude, longitude, elevation, and name.

# COMMAND ----------

# Load the data of 2015 weather
parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2015a.parquet"]
weather_eda = spark.read.option("header", "true").parquet(*parquet_path)

# COMMAND ----------

# Count of each column
weather_eda_col = weather_eda.select("STATION").count()
print("Total Columns:", weather_eda_col)

# Unique value of each column
for col in weather_eda.columns:
  print(col,weather_eda.select(col).distinct().count())

# COMMAND ----------

# Count percentage of completeness of the whole dataset
weather_eda_agg = weather_eda.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in weather_eda.columns])
# for col in weather_eda.columns:
#   print(col,weather_eda.where(F.col(col).isNull()).count()/weather_eda_col)
display(weather_eda_agg)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Findings from weather data
# MAGIC The key findings coming out of the weather dataset can be summarized as follows:
# MAGIC 
# MAGIC - These datasets possess temporal and geospatial attributes which can be used to join it to the airline dataset
# MAGIC - A large number of columns are related to weather reporting rather than the parameters of the weather itself (i.e. a lot of what's available is metadata)
# MAGIC - A lot of the features are extremely sparse, either will empty cells or duplicated values
# MAGIC - Broadly speaking, the weather dataset can be broken down into eight categories:
# MAGIC   1. Wind （WND)
# MAGIC   2. Ceiling Height (CIG, GA1-GA4)
# MAGIC   3. Air Temperature (TMP)
# MAGIC   4. Visibility (VIS, AU1-AU4)
# MAGIC   5. Air Pressure (SLP)
# MAGIC   6. Precipitation <Dew Point, Relative Humidity, Snow Depth> (AA1 - AA4， AJ1)
# MAGIC   7. Solar (GN1)
# MAGIC   8. General Weather Code (AU1 to AU4)
# MAGIC - Most of columns are not NULL but space strings, which makes harder to identify missing records

# COMMAND ----------

# Check for Ceiling Height GA1 to GA4 
# sky cover layer coverage code, coverage quality code, base height dimension, height quality code
for col in ['GA1','GA2','GA3','GA4']:
  weather_eda.select(col).distinct().show(4)
  
# Check for Ceiling Height GA1 to GA4 
# Intensity code, descirptor code, precipitation code, obscuration code
for col in ['AU1','AU2','AU3','AU4']:
  weather_eda.select(col).distinct().show(4)
  
# Check for Precipitation AA1 to AA4
# period, depth, condition, quality code
for col in ['AA1','AA2','AA3','AA4']:
  weather_eda.select(col).distinct().show(4)
  
# Solar data
weather_eda.select('GN1').distinct().show(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Missing value interpolation
# MAGIC 
# MAGIC As part of the EDA, we attempted to impute the missing values by using Pandas, which required conversion of the data type.  From there, we tried to determine if there was sufficient non-missing data records to allow us to fill in the gaps between them. Based on the check of the null record percentages, there were a lot of columns which were almost completely devoid of data.  The extreme sparsity of those attributes made it impossible to interpolate the missing values.  Thus, these attributes were excluded from the features used for the models.  Going forward, it was decided that we would only use attributes with at least 80% data completeness.  These attributes were `WND` (wind angle and speed), `CIG` (ceiling height) , `TMP` (temperature), `VIS` (visibility), and `SLP` (sea level pressure).

# COMMAND ----------

# Making a sample set of the full wather_eda set to conduct EDA 
weather_sample = weather_eda.sample(False, 0.001, 42)
print(weather_sample.count())
weather_sample_df = weather_sample.toPandas()

# Get the related features
weather_columns_tokeep = ['WND', 'CIG', 'GA1','GA2','GA3','GA4', 'TMP', 'VIS', 'AU1', 'AU2', 'AU3','AU4', 'SLP', 'AA1','AA2', 'AA3', 'AA4', 'AJ1', 'GN1']
weather_sample_df2 = weather_sample_df[weather_columns_tokeep]

# Replace space string columns into NAN
weather_sample_df2 = weather_sample_df2.replace(r'^\s*$', np.nan, regex=True)
# Get the total number of NAN of each column
weather_results = weather_sample_df2.isnull().sum(axis = 0)

# Check the percentage of NAN values
weather_results/116487

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Weather Data Transformation
# MAGIC 
# MAGIC Data transformations were also undertaken for the available weather data following the EDA stage.  The following is the list of transformations applied to the weather data and their reasons:
# MAGIC 
# MAGIC **Transformation 1**: Drop most of columns with vast null values
# MAGIC 
# MAGIC *Rationale*: Due to the sparsity of data in these columns, as well as the need to minimize the file sizes to improve processing speeds, most of the columns primarily comprised of null values were dropped
# MAGIC 
# MAGIC **Transformation 2**: Imputation of missing values for weather attributes
# MAGIC 
# MAGIC *Rationale*: For the `WND`, `CIG`, `VIS`, `TMP`, `DEW`, and `SLP` attributes, [the format document](https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf) indicates that missing values are denoted by extremely large numbers - for example, the missing value for wind speed equates to 999.9 metres per second.  Obviously, these values have the potential to spoil the quality of any model that accepts those placeholder numbers at face value.  Thus, specific values were imputed to account for missing data.  Wherever possible, default values were chosen that reflected normal conditions.  The missing value imputations were conducted as follows:
# MAGIC - `WND_ANGLE` (Wind Angle): Missing value placeholders (999) were converted to 0
# MAGIC - `WND_SPEED`:Missing value placeholders (9999) were converted to 0
# MAGIC - `TMP` (Temperature) - Missing value placeholders (9999) were converted to 15 degrees celsius
# MAGIC - `DEW` (Dew point) - Missing value placeholders (9999) were converted to 15 degrees celsius
# MAGIC - `SLP` (Sea level pressure) - Missing value placeholders (99999) were converted to 10132 - standard sea-level pressure is 1013.25 hectopascals; input unit is scaled by 10, so 1013.25 * 10 = 10132)

# COMMAND ----------

WEATHER_COLUMNS = ['STATION', 'ELEVATION', 'WND_ANGLE', 'WND_SPEED', 'CIG_HEIGHT', 'VIS_DIST', 'TMP_F', 'DEW_1', 'SLP_1', 'LATITUDE', 'LONGITUDE', 'DATE']

def load_weather_data():
    """
    load weather data and transform
    """
    # keep only interesting variables
    
    def transform_weather_data(weather):
        """
        string => integer
        WND_ANGLE => 0 if missing
        WND_SPEED => 0 if missing
        TMP => 15 (standard temperature) if missing 
        DEW => 16 (standard dew point) if missing
        SLP => 10132 (standard see level pressure) if missing
        """
        split_weather_field = F.split(weather['WND'], ',')
        weather = weather.withColumn('WND_ANGLE', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('WND_ANGLE', F.when(weather['WND_ANGLE'] == 999, 0).otherwise(weather['WND_ANGLE']) )

        weather = weather.withColumn('WND_SPEED', split_weather_field.getItem(3).cast("double"))
        weather = weather.withColumn('WND_SPEED', F.when(weather['WND_SPEED'] == 9999, 0).otherwise(weather['WND_SPEED']) )

        split_weather_field = F.split(weather['CIG'], ',')
        weather = weather.withColumn('CIG_HEIGHT', split_weather_field.getItem(0).cast("double"))

        split_weather_field = F.split(weather['VIS'], ',')
        weather = weather.withColumn('VIS_DIST', split_weather_field.getItem(0).cast("double"))

        split_weather_field = F.split(weather['TMP'], ',')
        weather = weather.withColumn('TMP_F', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('TMP_F', F.when(weather['TMP_F'] == 9999, 15).otherwise(weather['TMP_F']) )

        split_weather_field = F.split(weather['DEW'], ',')
        weather = weather.withColumn('DEW_1', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('DEW_1', F.when(weather['DEW_1'] == 9999, 15).otherwise(weather['DEW_1']) )

        split_weather_field = F.split(weather['SLP'], ',')
        weather = weather.withColumn('SLP_1', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('SLP_1', F.when(weather['SLP_1'] == 99999, 10132).otherwise(weather['SLP_1']) )

        weather = weather.select(WEATHER_COLUMNS)

        # Seperate DATE Columns to DATE and Hour for airline join
        weather = weather.withColumn("DATE_PART", F.to_date(F.col("DATE")))\
                                 .withColumn("HOUR_PART", F.hour(F.col("DATE"))) 
        
        return weather 
    
    # Filter weather dataset with only 2015 first 6 months.
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2015a.parquet"]
    weather_sample = spark.read.option("header", "true").parquet(*parquet_path)
    weather_sample = weather_sample.filter(F.col('DATE')<='2015-07-01')
    weather_sample = transform_weather_data(weather_sample)
    
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2015a.parquet", 
                   "dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2016a.parquet",
                   "dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2017a.parquet",
                   "dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2018a.parquet"]
    weather_train = spark.read.option("header", "true").parquet(*parquet_path)
    weather_train = transform_weather_data(weather_train)
    
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2019a.parquet"]
    weather_test = spark.read.option("header", "true").parquet(*parquet_path)
    weather_test = transform_weather_data(weather_test)
    
    return weather_train, weather_test, weather_sample

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Create Index table
# MAGIC 
# MAGIC The next step that we had to take was to join the weather data with the airport data itself.  To accomplish this, we got lat/long data [from the BTS](https://www.transtats.bts.gov/Fields.asp?Table_ID=288), as well as [the weather stations themselves](http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html).  Once we had the coordinates of the respective points, we created indices for both the airport and weather tables by station ID, longitude, and latitude in order to expedite the joining process.  Finally, the two tables were joined by Euclidean, or straight line distance using the `pairwise_distances_argmin_min` function (i.e. we associated the closest weather stations to each of the airports) in order to associate weather station data with the airports.  For more on the rationale behind the chosen distance metric, please see the main report.
# MAGIC 
# MAGIC - Obtain geo-coordinate for airports from the same website (193497807_T_MASTER_CORD.csv)
# MAGIC - Create Index airport table with Airport Station ID, LATITUDE, LONGITUDE
# MAGIC - Create Index weather table for only Station ID, LATITUDE, LONGITUDE
# MAGIC - Joined both index table by euclidean distance

# COMMAND ----------

def create_index_tables(weather):
  """
  create index table 
  todo: push csv file to s3
  """
  
  def create_weather_index(airport_idx, weather):
      # Create Index weather table for only Station ID, LATITUDE, LONGITUDE
      weather.createOrReplaceTempView('weather_tmp_view')
      weather_index = spark.sql(
        """
        SELECT
          DISTINCT STATION, LATITUDE, LONGITUDE
        FROM
          weather_tmp_view """
      ).toPandas()

      # Remove NAs
      weather_index = weather_index.dropna()
      
      # Calculate by euclidean distance
      weather_station_idx = metrics.pairwise_distances_argmin_min(airport_index[['LATITUDE', 'LONGITUDE']], weather_index[['LATITUDE', 'LONGITUDE']], metric='euclidean')[0]
      airport_idx['weather_station_id'] = [weather_index.iloc[i]['STATION'] for i in weather_station_idx]

      # Rename it to be airport_weather_index
      airport_weather_index = spark.createDataFrame(airport_index)
      airport_weather_index.createOrReplaceTempView('airport_weather_index')
  
      return airport_weather_index
  
  airport_geocoord_columns_tokeep = ['AIRPORT_ID', 'DISPLAY_AIRPORT_NAME', 'LATITUDE', 'LONGITUDE']
  # import lat/lon
  # 193497807_T_MASTER_CORD.csv 
  airport_geocoord = spark.read.csv(MASTER_CORD_PATH, header="true", inferSchema="true")

  airport_geocoord = airport_geocoord.select(airport_geocoord_columns_tokeep)

  # Filter it with only US airports (But destination airport might be international)
  airport_geocoord = airport_geocoord.groupBy('AIRPORT_ID').agg(F.avg('LATITUDE'), F.avg('LONGITUDE'))

  #display(airport_geocoord.sample(False, 0.001))

  # Create Index airport table
  airport_index = airport_geocoord.withColumnRenamed("avg(LATITUDE)","LATITUDE").withColumnRenamed("avg(LONGITUDE)","LONGITUDE").toPandas()
  airport_index = airport_index.dropna()
  #airport_index.shape
  #airport_index.head

  #airport_weather_index_sample = create_weather_index(airport_index, weather_sample)
  airport_weather_index = create_weather_index(airport_index, weather)
  #airport_weather_index_test = create_weather_index(airport_index, weather_test)

  return airport_weather_index

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Join Weather Data
# MAGIC 
# MAGIC - Joined Weather Data to Airline Data based on index table and time

# COMMAND ----------

def join_weather_airline(airlines, weather, airport_weather_index):
    """
    join airline and weather data using index table
    """
  
    # Join the weather station id to origin airport 
    airlines_weather_all = airlines.join(F.broadcast(airport_weather_index), airport_weather_index.AIRPORT_ID==airlines.ORIGIN_AIRPORT_ID, 'inner')
      # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("LATITUDE","ORIGIN_LAT").withColumnRenamed("LONGITUDE","ORIGIN_LON")\
                      .withColumnRenamed("AIRPORT_ID","AIRPORT_ORG").withColumnRenamed("weather_station_id","ORG_WS_ID")  

    # Join the weather station id to Destination airport 
    airlines_weather_all = airlines_weather_all.join(F.broadcast(airport_weather_index), airport_weather_index.AIRPORT_ID==airlines_weather_all.DEST_AIRPORT_ID, 'inner')
      # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("LATITUDE","DEST_LAT").withColumnRenamed("LONGITUDE","DEST_LON")\
                      .withColumnRenamed("AIRPORT_ID","AIRPORT_DEST").withColumnRenamed("weather_station_id","DEST_WS_ID")

    # weather to airline join (Origin Airport)
    airlines_weather_all = airlines_weather_all.join( weather, [airlines_weather_all.ORG_WS_ID == weather.STATION, airlines_weather_all.FL_DATE == weather.DATE_PART, airlines_weather_all.DEP_HOUR == weather.HOUR_PART], 'left')

    # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("STATION","ORG_WS_ID_2").withColumnRenamed("ELEVATION","ORG_ELEVATION")\
                      .withColumnRenamed("WND_ANGLE","ORG_WND_ANGLE").withColumnRenamed("WND_SPEED","ORG_WND_SPEED")\
                      .withColumnRenamed("CIG_HEIGHT","ORG_CIG_HEIGHT").withColumnRenamed("VIS_DIST","ORG_VIS_DIST")\
                      .withColumnRenamed("TMP_F","ORG_TMP_F").withColumnRenamed("DEW_1","ORG_DEW_1").withColumnRenamed("SLP_1","ORG_SLP_1")\
                      .withColumnRenamed("LATITUDE","ORG_LATITUDE").withColumnRenamed("LONGITUDE","ORG_LONGITUDE")\
                      .withColumnRenamed("DATE","ORG_DATE").withColumnRenamed("DATE_PART","ORG_DATE_PART").withColumnRenamed("HOUR_PART","ORG_HOUR_PART")

    # weather to airline join (Destination Airport)
    airlines_weather_all = airlines_weather_all.join( weather, [airlines_weather_all.DEST_WS_ID == weather.STATION, airlines_weather_all.FL_DATE == weather.DATE_PART, airlines_weather_all.DEP_HOUR == weather.HOUR_PART], 'left')

    # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("STATION","DEST_WS_ID_2").withColumnRenamed("ELEVATION","DEST_ELEVATION")\
                      .withColumnRenamed("WND_ANGLE","DEST_WND_ANGLE").withColumnRenamed("WND_SPEED","DEST_WND_SPEED")\
                      .withColumnRenamed("CIG_HEIGHT","DEST_CIG_HEIGHT").withColumnRenamed("VIS_DIST","DEST_VIS_DIST")\
                      .withColumnRenamed("TMP_F","DEST_TMP_F").withColumnRenamed("DEW_1","DEST_DEW_1").withColumnRenamed("SLP_1","DEST_SLP_1")\
                      .withColumnRenamed("LATITUDE","DEST_LATITUDE").withColumnRenamed("LONGITUDE","DEST_LONGITUDE")\
                      .withColumnRenamed("DATE","DEST_DATE").withColumnRenamed("DATE_PART","DEST_DATE_PART").withColumnRenamed("HOUR_PART","DEST_HOUR_PART")
    
    return airlines_weather_all


# COMMAND ----------

def preload_data():
    """
    output: sample, test, and train
    """
    # 1. load airline data 
    airline_train, airline_test, airline_sample = load_airline_data()

    # 2. load weather data 
    weather_train, weather_test, weather_sample = load_weather_data()

    # 3. create index table 
    airport_weather_index = create_index_tables(weather_train)
    print("airport_weather_index", airport_weather_index.count())

    # 4 join airline and weather data using index table 
    airlines_weather_all_sample = join_weather_airline(airline_sample, weather_sample, airport_weather_index)
    airlines_weather_all_test = join_weather_airline(airline_test, weather_test, airport_weather_index)
    airlines_weather_all_train = join_weather_airline(airline_train, weather_train, airport_weather_index)

    print("airlines_weather_all_sample", airlines_weather_all_sample.count())
    print("airlines_weather_all_test", airlines_weather_all_test.count())
    print("airlines_weather_all_train", airlines_weather_all_train.count())
    
    return airlines_weather_all_sample, airlines_weather_all_test, airlines_weather_all_train
  
airlines_weather_all_sample, airlines_weather_all_test, airlines_weather_all_train = preload_data()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Findings from the Feature Engineering
# MAGIC 
# MAGIC To evaluate the correlations of the created features on the outcome variable, we plotted the features on a correlation heat map matrix to examine the results.  At a glance, we noticed that the outcome variable `DEP_DEL15` was correlated to most of the features we selected.  We also observed some internal correlations, but most of them were obvious, like Airport ID to attributes relating to the airport (e.g. airport eleveation, latitude and longitude), and the departure hour to the arrival hour.  Some internal correlation was also observed in the weather data.

# COMMAND ----------

# Making a sample set of the full joined dataset do the Correlation Matrix 
airlines_weather_all_sample_df = airlines_weather_all_sample.sample(False, 0.001, 42)
print(weather_sample.count())
airlines_weather_all_sample_pd = airlines_weather_all_sample_df.toPandas()

# Get the related features
FIELDS = airlines_weather_all_sample_df.columns

# plot heatmap for correlations matrix 
corr = airlines_weather_all_sample_pd[FIELDS[:]].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.5)
plt.title("Correlations between the selected features.")
display(plt.show())

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Baseline Model
# MAGIC 
# MAGIC In order to establish baseline model performance, we ran the three selected algorithms with three different sets of features. For the first iterations of the model, we did not use any delay attributes, as we were still working out how to incorporate airport and security delays at the time. The first model included, for both origin and destination, the wind angle, wind speed, ceiling height, visibility distance, and temperature. We tried three different algorithms: **Logistic Regression, Random Forests, and Gradient Boosted Trees**.

# COMMAND ----------

def run_LR(label, train, test):
    """
    train the model with logistic regression and predict on test set
    evaluate using ROC and PRC
    https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression
    """
    start = time.time()
    print("--------------------------------")
    print("Lostistic Regression")
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol = 'features', labelCol = label)
    lrModel = lr.fit(train)

    trainingSummary = lrModel.summary
    roc = trainingSummary.roc.toPandas()

    #evaluation
    predictions = lrModel.transform(test)

    evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC").setLabelCol(label)
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderPR").setLabelCol(label)
    print('Test Area Under PR', evaluator.evaluate(predictions))
    print(f"{time.time() - start:.2f} seconds elapsed.")
    print("--------------------------------")    
  
def run_RF(label, train, test):
    """
    train the model with random forest and predict on test set
    evaluate using ROC and PRC
    """
    start = time.time()
    print("--------------------------------")
    print("Random Forest")
    rf = RandomForestClassifier(numTrees=10, featuresCol = 'features', labelCol = label)
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)

    evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC").setLabelCol(label)
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderPR").setLabelCol(label)
    print('Test Area Under PR', evaluator.evaluate(predictions))
    print(f"{time.time() - start:.2f} seconds elapsed.")
    print("--------------------------------")    

def run_GBT(label, train, test, max_iter=10):
    """
    train the model with GBT and predict on test set
    evaluate using ROC and PRC
    """
    start = time.time()
    print("--------------------------------")
    print("GBT")
    gbt = GBTClassifier(maxIter=max_iter, seed=42, featuresCol = 'features', labelCol = label)
    gbtModel = gbt.fit(train)
    predictions = gbtModel.transform(test)

    evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC").setLabelCol(label)
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderPR").setLabelCol(label)
    print('Test Area Under PR', evaluator.evaluate(predictions))
    print(f"{time.time() - start:.2f} seconds elapsed.")
    print("--------------------------------")

# COMMAND ----------

def label_datasets(label, train, test = None):
    """
    - (todo) there should be no missing data at this moment but there is still
    - mark the dataset with label and features
    """
    def label_dataset(label, df):
        """
        help function to label single data set
        """
        df = df.na.drop()

        features = df.columns
        features.remove(label)

        # Set up a ML pipeline
        stages = []

        assembler = VectorAssembler(inputCols=features, outputCol="features")
        stages += [assembler]

        pipeline = Pipeline(stages = stages)
        pipelineModel = pipeline.fit(df)
        df = pipelineModel.transform(df)

        selectedCols = [label, 'features'] + features
        df = df.select(selectedCols)
        
        return df 
  
    # if using sample set
  

    if test is None:
        train, test = label_dataset(label, train).randomSplit([0.8, 0.2], seed = 42)
    else:
        train = label_dataset(label, train)
        test = label_dataset(label, test)

    return train, test
  
def filter_datasets(sample, full_test, full_train, variables_tokeep):
    """
    filter fields with only input variables and label variable to be used for modeling
    """
    sample = sample.select(variables_tokeep)
    full_test = full_test.select(variables_tokeep)
    full_train = full_train.select(variables_tokeep)
    
    return sample, full_test, full_train
  
def run_models(params, sample_all, full_test_all, full_train_all, use_full_data = True):
    """
    run 3 models using given dataset 
    """
    sample, full_test, full_train = filter_datasets(sample_all, full_test_all, full_train_all, params)    
    if use_full_data:
      train, test = label_datasets(params[0], full_train, full_test)
    else:
      train, test = label_datasets(params[0], sample)
    print("--------------------------------")
    print(params[1:])
    print("--------------------------------")
    run_LR(params[0], train, test)
    run_RF(params[0], train, test)
    run_GBT(params[0], train, test)
    print()

# COMMAND ----------

def baseline():
  sample_all, full_test_all, full_train_all = preload_data()
  # weather data for departure and arrival airports 
  # the first index is an outcome variable 
  model_params = ['DEP_DEL15', 'ORG_ELEVATION', 'ORG_WND_ANGLE', 'ORG_WND_SPEED', 'ORG_CIG_HEIGHT', 'ORG_VIS_DIST', 'ORG_TMP_F', 'ORG_DEW_1', 'ORG_SLP_1', 'DEST_ELEVATION', 'DEST_WND_ANGLE', 'DEST_WND_SPEED', 'DEST_CIG_HEIGHT', 'DEST_VIS_DIST', 'DEST_TMP_F', 'DEST_DEW_1', 'DEST_SLP_1']

  run_models(model_params, sample_all, full_test_all, full_train_all)

# COMMAND ----------

baseline()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Baseline Model Results And Analysis
# MAGIC 
# MAGIC The first iteration of the model yielded results of fairly dubious utility, as seen by the AUROC and AUPRC metrics, which ranged from `0.62` to `0.65` and `0.27` to `0.31`, respectively.
# MAGIC 
# MAGIC For the second iteration, we introduced elevation, dew point, and sea level pressure for both origin and destination.  Interestingly, this actually made LR and RF perform worse, though GBT saw a very slight improvement.  Based on this, we decided that these attributes were not useful.
# MAGIC 
# MAGIC Thus, for the third model, we elected to discard the variables added to the second iteration, and instead add the day of week and departure time.  This actually yielded noticeable improvements on the first iteration, with the AUROC metrics rising by about `0.06` for all models, and the AUPRC metric increasing by around `0.05` for LR, and around `0.06` for RF and GBT.

# COMMAND ----------

# MAGIC %md ## Optimization 
# MAGIC 
# MAGIC #### Feature Engineering 
# MAGIC 
# MAGIC After we completed the baseline model, we switched our attention to feature engineering, and turned our attention to making use of the delay attributes.  The first thing we had to do was figure out how to incorporate the security and aircraft delays.  To do this, we took the average values for those delays, grouped by carrier and by airport.  In this process, we evaluated the use of both mean and median for aggregation.
# MAGIC 
# MAGIC ##### Incorporating arrival delay for inbound flight that is used for outbound flight
# MAGIC 
# MAGIC Next, we wanted to incorporate the the prior arrival delay for flights, based on the idea that delays can cascade and cause downstream delays.  In order to get the prior arrival delay, we joined the outbound and inbound flights on a number of criteria.  Specifically, we joined the inbound and outbound flights by their origin and destination pairs, using the flight date and tail numbers as join criteria.  In terms of time, we also added criteria of departure time of two hours after the arrival time, and four hours prior to it.  In this process, we excluded a small minority of late night flights whose prior arrival and departure crossed the midnight line, as that caused issues with the joining process.

# COMMAND ----------

AIRLINE_INBOUND_COLUMNS_TOKEEP = ['FL_DATE', 'DEST_AIRPORT_ID', 'TAIL_NUM', 'CRS_ARR_TIME', 'ARR_DELAY_NEW']
AIRLINE_OUTBOUND_COLUMNS_TOKEEP = ['DEP_DEL15', 'WEATHER_DELAY', 'NAS_DELAY',  'CARRIER_DELAY', 'OP_CARRIER', 'SECURITY_DELAY', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'LATE_AIRCRAFT_DELAY', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_TIME', 'FL_DATE', "DEP_HOUR", "DAY_OF_WEEK", "DAY_OF_MONTH", 'TAIL_NUM' ,'MONTH', 'YEAR']

def join_arrival_data(airline):
  airline_outbound = airline.select(AIRLINE_OUTBOUND_COLUMNS_TOKEEP)
  airline_inbound = airline.select(AIRLINE_INBOUND_COLUMNS_TOKEEP)\
                    .withColumnRenamed("DEST_AIRPORT_ID","DEST_AIRPORT_ID_IN")\
                    .withColumnRenamed("TAIL_NUM","TAIL_NUM_IN").withColumnRenamed("FL_DATE","FL_DATE_IN")\
                    .withColumnRenamed("ARR_DELAY_NEW","INBOUND_ARR_DELAY_NEW")
  airline_joined = airline_outbound.alias("outbound").join(airline_inbound.alias("inbound"), 
                    [F.col("outbound.ORIGIN_AIRPORT_ID") == F.col("inbound.DEST_AIRPORT_ID_IN")
                    , F.col("outbound.TAIL_NUM") == F.col("inbound.TAIL_NUM_IN")
                    , F.col("outbound.FL_DATE") == F.col("inbound.FL_DATE_IN")
                    , (F.col("outbound.DEP_TIME") - 200) > F.col("inbound.CRS_ARR_TIME")
                    , (F.col("outbound.DEP_TIME") - 400) < F.col("inbound.CRS_ARR_TIME")], "left")

#modified to join airline data to include arrival delay
def preload_data_v2():
    """
    output: sample, test, and train
    """
    # 1. load airline data 
    airline_train, airline_test, airline_sample = load_airline_data()
    
    # (additional) add arrival delay    
    airline_train = join_arrival_data(airline_train)
    airline_test = join_arrival_data(airline_test)
    airline_sample = join_arrival_data(airline_sample)

    # 2. load weather data 
    weather_train, weather_test, weather_sample = load_weather_data()

    # 3. create index table 
    airport_weather_index = create_index_tables(weather_train)

    # 4 join airline and weather data using index table 
    airlines_weather_all_sample = join_weather_airline(airline_sample, weather_sample, airport_weather_index)
    airlines_weather_all_test = join_weather_airline(airline_test, weather_test, airport_weather_index)
    airlines_weather_all_train = join_weather_airline(airline_train, weather_train, airport_weather_index)

    print("airlines_weather_all_sample", airlines_weather_all_sample.count())
    print("airlines_weather_all_test", airlines_weather_all_test.count())
    print("airlines_weather_all_train", airlines_weather_all_train.count())
    
    return airlines_weather_all_sample, airlines_weather_all_test, airlines_weather_all_train
  

# COMMAND ----------

def create_carrier_security_delay_index_avg(airline_train):
    """
    calculate average value of delays grouped by carrier and airport
    """
    carrier_security_delay_index = airline_train.groupBy("OP_CARRIER", "ORIGIN_AIRPORT_ID").agg(F.mean('LATE_AIRCRAFT_DELAY').alias("MED_LATE_AIRCRAFT_DELAY"), 
                                                                                                F.mean('CARRIER_DELAY').alias("MED_CARRIER_DELAY"), 
                                                                                                F.mean('SECURITY_DELAY').alias("MED_SECURITY_DELAY") )
    return carrier_security_delay_index

def create_carrier_security_delay_index_median(airline_train):
    """
    calculate median value of delays grouped by carrier and airport
    """
#     carrier_security_delay_index = airline_train.groupBy("OP_CARRIER", "ORIGIN_AIRPORT_ID").agg(F.mean('LATE_AIRCRAFT_DELAY'), F.mean('CARRIER_DELAY'), F.mean('SECURITY_DELAY'))
    airline_train.registerTempTable('df')
    carrier_security_delay_index = spark.sql(
      """
      SELECT
        OP_CARRIER, ORIGIN_AIRPORT_ID, percentile_approx(LATE_AIRCRAFT_DELAY,0.5) as MED_LATE_AIRCRAFT_DELAY,\
        percentile_approx(CARRIER_DELAY,0.5) as MED_CARRIER_DELAY,\
        percentile_approx(SECURITY_DELAY,0.5) as MED_SECURITY_DELAY
      FROM
        df
      GROUP BY
        OP_CARRIER, ORIGIN_AIRPORT_ID""")
    
    return carrier_security_delay_index
  
def join_delays_groupby_airport_carrier(carrier_security_delay_index, airline):
    """
    join delay by carrier and airport ID 
    """
    airlines_with_delays = airline.join(F.broadcast(carrier_security_delay_index), 
                                         [ "OP_CARRIER", "ORIGIN_AIRPORT_ID"], 'inner')

    return airlines_with_delays
  
def preload_data_v3():
    """
    load data and transform to be fed into the models 
    - load airline data 
    - add inbound arrival delay
    - associate delay info with airport and carrier
    - load weather data 
    - join airline and weather data  
    output: sample, test, and train
    """
    # 1. load airline data 
    airline_train, airline_test, airline_sample = load_airline_data()
    
    # 2. add inbound arrival delay 
    airline_train = join_arrival_data(airline_train)
    airline_test = join_arrival_data(airline_test)
    airline_sample = join_arrival_data(airline_sample)
    
    # 3. filter previous year delay
    airline_train_2018 = airline_train.where( (F.col('YEAR') == 2018) )
      
    # 4. associate delay info with airport and carrier
#     carrier_security_delay_index = create_carrier_security_delay_index_median(airline_train)
    carrier_security_delay_index = create_carrier_security_delay_index_avg(airline_train_2018)
    
    airline_train = join_delays_groupby_airport_carrier(carrier_security_delay_index, airline_train)
    airline_test = join_delays_groupby_airport_carrier(carrier_security_delay_index, airline_test)
    airline_sample = join_delays_groupby_airport_carrier(carrier_security_delay_index, airline_sample)

    # 5. load weather data 
    weather_train, weather_test, weather_sample = load_weather_data()

    # 6. create index table 
    airport_weather_index = create_index_tables(weather_train)
    print("airport_weather_index", airport_weather_index.count())

    # 7. join airline and weather data using index table 
    airlines_weather_all_sample = join_weather_airline(airline_sample, weather_sample, airport_weather_index)
    airlines_weather_all_test = join_weather_airline(airline_test, weather_test, airport_weather_index)
    airlines_weather_all_train = join_weather_airline(airline_train, weather_train, airport_weather_index)

    print("airlines_weather_all_sample", airlines_weather_all_sample.count())
    print("airlines_weather_all_test", airlines_weather_all_test.count())
    print("airlines_weather_all_train", airlines_weather_all_train.count())
    
    return airlines_weather_all_sample, airlines_weather_all_test, airlines_weather_all_train

# COMMAND ----------

# added new variable for airport and carrier avg(security delay), avg(aircraft delay), avg(carrier delay)
def best_model(usefullset = True):
  """
  run the models with all the parameters that are selected.
  """
  sample_all, full_test_all, full_train_all = preload_data_v3()
    
  model_params = ['DEP_DEL15', 'ORG_ELEVATION', 'ORG_WND_ANGLE', 'ORG_WND_SPEED', 'ORG_CIG_HEIGHT', 'ORG_VIS_DIST', 'ORG_TMP_F', 'ORG_DEW_1', 'ORG_SLP_1', 'DEST_ELEVATION', 'DEST_WND_ANGLE', 'DEST_WND_SPEED', 'DEST_CIG_HEIGHT', 'DEST_VIS_DIST', 'DEST_TMP_F', 'DEST_DEW_1', 'DEST_SLP_1', 'DAY_OF_WEEK', 'DEP_TIME', 'MONTH', 'INBOUND_ARR_DELAY_NEW', "MED_LATE_AIRCRAFT_DELAY", "MED_CARRIER_DELAY", "MED_SECURITY_DELAY"]

  run_models(model_params, sample_all, full_test_all, full_train_all, usefullset)
  
best_model()  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Algorithm Exploration and Experiment
# MAGIC 
# MAGIC In this section, we investigate the three ML algorithms we have explored for this project. In particular, we review their underlying theories and trade-offs and justify why we have chosen a certain set of features for our final model. The three models of our interest are listed below.  
# MAGIC - Logistic Regression (LR)
# MAGIC - Random Forest (RF)
# MAGIC - Gradient Boosted Trees (GBT)
# MAGIC 
# MAGIC We trained each model with the sample data that includes flight information **from January 2015 to June 2015**. Each model is compared against one another using AUROC and AUPRC metrics to identify the most suitable model for this project. We have also investigated the trained models' attributes to reveal and confirm the underlying relationships between our select features and the flight departure delay.  The following is a summary of what we learned through this experiment:
# MAGIC 
# MAGIC - Arrival delay for the inbound flight that is used for an outbound flight affects the flight's departure delay most dominantly
# MAGIC - Arrival delays of incoming flights at an airport critically affect departure delays for the airport's outbound flights (Airport-wise congestion)
# MAGIC - Airport security delays and carrier-wise delays also significantly affect the flight departure delay

# COMMAND ----------

# load the sample/test/train data
sample_all, full_test_all, full_train_all = preload_data_v3()   

# COMMAND ----------

# Select the feature columns and create train & test DataFrame

model_6_params = ['DEP_DEL15', 'MONTH', 'DAY_OF_WEEK', 'DEP_TIME', 'ORG_WND_ANGLE', 'ORG_WND_SPEED', 'ORG_CIG_HEIGHT', 'ORG_VIS_DIST', 'ORG_TMP_F', 'DEST_WND_ANGLE', 'DEST_WND_SPEED', 'DEST_CIG_HEIGHT', 'DEST_VIS_DIST', 'DEST_TMP_F', "MED_LATE_AIRCRAFT_DELAY", "MED_CARRIER_DELAY", "MED_SECURITY_DELAY","ARR_DELAY_NEW" ]#, 'INBOUND_ARR_DELAY_NEW']

params= model_6_params
train = None
test = None
use_full_data = False

sample, full_test, full_train = filter_datasets(sample_all, full_test_all, full_train_all, params)

if use_full_data:
  train, test = label_datasets(params[0], full_train, full_test)
else:
  train, test = label_datasets(params[0], sample)


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Logistic Regression
# MAGIC 
# MAGIC Logistic regression is a widely used classification algorithm. For binary classification problems, the algorithm outputs the probability of the outcome being positive (i.e. in the context of this problem, whether the flight will be delayed). Specifically, the logistic regression model uses the logistic function to calculate the probability 
# MAGIC $$ P(z) = \frac{1}{1+e^{z}} $$ where $$z = w^{T}x $$ and the \\( w \\) is calcuated based on the following loss function.
# MAGIC 
# MAGIC $$ L(w;x,y):=log(1 + exp(-yw^Tx))) $$
# MAGIC 
# MAGIC **Model Flexibility**
# MAGIC 
# MAGIC Because of the underlying linear relationship, \\( z = w^{T}x \\), logistic regression is suitable when the relationship between features and the response is relatively simple.  As such, it might not be useful in representing a complicated relationship. Although introducing high-order polynomial terms might help model more complex relationships, such terms result in a loss of explainability because coefficients for higher-order terms are harder to interpret. When modeling complicated relationships, more flexible models, such as tree-based models or support vector machines, are preferred. 
# MAGIC 
# MAGIC 
# MAGIC **Interpretability**
# MAGIC 
# MAGIC The logistic regression allows for interpreting the relationship between features and the response by investigating the coefficients for individual variables. For instance, a positive coefficient for a feature means a positive correlation between the feature and the response, while a negative coefficient indicates a negative correlation.     
# MAGIC 
# MAGIC **Feature Processing**
# MAGIC 
# MAGIC Logistic regression requires the standardization of input features for the trained model to perform well. This is handled automatically by Spark's implementation of the algorithm if the standardization parameter is set to True. 
# MAGIC 
# MAGIC 
# MAGIC **Hyperparameters of Interest**
# MAGIC 
# MAGIC - elasticNetParam
# MAGIC - threshold
# MAGIC - standardization
# MAGIC 
# MAGIC `elasticNetParam`: This hyperparameter determines the regularization degree between L2 and L1. For elasticNetParam = 0,an L2 penalty is applied. For elasticNetParam = 1, it is an L1 penalty. L1 regularization in particular pushes the coefficients of less important features to zero during training, while L2 shrinks coefficients without eliminating them outright like L1.
# MAGIC 
# MAGIC `threshold`: This hyperparameter determines the threshold probability to decide whether the response is positive or negative. The value can range from 0.0 to 1.0, and the default is 0.5. We can adjust precision and recall according to the requirements of the application by varying the threshold.
# MAGIC 
# MAGIC `standardization`: This hyperparameter enables automatic standardization of input features, as mentioned above. 

# COMMAND ----------

#Train a Logistic Regression model 
label = params[0] 
lrModel = LogisticRegression(featuresCol = 'features', labelCol = label, maxIter=10).fit(train)
predictions = lrModel.transform(test)

#Extract summary of the trained model 
trainingSummary = lrModel.summary

# COMMAND ----------

#evaluation
predictions = lrModel.transform(test)
evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC").setLabelCol(label)
print('Test Area Under ROC', evaluator.evaluate(predictions))
evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderPR").setLabelCol(label)
print('Test Area Under PR', evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Random Forest
# MAGIC 
# MAGIC Random forests are ensembles of decision trees and popularly used for classification and regression. Random forests resolve the well-known overfitting issue of decision trees by combining many simple uncorrelated decision trees to reduce the prediction variance. Random forests limit each tree to use a randomly-chosen subset of variables (features) to ensure low correlations between different decision trees. Each tree is trained based on a dataset bootstrapped from the original training set.
# MAGIC 
# MAGIC 
# MAGIC **Model Flexibility**
# MAGIC 
# MAGIC Like decision trees, random forests can capture non-linear relationships between the response and features. Because of this flexibility, random forests generally work better for complicated classification problems than logistic regression does. 
# MAGIC 
# MAGIC 
# MAGIC **Interpretability**
# MAGIC 
# MAGIC Random Forests provide feature importance measures, which allow us to compare individual features' influences for prediction. Feature importance measures are calculated as follows. 
# MAGIC - For each variable (feature), identify all the splitting points on the variable over all the trees.
# MAGIC - For all identified splitting points for a variable, calculate the split's improvements and averaging them over all the trees. The improvement is defined as information gain based on entropy or Gini index. 
# MAGIC - The averaged number over all trees is the variable importance for the feature. 
# MAGIC 
# MAGIC **Feature Processing**
# MAGIC 
# MAGIC Random forests handle categorical features without one-hot encoding, do not require feature scaling for non-categorical features, and can capture non-linearities and feature interactions without extra feature engineering.
# MAGIC 
# MAGIC **Hyperparameters of Interest**
# MAGIC 
# MAGIC Depending on how we configure the following hyperparameters, the training-time and bias/variance of the trained model can change. 
# MAGIC 
# MAGIC - numTrees 
# MAGIC - maxDepth
# MAGIC - featureSubsetStrategy
# MAGIC 
# MAGIC `numTrees`: This hyperparameter determines how many decision trees are trained using the bootstrapping process. The more trees a Random Forest uses, the lower variance we get. However, a large number of trees require more computing resources and training time. 
# MAGIC 
# MAGIC `maxDepth`: This hyperparameter determines the maximum depth of each tree. If the parameter is too small, the trained random forest may suffer from a high bias. On the other, if the parameter is too large, the trained random forest may have a high variance, overfitting the training data. A high value also requires more computing resources and training time.  
# MAGIC 
# MAGIC `featureSubsetStrategy`: This hyperparameter adjusts the number of features used to train each tree. Lower numbers for the parameter may cause a high bias, while numbers close to the total number of features may cause a high variance. A high value also requires more computing resources and training time. Apache Spark currently supports ['all', 'onethird', 'sqrt', 'log2'] for this parameter. 

# COMMAND ----------

rfModel = RandomForestClassifier(featuresCol = 'features', labelCol = label).fit(train)
predictions = rfModel.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Analysis of Feature Importance 
# MAGIC The following graph is the feature importance graph that shows the relative importance of individual features in predicting departure delay. As shown below, all the features that we have chosen contribute to predicting departure delay. Specifically, the following four features are the most influential.
# MAGIC - INBOUND_ARR_DELAY_NEW (ARR_DELAY_NEW)
# MAGIC - MED_LATE_AIRCRAFT_DELAY 
# MAGIC - MED_SECURITY_DELAY 
# MAGIC - MED_CARRIER_DELAY
# MAGIC 
# MAGIC MED_LATE_AIRCRAFT_DELAY, MED_SECURITY_DELAY, and MED_CARRIER_DELAY are estimates of a daily flight delay, security delay, and carrier delay at a particular airport. We have estimated these delays by taking a median of corresponding delays during the six months, and these estimates turn out to be quite accurate for the six months as the prediction performance proves.However, accurately estimating a daily flight delay, security delay, and carrier delay in 2019 using the training set, which spands from 2015 to 2018, could be challenging. Such inaccurate estimates for those delays can cause our model to perform worse in predicting departure delays in 2019. 

# COMMAND ----------

# Draw a feature importance graph
plt.figure(figsize=(10,5))
plt.bar(params[1:], rfModel.featureImportances.toArray())
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC").setLabelCol(label)
print('Test Area Under ROC', evaluator.evaluate(predictions))

evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderPR").setLabelCol(label)
print('Test Area Under PR', evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Gradient Boosted Trees
# MAGIC 
# MAGIC Boosting is a general method that can be applied to many machine learning models. Here, we have chosen Gradient Boosted Trees (GBT) to model the flight departure delay. GBTs train a sequence of decision trees iteratively. On each iteration, the algorithm uses the current ensemble to predict the label of each training instance and then compares the prediction with the true label. The dataset is relabeled to put more emphasis on training instances with poor predictions. Thus, in the next iteration, the decision tree effectively learns from its previous mistakes.
# MAGIC 
# MAGIC Unlike Random Forests, GBTs do not involve bootstrap sampling but train the next tree on the errors that resulted from the previous trees. Individual trees are usually simple with one or two splits, but typically, a boosted tree consists of thousands of trees. Boosting performs better than random forests due to its lower prediction variance.
# MAGIC 
# MAGIC **Model Flexibility**
# MAGIC 
# MAGIC Gradient Boosted Trees can capture non-linear relationships between the response and features. Because of this flexibility, GBTs generally work better for complicated classification problems than logistic regression does. Furthermore, the ensemble of trees reduces the variance of prediction, resulting in better accuracy with unseen data.  
# MAGIC 
# MAGIC **Interpretability**
# MAGIC 
# MAGIC Unlike the Logistic Regression and Random Forests, GBTs do not provide feasible ways to explain the association between features and response, although GBTs would predict better than them. Fortunately, accurate prediction is the key to the success of this project, and identifying potential causalities between features and response does not matter much in this context. Therefore, we will use this model as our main predictor as long as it performs better than the other algorithms.     
# MAGIC 
# MAGIC 
# MAGIC **Feature Processing**
# MAGIC 
# MAGIC GBTs handle categorical features without one-hot encoding, do not require feature scaling for non-categorical features, and can capture non-linearities and feature interactions without extra feature engineering.
# MAGIC 
# MAGIC **Hyperparameters of Interest**
# MAGIC 
# MAGIC The following hyperparameters determine the structure of the trained GBT model.
# MAGIC - numIterations 
# MAGIC - maxDepth
# MAGIC - supportedFeatureSubsetStrategies
# MAGIC 
# MAGIC `numIterations`: This hyperparameter determines the number of small boosting trees that build a boosted tree. Typically, thousands of simple trees are used.  
# MAGIC 
# MAGIC `maxDepth`: This hyperprameter determines the maximum depth of each tree. GBTs typically use simple shallow trees with one or two splits but use thousands of trees to form a boosted tree. A high value for this hyperparamter requires more computing resources and training time.
# MAGIC 
# MAGIC `supportedFeatureSubsetStrategies`: This hyperparameter plays the same role as it does in the Random Forests algorithm. 

# COMMAND ----------

gbt = GBTClassifier(maxIter=10, featuresCol = 'features', labelCol = label)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC").setLabelCol(label)
print('Test Area Under ROC', evaluator.evaluate(predictions))

evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderPR").setLabelCol(label)
print('Test Area Under PR', evaluator.evaluate(predictions))


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Application of Course Concepts:
# MAGIC 
# MAGIC  Throughout the course of this project, we made an effort to keep track of the course concepts relayed to us throughout the semester, and to apply them to our work wherever possible.  While not all course concepts were relevant to the work, the following concepts were considered incorporated into our development process:
# MAGIC 
# MAGIC - Model complexity/regularization: As mentioned above, we had a choice of two outcome variables, both of which would have satisfied the parameters of the final project.  While going with quantifying the length of the predicted delay was tempting, it was out of consideration of reducing model complexity that we chose to go with a simple binary prediction of whether or not any given flight would be delayed, rather than attempting to bucket it into the length of delay fields. As for regularization terms to penalize model complexity, this was the driving motivation behind considering the proper `elasticNetParam` hyperparameter for logistic regression models in particular.
# MAGIC 
# MAGIC - Scalability / time complexity / I/O vs Memory: Given the significant quantity of data that we had to process, ensuring scaleability on the part of the process was a major consideration.  In order to do this, we made use of index tables in order to reduce the time complexity of the join process between weather and airport data.  In addition, we explicitly made sure that Parquet was in use over other data formats such as Avro or CSV, owing to its column-based data format, which provided an invaluable efficiency boost in the model training stage.  Finally, the decision to drop all but the relevant columns was motivated by a desire to reduce the memory footprint of the input files.
# MAGIC 
# MAGIC - Broadcasting / caching / DAGs / lazy evaluation: Pursuant to the objective of scalability and reducing time complexity as described above, we made use of Spark’s optimization capabilities over the course of the project.  In particular, we made use of PySpark’s Dataframes, which brought with it automatic caching and lazy evaluation capabilities.  Finally, the weather index was broadcasted as part of the join process..

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Future Work and Lesson Learned  
# MAGIC 
# MAGIC - cross-validation and grid search
# MAGIC - Takes long time to run anything (5~15 mins) depending on how busy the cluster is
# MAGIC - What worked: Incorporating prior arrival delay into models
# MAGIC - what didn't work: Joining by time stamps, geo-joining with Geospark
# MAGIC - future work: Explore expansion of flight delay predictions to include length of delay
# MAGIC 
# MAGIC We keep only baseline and best model to reduce the contents. Please see [this notebook](https://dbc-9bfca75b-725c.cloud.databricks.com/?o=52497824689954#notebook/2542098129672994/command/2542098129682577) for all the models. 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Reference
# MAGIC 
# MAGIC 1. [Airline dataset field description](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236)
# MAGIC - [Understanding the Reporting of Causes of Flight Delays and Cancellations](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)
# MAGIC - [World Airport Codes](https://www.world-airport-codes.com/)
# MAGIC - [Weather Station List](http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html)
# MAGIC - [Weather isd-format-document](https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf)
# MAGIC - [spark MLlib user guide](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/classification.html)
# MAGIC - Chakrabarty, N. (2019) A Data Mining Approach to Flight Arrival Delay Prediction for American Airlines.  Retrieved July 20, 2020 from https://arxiv.org/ftp/arxiv/papers/1903/1903.06740.pdf .
# MAGIC - Gopalakrishnan, K. and amsa Balakrishnan (2017).  A Comparative Analysis of Models forPredicting Delays in Air Traffic Networks.  Retrieved July 20, 2020 from http://www.mit.edu/~hamsa/pubs/GopalakrishnanBalakrishnanATM2017.pdf .
# MAGIC - Henriques, R., & Feiteira, I. (2018). Predictive Modelling: Flight Delays and Associated Factors, Hartsfield–Jackson Atlanta International Airport. Procedia Computer Science, 138, 638-645. doi:10.1016/j.procs.2018.10.085
# MAGIC - Kuhn, Nathalie and Navaneeth Jamadagni (2017).  Application of Machine Learning Algorithms toPredict Flight Arrival Delays.  Retrieved July 21, 2020 from http://cs229.stanford.edu/proj2017/final-reports/5243248.pdf .
# MAGIC - Shao, W., Prabowo, A., Zhao, S., Tan, S., Koniusz, P., Chan, J., Salim, F. D. (2019). Flight Delay Prediction using Airport Situational Awareness Map. Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems. doi:10.1145/3347146.3359079
# MAGIC - Yanying, Y., Mo, H., & Haifeng, L. (2019). A Classification Prediction Analysis of Flight Cancellation Based on Spark. Procedia Computer Science, 162, 480-486. doi:10.1016/j.procs.2019.12.014
# MAGIC - Ye, B., Liu, B., Tian, Y., & Wan, L. (2020). A Methodology for Predicting Aggregate Flight Departure Delays in Airports Based on Supervised Learning. Sustainability, 12(7), 2749. doi:10.3390/su12072749

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Model Package / Data Pipeline
# MAGIC To run the best model in one cell, without loading any cell above, run the cell below.  

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

from pyspark.sql import types
import pyspark.sql.functions as F

from sklearn import metrics

#MLlib
from pyspark.ml.feature import RFormula 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from pyspark.ml.feature import VectorAssembler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import time

sqlContext = SQLContext(sc)

DATA_PATH = 'dbfs:/mnt/mids-w261/data/datasets_final_project/'
MASTER_CORD_PATH = '/FileStore/tables/407131831_T_MASTER_CORD.csv'
WEATHER_COLUMNS = ['STATION', 'ELEVATION', 'WND_ANGLE', 'WND_SPEED', 'CIG_HEIGHT', 'VIS_DIST', 'TMP_F', 'DEW_1', 'SLP_1', 'LATITUDE', 'LONGITUDE', 'DATE']
AIRLINE_COLUMNS_TOKEEP = ['DEP_DEL15', 'WEATHER_DELAY', 'NAS_DELAY',  'CARRIER_DELAY', 'OP_CARRIER', 'SECURITY_DELAY', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'LATE_AIRCRAFT_DELAY', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_TIME', 'FL_DATE', "DEP_HOUR", "DAY_OF_WEEK", "DAY_OF_MONTH", 'TAIL_NUM', 'CRS_ARR_TIME', 'ARR_DELAY_NEW', 'MONTH', 'YEAR']

AIRLINE_INBOUND_COLUMNS_TOKEEP = ['FL_DATE', 'DEST_AIRPORT_ID', 'TAIL_NUM', 'CRS_ARR_TIME', 'ARR_DELAY_NEW']
AIRLINE_OUTBOUND_COLUMNS_TOKEEP = ['DEP_DEL15', 'WEATHER_DELAY', 'NAS_DELAY',  'CARRIER_DELAY', 'OP_CARRIER', 'SECURITY_DELAY', 'ORIGIN', 'ORIGIN_AIRPORT_ID', 'LATE_AIRCRAFT_DELAY', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_TIME', 'FL_DATE', "DEP_HOUR", "DAY_OF_WEEK", "DAY_OF_MONTH", 'TAIL_NUM', "MONTH", 'YEAR']

# update the following parameeters for predictions
airline_parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2019.parquet"] 
current_weather_data_parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2019a.parquet"]

def load_airline_data(use_small_set_only = False):
    """
    load ariline data and transform 
    """

    def transform_airline_data(parquet_path):
        """
        read from s3 parquets
        transform given airline dataset to
        - for delay fields, replace null value by zero
        """
        
        airlines = spark.read.option("header", "true").parquet(*parquet_path)
          
        # Seperate DEP_TIME into Hour for join (We might want to use TimeStamp values with date to join for 2 hours before)
        airlines = airlines.withColumn("DEP_HOUR", F.bround('DEP_TIME', -2)/100)

        # drop cancelled flight for baseline 
        airlines = airlines.where(F.col('cancelled') == 0).select(AIRLINE_COLUMNS_TOKEEP)

        # replace null value by zeros
        airlines = airlines.withColumn('NAS_DELAY', F.when(airlines['NAS_DELAY'].isNull(), 0).otherwise(airlines['NAS_DELAY']) )
        airlines = airlines.withColumn('WEATHER_DELAY', F.when(airlines['WEATHER_DELAY'].isNull(), 0).otherwise(airlines['WEATHER_DELAY']) )
        airlines = airlines.withColumn('SECURITY_DELAY', F.when(airlines['SECURITY_DELAY'].isNull(), 0).otherwise(airlines['SECURITY_DELAY']) )
        airlines = airlines.withColumn('LATE_AIRCRAFT_DELAY', F.when(airlines['LATE_AIRCRAFT_DELAY'].isNull(), 0).otherwise(airlines['LATE_AIRCRAFT_DELAY']) )
        airlines = airlines.withColumn('CARRIER_DELAY', F.when(airlines['CARRIER_DELAY'].isNull(), 0).otherwise(airlines['CARRIER_DELAY']) )
        airlines = airlines.withColumn('ARR_DELAY_NEW', F.when(airlines['ARR_DELAY_NEW'].isNull(), 0).otherwise(airlines['ARR_DELAY_NEW']) )

        return airlines
  
    airline_train, airline_test = None, None
    if not use_small_set_only:
        # load airlines data
        parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2015.parquet", 
                        "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2016.parquet",
                        "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2017.parquet",
                        "dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2018.parquet"]
        airline_train = transform_airline_data(parquet_path)

#         airline_parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/2019.parquet"] 
        airline_test = transform_airline_data(airline_parquet_path)
    
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data_6m/part-*.parquet"] 
    airline_6m = transform_airline_data(parquet_path)
      
    return airline_train, airline_test, airline_6m
  
def load_weather_data(use_small_set_only = False):
    """
    load weather data and transform
    """
    # keep only interesting variables
    
    def transform_weather_data(weather):
        """
        string => integer
        WND_ANGLE => 0 if missing
        WND_SPEED => 0 if missing
        TMP => 15 (standard temperature) if missing 
        DEW => 16 (standard dew point) if missing
        SLP => 10132 (standard see level pressure) if missing
        """
        split_weather_field = F.split(weather['WND'], ',')
        weather = weather.withColumn('WND_ANGLE', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('WND_ANGLE', F.when(weather['WND_ANGLE'] == 999, 0).otherwise(weather['WND_ANGLE']) )

        weather = weather.withColumn('WND_SPEED', split_weather_field.getItem(3).cast("double"))
        weather = weather.withColumn('WND_SPEED', F.when(weather['WND_SPEED'] == 9999, 0).otherwise(weather['WND_SPEED']) )

        split_weather_field = F.split(weather['CIG'], ',')
        weather = weather.withColumn('CIG_HEIGHT', split_weather_field.getItem(0).cast("double"))

        split_weather_field = F.split(weather['VIS'], ',')
        weather = weather.withColumn('VIS_DIST', split_weather_field.getItem(0).cast("double"))

        split_weather_field = F.split(weather['TMP'], ',')
        weather = weather.withColumn('TMP_F', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('TMP_F', F.when(weather['TMP_F'] == 9999, 15).otherwise(weather['TMP_F']) )

        split_weather_field = F.split(weather['DEW'], ',')
        weather = weather.withColumn('DEW_1', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('DEW_1', F.when(weather['DEW_1'] == 9999, 15).otherwise(weather['DEW_1']) )

        split_weather_field = F.split(weather['SLP'], ',')
        weather = weather.withColumn('SLP_1', split_weather_field.getItem(0).cast("double"))
        weather = weather.withColumn('SLP_1', F.when(weather['SLP_1'] == 99999, 10132).otherwise(weather['SLP_1']) )

        weather = weather.select(WEATHER_COLUMNS)

        # Seperate DATE Columns to DATE and Hour for airline join
        weather = weather.withColumn("DATE_PART", F.to_date(F.col("DATE")))\
                                 .withColumn("HOUR_PART", F.hour(F.col("DATE"))) 
        
        return weather 
  
    # Filter weather dataset with only 2015 first 6 months.
    parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2015a.parquet"]
    weather_sample = spark.read.option("header", "true").parquet(*parquet_path)
    weather_sample = weather_sample.filter(F.col('DATE')<='2015-07-01')
    weather_sample = transform_weather_data(weather_sample)
    
    weather_train, weather_test = None, None
    if not use_small_set_only:
        parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2015a.parquet", 
                       "dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2016a.parquet",
                       "dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2017a.parquet",
                       "dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2018a.parquet"]
        weather_train = spark.read.option("header", "true").parquet(*parquet_path)
        weather_train = transform_weather_data(weather_train)

#         parquet_path = ["dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/weather2019a.parquet"]
        weather_test = spark.read.option("header", "true").parquet(*current_weather_data_parquet_path)
        weather_test = transform_weather_data(weather_test)
    
    return weather_train, weather_test, weather_sample

def create_index_tables(weather):
  """
  create index table 
  todo: push csv file to s3
  """
  
  def create_weather_index(airport_idx, weather):
      """
      helper function to create weather index
      """
      # Create Index weather table for only Station ID, LATITUDE, LONGITUDE
      weather.createOrReplaceTempView('weather_tmp_view')
      weather_index = spark.sql(
        """
        SELECT
          DISTINCT STATION, LATITUDE, LONGITUDE
        FROM
          weather_tmp_view """
      ).toPandas()

      # Remove NAs
      weather_index = weather_index.dropna()
      
      # Calculate by euclidean distance
      weather_station_idx = metrics.pairwise_distances_argmin_min(airport_index[['LATITUDE', 'LONGITUDE']], weather_index[['LATITUDE', 'LONGITUDE']], metric='euclidean')[0]
      airport_idx['weather_station_id'] = [weather_index.iloc[i]['STATION'] for i in weather_station_idx]

      # Rename it to be airport_weather_index
      airport_weather_index = spark.createDataFrame(airport_index)
      airport_weather_index.createOrReplaceTempView('airport_weather_index')
  
      return airport_weather_index
  
  airport_geocoord_columns_tokeep = ['AIRPORT_ID', 'DISPLAY_AIRPORT_NAME', 'LATITUDE', 'LONGITUDE']
  # import lat/lon
  # 193497807_T_MASTER_CORD.csv 
  airport_geocoord = spark.read.csv(MASTER_CORD_PATH, header="true", inferSchema="true")

  airport_geocoord = airport_geocoord.select(airport_geocoord_columns_tokeep)

  # Filter it with only US airports (But destination airport might be international)
  airport_geocoord = airport_geocoord.groupBy('AIRPORT_ID').agg(F.avg('LATITUDE'), F.avg('LONGITUDE'))

  # Create Index airport table
  airport_index = airport_geocoord.withColumnRenamed("avg(LATITUDE)","LATITUDE").withColumnRenamed("avg(LONGITUDE)","LONGITUDE").toPandas()
  airport_index = airport_index.dropna()
  airport_weather_index = create_weather_index(airport_index, weather)

  return airport_weather_index

def join_weather_airline(airlines, weather, airport_weather_index):
    """
    join airline and weather data using index table
    """
  
    # Join the weather station id to origin airport 
    airlines_weather_all = airlines.join(F.broadcast(airport_weather_index), airport_weather_index.AIRPORT_ID==airlines.ORIGIN_AIRPORT_ID, 'inner')
      # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("LATITUDE","ORIGIN_LAT").withColumnRenamed("LONGITUDE","ORIGIN_LON")\
                      .withColumnRenamed("AIRPORT_ID","AIRPORT_ORG").withColumnRenamed("weather_station_id","ORG_WS_ID")  

    # Join the weather station id to Destination airport 
    airlines_weather_all = airlines_weather_all.join(F.broadcast(airport_weather_index), airport_weather_index.AIRPORT_ID==airlines_weather_all.DEST_AIRPORT_ID, 'inner')
      # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("LATITUDE","DEST_LAT").withColumnRenamed("LONGITUDE","DEST_LON")\
                      .withColumnRenamed("AIRPORT_ID","AIRPORT_DEST").withColumnRenamed("weather_station_id","DEST_WS_ID")

    # weather to airline join (Origin Airport)
    airlines_weather_all = airlines_weather_all.join( weather, [airlines_weather_all.ORG_WS_ID == weather.STATION, airlines_weather_all.FL_DATE == weather.DATE_PART, airlines_weather_all.DEP_HOUR == weather.HOUR_PART], 'left')

    # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("STATION","ORG_WS_ID_2").withColumnRenamed("ELEVATION","ORG_ELEVATION")\
                      .withColumnRenamed("WND_ANGLE","ORG_WND_ANGLE").withColumnRenamed("WND_SPEED","ORG_WND_SPEED")\
                      .withColumnRenamed("CIG_HEIGHT","ORG_CIG_HEIGHT").withColumnRenamed("VIS_DIST","ORG_VIS_DIST")\
                      .withColumnRenamed("TMP_F","ORG_TMP_F").withColumnRenamed("DEW_1","ORG_DEW_1").withColumnRenamed("SLP_1","ORG_SLP_1")\
                      .withColumnRenamed("LATITUDE","ORG_LATITUDE").withColumnRenamed("LONGITUDE","ORG_LONGITUDE")\
                      .withColumnRenamed("DATE","ORG_DATE").withColumnRenamed("DATE_PART","ORG_DATE_PART").withColumnRenamed("HOUR_PART","ORG_HOUR_PART")

    # weather to airline join (Destination Airport)
    airlines_weather_all = airlines_weather_all.join( weather, [airlines_weather_all.DEST_WS_ID == weather.STATION, airlines_weather_all.FL_DATE == weather.DATE_PART, airlines_weather_all.DEP_HOUR == weather.HOUR_PART], 'left')

    # Rename
    airlines_weather_all = airlines_weather_all.withColumnRenamed("STATION","DEST_WS_ID_2").withColumnRenamed("ELEVATION","DEST_ELEVATION")\
                      .withColumnRenamed("WND_ANGLE","DEST_WND_ANGLE").withColumnRenamed("WND_SPEED","DEST_WND_SPEED")\
                      .withColumnRenamed("CIG_HEIGHT","DEST_CIG_HEIGHT").withColumnRenamed("VIS_DIST","DEST_VIS_DIST")\
                      .withColumnRenamed("TMP_F","DEST_TMP_F").withColumnRenamed("DEW_1","DEST_DEW_1").withColumnRenamed("SLP_1","DEST_SLP_1")\
                      .withColumnRenamed("LATITUDE","DEST_LATITUDE").withColumnRenamed("LONGITUDE","DEST_LONGITUDE")\
                      .withColumnRenamed("DATE","DEST_DATE").withColumnRenamed("DATE_PART","DEST_DATE_PART").withColumnRenamed("HOUR_PART","DEST_HOUR_PART")
    
    return airlines_weather_all
  
def run_GBT(label, train, test, max_iter = 10):
    """
    train the model with GBT and predict on test set
    evaluate using ROC and PRC
    """
    start = time.time()
    print("--------------------------------")

    gbt = GBTClassifier(maxIter=30, featuresCol = 'features', labelCol = label)
    gbtModel = gbt.fit(train)
    predictions = gbtModel.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test accuracy= %g" % (accuracy))

    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName="f1")
    accuracy = evaluator.evaluate(predictions)
    print("f1 = %g" % (accuracy))

    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName="precisionByLabel")
    accuracy = evaluator.evaluate(predictions)
    print("precision = %g" % (accuracy))

    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName="recallByLabel")
    accuracy = evaluator.evaluate(predictions)
    print("recall = %g" % (accuracy))
    
    accuracy = evaluator.evaluate(predictions)
    print("recall = %g" % (accuracy))
    
    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName="falsePositiveRateByLabel")
    accuracy = evaluator.evaluate(predictions)
    print("falsePositive = %g" % (accuracy))
    
    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName="truePositiveRateByLabel")
    accuracy = evaluator.evaluate(predictions)
    print("truePositive = %g" % (accuracy))
    
    print("--------------------------------")    
    
def label_datasets(label, train, test = None):
    """
    - (todo) there should be no missing data at this moment but there is still
    - mark the dataset with label and features
    """
    def label_dataset(label, df):
        """
        help function to label single data set
        """
        df = df.na.drop()

        features = df.columns
        features.remove(label)

        # Set up a ML pipeline
        stages = []

        assembler = VectorAssembler(inputCols=features, outputCol="features")
        stages += [assembler]

        pipeline = Pipeline(stages = stages)
        pipelineModel = pipeline.fit(df)
        df = pipelineModel.transform(df)

        selectedCols = [label, 'features'] + features
        df = df.select(selectedCols)
        
        return df 
  
    # if using sample set
    if test is None:
        train, test = label_dataset(label, train).randomSplit([0.8, 0.2], seed = 42)
    else:
        train = label_dataset(label, train)
        test = label_dataset(label, test)

    return train, test
  
def filter_datasets(sample, full_test, full_train, variables_tokeep):
    """
    filter fields with only input variables and label variable to be used for modeling
    """
    sample = sample.select(variables_tokeep)
    full_test = full_test.select(variables_tokeep)
    full_train = full_train.select(variables_tokeep)
    
    return sample, full_test, full_train
  
def run_models(params, sample_all, full_test_all, full_train_all, use_full_data = True):
    """
    run 3 models using given dataset 
    """
    sample, full_test, full_train = filter_datasets(sample_all, full_test_all, full_train_all, params)    
    if use_full_data:
      train, test = label_datasets(params[0], full_train, full_test)
    else:
      train, test = label_datasets(params[0], sample)
    print("--------------------------------")
    print(params[1:])
    print("--------------------------------")
    run_GBT(params[0], train, test, 30)
    print()

def create_carrier_security_delay_index_avg(airline_train):
    """
    calculate average value of delays grouped by carrier and airport
    """
    carrier_security_delay_index = airline_train.groupBy("OP_CARRIER", "ORIGIN_AIRPORT_ID").agg(F.mean('LATE_AIRCRAFT_DELAY').alias("MED_LATE_AIRCRAFT_DELAY"), 
                                                                                                F.mean('CARRIER_DELAY').alias("MED_CARRIER_DELAY"), 
                                                                                                F.mean('SECURITY_DELAY').alias("MED_SECURITY_DELAY") )
    return carrier_security_delay_index

def create_carrier_security_delay_index_median(airline_train):
    """
    calculate median value of delays grouped by carrier and airport
    """
#     carrier_security_delay_index = airline_train.groupBy("OP_CARRIER", "ORIGIN_AIRPORT_ID").agg(F.mean('LATE_AIRCRAFT_DELAY'), F.mean('CARRIER_DELAY'), F.mean('SECURITY_DELAY'))
    airline_train.registerTempTable('df')
    carrier_security_delay_index = spark.sql(
      """
      SELECT
        OP_CARRIER, ORIGIN_AIRPORT_ID, percentile_approx(LATE_AIRCRAFT_DELAY,0.5) as MED_LATE_AIRCRAFT_DELAY,\
        percentile_approx(CARRIER_DELAY,0.5) as MED_CARRIER_DELAY,\
        percentile_approx(SECURITY_DELAY,0.5) as MED_SECURITY_DELAY
      FROM
        df
      GROUP BY
        OP_CARRIER, ORIGIN_AIRPORT_ID""")
    
    return carrier_security_delay_index
  
def join_delays_groupby_airport_carrier(carrier_security_delay_index, airline):
    """
    join delay by carrier and airport ID 
    """
    airlines_with_delays = airline.join(F.broadcast(carrier_security_delay_index), 
                                         [ "OP_CARRIER", "ORIGIN_AIRPORT_ID"], 'inner')

    return airlines_with_delays
  
def join_arrival_data(airline):
  airline_outbound = airline.select(AIRLINE_OUTBOUND_COLUMNS_TOKEEP)
  airline_inbound = airline.select(AIRLINE_INBOUND_COLUMNS_TOKEEP)\
                    .withColumnRenamed("DEST_AIRPORT_ID","DEST_AIRPORT_ID_IN")\
                    .withColumnRenamed("TAIL_NUM","TAIL_NUM_IN").withColumnRenamed("FL_DATE","FL_DATE_IN")\
                    .withColumnRenamed("ARR_DELAY_NEW","INBOUND_ARR_DELAY_NEW")
  airline_joined = airline_outbound.alias("outbound").join(airline_inbound.alias("inbound"), 
                    [F.col("outbound.ORIGIN_AIRPORT_ID") == F.col("inbound.DEST_AIRPORT_ID_IN")
                    , F.col("outbound.TAIL_NUM") == F.col("inbound.TAIL_NUM_IN")
                    , F.col("outbound.FL_DATE") == F.col("inbound.FL_DATE_IN")
                    , (F.col("outbound.DEP_TIME") - 200) > F.col("inbound.CRS_ARR_TIME")
                    , (F.col("outbound.DEP_TIME") - 400) < F.col("inbound.CRS_ARR_TIME")], "left")
  
  print("before join", airline.count())
  print("after left join", airline_joined.count())

  return airline_joined

def preload_data_v3():
    """
    load data and transform to be fed into the models 
    - load airline data 
    - add inbound arrival delay
    - associate delay info with airport and carrier
    - load weather data 
    - join airline and weather data  
    output: sample, test, and train
    """
    # 1. load airline data 
    airline_train, airline_test, airline_sample = load_airline_data()
    
    # 2. add inbound arrival delay 
    airline_train = join_arrival_data(airline_train)
    airline_test = join_arrival_data(airline_test)
    airline_sample = join_arrival_data(airline_sample)
    
    # 3. filter previous year delay
    airline_train_2018 = airline_train.where( (F.col('YEAR') == 2018) )
      
    # 4. associate delay info with airport and carrier
    carrier_security_delay_index = create_carrier_security_delay_index_avg(airline_train_2018)
    
    airline_train = join_delays_groupby_airport_carrier(carrier_security_delay_index, airline_train)
    airline_test = join_delays_groupby_airport_carrier(carrier_security_delay_index, airline_test)
    airline_sample = join_delays_groupby_airport_carrier(carrier_security_delay_index, airline_sample)

    # 5. load weather data 
    weather_train, weather_test, weather_sample = load_weather_data()

    # 6. create index table 
    airport_weather_index = create_index_tables(weather_train)
    print("airport_weather_index", airport_weather_index.count())

    # 7. join airline and weather data using index table 
    airlines_weather_all_sample = join_weather_airline(airline_sample, weather_sample, airport_weather_index)
    airlines_weather_all_test = join_weather_airline(airline_test, weather_test, airport_weather_index)
    airlines_weather_all_train = join_weather_airline(airline_train, weather_train, airport_weather_index)

    print("airlines_weather_all_sample", airlines_weather_all_sample.count())
    print("airlines_weather_all_test", airlines_weather_all_test.count())
    print("airlines_weather_all_train", airlines_weather_all_train.count())
    
    return airlines_weather_all_sample, airlines_weather_all_test, airlines_weather_all_train
  
def best_model(usefullset = True):
  """
  run the models with all the parameters that are selected.
  """
  sample_all, full_test_all, full_train_all = preload_data_v3()
    
  model_params = ['DEP_DEL15', 'ORG_ELEVATION', 'ORG_WND_ANGLE', 'ORG_WND_SPEED', 'ORG_CIG_HEIGHT', 'ORG_VIS_DIST', 'ORG_TMP_F', 'ORG_DEW_1', 'ORG_SLP_1', 'DEST_ELEVATION', 'DEST_WND_ANGLE', 'DEST_WND_SPEED', 'DEST_CIG_HEIGHT', 'DEST_VIS_DIST', 'DEST_TMP_F', 'DEST_DEW_1', 'DEST_SLP_1', 'DAY_OF_WEEK', 'DEP_TIME', 'MONTH', 'INBOUND_ARR_DELAY_NEW', "MED_LATE_AIRCRAFT_DELAY", "MED_CARRIER_DELAY", "MED_SECURITY_DELAY"]

  run_models(model_params, sample_all, full_test_all, full_train_all, usefullset)
  
best_model()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC END OF NOTEBOOK
