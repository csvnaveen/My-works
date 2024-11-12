# Databricks notebook source
!pip install pandas-gbq google-cloud-storage google-cloud-bigquery
!pip install xlrd
!pip install openpyxl
!pip install gcsfs
!pip install gsutil

# COMMAND ----------

!pip install xgboost --version

# COMMAND ----------

from google.cloud import storage
from io import BytesIO
import gcsfs
import pandas as pd
import numpy as np
import glob
import openpyxl
import xlrd
import datetime
from datetime import datetime as dt, timedelta, date

import warnings
warnings.simplefilter(action='ignore')
import sys

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.sql import functions as F
from dateutil.relativedelta import relativedelta

# COMMAND ----------

spark = SparkSession.builder \
    .master("local") \
    .appName("escalation_detection") \
    .enableHiveSupport() \
    .getOrCreate()

# COMMAND ----------

##########################################################################################
# Performing manual Sentiment Analysis:  along with other features : create combine score
###########################################################################################
# today_date=pd.Timestamp("today").strftime("%m%d%Y")
today_date=pd.Timestamp("today").strftime("%Y%m%d")
pred_path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_predict/'+'clean_message_sub_data_predict_'+str(today_date)+'.csv'

#val_data = pd.read_csv(pred_path)
# val_data = pd.read_csv('gs://its-managed-dbx-ds-01-p-goas-scientists-workarea/escalation_detection_project/output_predict/clean_message_sub_data_predict_11242022.csv')


# COMMAND ----------

# bucket_name = 'its-managed-dbx-ds-01-p-goas-scientists-workarea'
# client = storage.Client()
# bucket = client.bucket(bucket_name)
# print(pd.datetime.now())

# COMMAND ----------

# blobs = bucket.list_blobs(prefix="escalation_detection_project/output_predict/")

# COMMAND ----------

# file = []
# for blob in blobs:
#     file.append(blob.name)

# COMMAND ----------

# file

# COMMAND ----------

# latest_file=file[-1]
# latest_file

# COMMAND ----------

import glob 
import os 

for file in glob.glob("/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_predict/*"):
    print(file)

# COMMAND ----------

file

# COMMAND ----------

val_data = pd.read_csv(file)
val_data

# COMMAND ----------

# fs = gcsfs.GCSFileSystem(project='its-managed-dbx-ds-01-p')

# COMMAND ----------

# with fs.open(file) as f:
#     val_data = pd.read_csv(f, keep_default_na=False)

# COMMAND ----------

val_data['Closed Date'].unique()

# COMMAND ----------

# filter Emails only from Customers :
val_data_cust = val_data[val_data['Agent_OR_Customer']=='Customer']
val_data_cust.shape, val_data.shape

# COMMAND ----------

####### Customer sentiment using List Comprehenisve

# # loading positve, negative and escalated words dictionary
negative_dict = spark.read.format("csv").option("inferSchema", "true").option("header", "false").option("sep", ",").option("multiLine", "true").option("quote","\"").load("/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/negative_words.txt").toDF("negative_words").toPandas().negative_words.tolist()
escalation_dict = spark.read.format("csv").option("inferSchema", "true").option("header", "false").option("sep", ",").option("multiLine", "true").option("quote","\"").load("/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/escalation_words.txt").toDF("escalation_words").toPandas().escalation_words.tolist()
positive_dict = spark.read.format("csv").option("inferSchema", "true").option("header", "false").option("sep", ",").option("multiLine", "true").option("quote","\"").load("/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/positive_words.txt").toDF("positive_words").toPandas().positive_words.tolist()



#===============================================================
#Removing list of positive words (greetings) from customer reply
#===============================================================
greeting_words = ['best','thank','great','like','kindly','patience','good','fine','appreciate','successfully','succeed',
                 'clearly','sincerely','rich','works']

positive_dict_update = [word for word in positive_dict if word not in greeting_words  ]

def Sentiment_words(text):
    try:
        global Sentiment,positive_words,negative_words,escalation_words,total_score,total_score_intesity
        raw_words = text.split(" ")
        
        #modified calculation approach to overcome N-gram sentiment
        positive_score = len([word for word in positive_dict_update  if word in raw_words])
        negative_score = len([word for word in negative_dict  if word in raw_words])
        escalation_score = len([word for word in escalation_dict  if word in raw_words])
        
#         positive_score = len([word for word in raw_words if word in positive_dict])
#         negative_score = len([word for word in raw_words if word in negative_dict])
#         escalation_score = len([word for word in raw_words if word in escalation_dict])

        # fetching positve and negative value
        positive_words = [word for word in  positive_dict_update if word in raw_words ]
        negative_words = [word for word in negative_dict  if word in raw_words]
        escalation_words = [word for word in escalation_dict if word in raw_words]


        total_score = -3* escalation_score + (positive_score - negative_score)
        total_score_intesity = 100*total_score/len(raw_words)
        
        words =[]
        if(total_score >0 ):
            Sentiment ='Positive'
            words.append(positive_words)
        elif((total_score < 0 ) & (escalation_score == 0)):
            Sentiment ='Negative'
            words.append(negative_words)
        elif(escalation_score > 0):
            Sentiment ='Escalation'
            words.append(escalation_words)    
        else : 
            Sentiment ='Neutral'
            
            return(Sentiment,positive_words,negative_words,escalation_words,total_score,total_score_intesity)

    except:
             pass

# COMMAND ----------

val_data_cust

# COMMAND ----------

# running sentiment 
############################

import time
startTime = time.time()

val_data['manual_sentiment'] =val_data.apply(lambda x: Sentiment_words(x['clean_message']),axis=1 )

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

# COMMAND ----------

val_data

# COMMAND ----------

temp=val_data
temp

# COMMAND ----------

temp['manual_sentiment']=temp['manual_sentiment'].fillna(0)
temp

# COMMAND ----------

temp1=temp.loc[temp['manual_sentiment'] != 0]
temp1

# COMMAND ----------

temp2=temp.loc[temp['manual_sentiment'] == 0]
temp2

# COMMAND ----------

temp1[['Sentiment','Positive_Words','Negative_Words','escalation_words','Total_Sentiment_Score','total_score_intesity']] = pd.DataFrame(temp1['manual_sentiment'].tolist(), index = temp1.index)
temp1

# COMMAND ----------

temp2[['Sentiment','Positive_Words','Negative_Words','escalation_words','Total_Sentiment_Score','total_score_intesity']] = 0
temp2

# COMMAND ----------

temp3=temp1.append(temp2)
temp3

# COMMAND ----------

val_data=temp1
val_data

# COMMAND ----------

#========================================
# count of emails per case from customer
#========================================

# without removing agent emails
cust_email_count= pd.DataFrame(val_data.groupby(['Case Number']).size().sort_values(ascending=False)).reset_index()
cust_email_count = cust_email_count.rename(columns= {'Case Number':'case_number', 0:'count_of_email'})

#considering both agent and customer email
val_data_2 = pd.merge(val_data,cust_email_count, left_on='Case Number',right_on='case_number',how='left')

# COMMAND ----------

####################################################
# Creating label using Case Age Duration
####################################################

def duration_weight(age):
    
    if (age < 10):
        label = 0
    elif (age >=10) & (age < 15):
        label = 50
    elif (age >=15) & (age < 20):
        label = 60
    elif (age >=20) & (age < 25):
        label = 70
    elif (age >=25) & (age < 30):
        label = 80
    elif (age >=30) & (age < 35):
        label = 90
    elif (age >=35):
        label = 100
        
    return(label)

# COMMAND ----------

####################################################
# Creating label using Case Age Duration
####################################################

def num_of_customer_emails_weight(age):
    
    if (age < 10):
        label = 0
    elif (age >=10) & (age < 15):
        label = 50
    elif (age >=15) & (age < 20):
        label = 60
    elif (age >=20) & (age < 25):
        label = 70
    elif (age >=25) & (age < 30):
        label = 80
    elif (age >=30) & (age < 35):
        label = 90
    elif (age >=35):
        label = 100
        
    return(label)

# COMMAND ----------

#======================================================
# Implementing weightage based on duration of the case
#======================================================

val_data_2['duration_weight'] =val_data_2['Case_Duration_days'].apply(lambda x: duration_weight(x))

#======================================================
# Implementing weightage based on duration of the case
#======================================================
val_data_2['count_of_email'] = val_data_2['count_of_email'].fillna(0)

val_data_2['Num_of_Email_weight'] =val_data_2['count_of_email'].apply(lambda x: num_of_customer_emails_weight(x))

# normalizing Sentiment and Intensity Score between 0-100
#=========================================================
x_min = np.min(val_data_2['Total_Sentiment_Score'])
x_max  = np.max(val_data_2['Total_Sentiment_Score'])
val_data_2['Total_Sentiment_Score_0_100'] = val_data_2['Total_Sentiment_Score'].apply(lambda x: (x - x_min)*100/(x_max-x_min))


xmin = np.min(val_data_2['total_score_intesity'])
xmax  = np.max(val_data_2['total_score_intesity'])
val_data_2['total_score_intesity_0_100'] = val_data_2['total_score_intesity'].apply(lambda x: (x - xmin)*100/(xmax-xmin))


#======================================================
# Weightage Escalation Score : Based on above 4 columns
#======================================================

val_data_2['Weightage_Escalation_Score'] = (val_data_2['duration_weight'] + val_data_2['Num_of_Email_weight']
                                           +val_data_2['Total_Sentiment_Score_0_100']+val_data_2['total_score_intesity_0_100'])/4


#pred Class label
val_data_2['Escalation_Class']= np.where( val_data_2['Weightage_Escalation_Score'] >50 ,1, 0)


# CHeck if any case number has NAN values remove the rows # 1 row present
val_data_2 = val_data_2[val_data_2['Case Number'].isnull()==False]

val_data_2['Negative_Words_Casewise']   = val_data_2['Negative_Words'].apply(lambda x: ','.join(x))
val_data_2['Positive_Words_Casewise']   = val_data_2['Positive_Words'].apply(lambda x: ','.join(x))
val_data_2['Escalation_Words_Casewise'] = val_data_2['escalation_words'].apply(lambda x: ','.join(x))


val_data_2['Negative_Words_Casewise']   = val_data_2.groupby(['Case Number'])['Negative_Words_Casewise'].transform(
                                              lambda x: ' '.join(x))
val_data_2['Positive_Words_Casewise']   = val_data_2.groupby(['Case Number'])['Positive_Words_Casewise'].transform(
                                              lambda x: ' '.join(x))
val_data_2['Escalation_Words_Casewise'] = val_data_2.groupby(['Case Number'])['Escalation_Words_Casewise'].transform(
                                              lambda x: ' '.join(x))



# COMMAND ----------

def word_count(text):
    text1 = text.split(',')
    mylist = list(dict.fromkeys(text1))
    text2 = mylist
    count = len(text2)
    return(count)
    

# COMMAND ----------

# there are inbuilt function in python for De-Duplication : Method 1: Use Set :  list(Set(list of duplicate values))
#                                                           Method 2: Using from collections import OrderedDict: --> Dict:fromkeys(list of duplicate values)   
#                                                           Method 3: Using Numpy : np.unique(list of duplicate values)
def remove_duplicate_words(text):
    text1 = text.split(',')
    unique_list = []
    
    for i in text1:
        if i not in unique_list:
            unique_list.append(i)
            
    return(unique_list)

# COMMAND ----------

#==================================================
#=#
#====================================================


val_data_2['Negative_Word_count'] = val_data_2['Negative_Words_Casewise'].apply(lambda x: word_count(x))
val_data_2['Positive_Word_count'] = val_data_2['Positive_Words_Casewise'].apply(lambda x: word_count(x))
val_data_2['Escalation_Word_count'] = val_data_2['Escalation_Words_Casewise'].apply(lambda x: word_count(x))

# COMMAND ----------

val_data_2

# COMMAND ----------

val_data_2.columns

# COMMAND ----------

val_data_2.head(1)

# COMMAND ----------

val=val_data_2[['sent', 'subject', 'message','Case Number', 'Opened Date', 'Closed Date',
       'Case Record Type', 'Case Status', 'To Address', 'From Address',
       'Agent_OR_Customer', 'Case_Duration', 'clean_message_2',
       'Sentiment', 'Positive_Words', 'Negative_Words',
       'escalation_words', 'Total_Sentiment_Score', 'total_score_intesity','count_of_email',
       'Negative_Words_Casewise',
       'Positive_Words_Casewise', 'Escalation_Words_Casewise',
       'Negative_Word_count', 'Positive_Word_count', 'Escalation_Word_count']]
val

# COMMAND ----------

val_data_final=val.append(temp2)
val_data_final

# COMMAND ----------

val_final=val_data_final[['sent', 'subject', 'message','Case Number', 'Opened Date', 'Closed Date',
       'Case Record Type', 'Case Status', 'To Address', 'From Address',
       'Agent_OR_Customer', 'Case_Duration', 'clean_message_2',
       'Sentiment', 'Positive_Words', 'Negative_Words',
       'escalation_words', 'Total_Sentiment_Score', 'total_score_intesity','count_of_email',
       'Negative_Words_Casewise',
       'Positive_Words_Casewise', 'Escalation_Words_Casewise',
       'Negative_Word_count', 'Positive_Word_count', 'Escalation_Word_count']]
val_final

# COMMAND ----------

val_final["count_of_email"]=val_final["count_of_email"].fillna(0)
val_final["Negative_Word_count"]=val_final["Negative_Word_count"].fillna(0)
val_final["Positive_Word_count"]=val_final["Positive_Word_count"].fillna(0)
val_final["Escalation_Word_count"]=val_final["Escalation_Word_count"].fillna(0)
val_final

# COMMAND ----------

val_final.sort_values(by=['Opened Date'])

# COMMAND ----------

val_data_22=val_data_2.append(temp2)
val_data_22

# COMMAND ----------

#============================================
# aggregating Numeric Columns at case level
#============================================

val_data_3=pd.DataFrame(val_data_22.groupby(['Case Number']).agg({'Case_Duration_days':'mean', 'Total_Sentiment_Score':'sum',
                                        'total_score_intesity':'sum', 'count_of_email':'mean',
                                        'duration_weight':'mean','Num_of_Email_weight':'mean',
                                        'Total_Sentiment_Score_0_100':'mean', 'total_score_intesity_0_100':'mean',
                                        'Weightage_Escalation_Score':'mean','Escalation_Class':'mean',
                                        'Negative_Word_count':'mean', 'Positive_Word_count':'mean',
                                        'Escalation_Word_count':'mean'})).reset_index()

# COMMAND ----------

val_data_3

# COMMAND ----------

# today_date=pd.Timestamp("today").strftime("%m%d%Y")
today_date=pd.Timestamp("today").strftime("%Y%m%d")
path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/val_data_3/'
val_data_3.to_csv(path+'val_data_3_'+str(today_date)+'.csv',index=False)

# COMMAND ----------

val_data_3.corr()

# COMMAND ----------

from xgboost import XGBClassifier, plot_importance
import pickle
file_name = '/dbfs/FileStore/xgb_clf.pkl'

# # save
# pickle.dump(XGB_clf, open(file_name, "wb"))

# load
xgb_model_loaded = pickle.load(open(file_name, "rb"))


X = val_data_3.copy()
X_test_sub = X.loc[:,~X.columns.isin(['Case Number'])]
prediction = xgb_model_loaded.predict(X_test_sub)

# COMMAND ----------

X_test_sub['XGB_pred'] = xgb_model_loaded.predict_proba(X_test_sub)[:,1]

# COMMAND ----------

# adding additional columns
X_test_sub['Case Number'] =val_data_3['Case Number']

# COMMAND ----------

X_test_sub

# COMMAND ----------

case_data_1 = spark.sql("select case_number, reason, line_of_business_c, zeb_product_family_c, zeb_product_line_c, Decode(sfdc_cases.abandoned_c, 1,'Y','N') abandoned_c, account_name_c, csr_account_number_c, zeb_account_type_c, account_country_c, zeb_affected_quantity_c, zeb_affected_region_c, Decode(sfdc_cases.zeb_alert_quality_c, 1,'Y','N') zeb_alert_quality_c, origin, entitlement_status_c, escalation_status_c, Decode(sfdc_cases.is_escalated,1,'Y','N') is_escalated, closed_date as closed_date_cst, priority, case_previous_owner_name_c, previous_case_record_type_c, zeb_case_age_in_business_days_c, Decode(sfdc_cases.case_closed_same_day_c, 1,'Y','N') case_closed_same_day_c, business_impact_avp_c, zeb_bug_or_enhancement_c, zeb_case_summary_isv_c, description, date_opened_c, closed_date, status, engineering_system_c, entitlement_name_c, sr_number_c, zeb_case_owner_role_c, id, previous_case_record_type_c, email_message_id_closed_case_c, end_user_account_c from silver_sfdc_prod.sfdc_cases")

# COMMAND ----------

from pyspark.sql import functions as f
case_data_1=case_data_1.withColumn("opened_date", f.from_unixtime(f.unix_timestamp("date_opened_c",'M/d/yyyy'),'yyyy-MM-dd').cast('date'))

# COMMAND ----------

from pyspark.sql.functions import expr

# case_data=case_data_1.where((col("opened_date") >= start) & (col("opened_date") <= end_date))
case_data=case_data_1.where((col("opened_date") >= '2020-01-01'))

# COMMAND ----------

case_data=case_data.filter(case_data.origin != "TS No Reply")

# COMMAND ----------

from pyspark.sql.functions import *
case_data = case_data.filter(col("closed_date").isNull())

# COMMAND ----------

case_data=case_data.fillna({'closed_date':'1901-01-01'})

# COMMAND ----------

# rename headers for case_data as per OAC report
newCols_case_data = ['Case Number', 'Case Reason', 'Line of Business', 'Product Family', 'Product Line', 'Abandoned', 'Account Name', 'Account Number', 'Account Type', 'Account Country', 'Affected Quantity', 'Affected Region', 'Alert Quality', 'Case Origin', 'Entitlement Status', 'Escalation Status', 'Escalated_c', 'Closed Date CST', 'Case Priority', 'Case Owner Name', 'Case Categorization', 'Case Age In Business Days', 'Case Closed Same Day', 'BusinessImpactAVP', 'Bug or Enhancement', 'Case Summary', 'Description', 'date_opened_c', 'Closed Date', 'Case Status', 'Engineering System', 'Escalation Title', 'SPR #', 'Case Owner Role', 'Case Id', 'Case Previous Record Type', 'Email Message ID', 'end_user_account_c', 'Opened Date']
case_data=case_data.toDF(*newCols_case_data)

# COMMAND ----------

cd_11=case_data.toPandas()
cd_11

# COMMAND ----------

cd_11['Closed Date'].unique()

# COMMAND ----------

cd_11['Case Categorization'].unique()

# COMMAND ----------

# cd_11=cd_11.loc[cd_11['Closed Date'].isnull()]
# # cd_11=cd_11.loc[cd_11['Closed Date'] == "1901-01-01 00:00:00"]
# cd_11

# COMMAND ----------

# cd_11['Closed Date']=cd_11['Closed Date'].fillna("1901-01-01 00:00:00")
# cd_11

# COMMAND ----------

cd_11['Opened Date'].min()

# COMMAND ----------

cd_11['Opened Date'].max()

# COMMAND ----------

# cd_11['Case Number'].nunique()

# COMMAND ----------

sfdc_accounts = spark.sql("SELECT id, name FROM silver_sfdc_prod.sfdc_accounts")
# sfdc_accounts.display()

# COMMAND ----------

sf_1=sfdc_accounts.toPandas()
sf_1= sf_1.rename(columns={'id': 'end_user_account_c', 'name': 'End User/Partner'})
sf_1

# COMMAND ----------

cd_1 = pd.merge(cd_11, sf_1, how='left', on='end_user_account_c')
cd_1

# COMMAND ----------

cd_1["End User/Partner"].unique()

# COMMAND ----------

X_test_sub.dtypes

# COMMAND ----------

X_test_sub["Case Number"]=pd.to_numeric(X_test_sub["Case Number"], errors='coerce')

# COMMAND ----------

X_test_sub["Case Number"] = X_test_sub["Case Number"].fillna(0)
X_test_sub["Case Number"]=X_test_sub["Case Number"].astype(int)

# COMMAND ----------

X_test_sub["Case Number"]=X_test_sub["Case Number"].astype(str)
X_test_sub.dtypes

# COMMAND ----------

X_test_sub.head(3)

# COMMAND ----------

cd_1.head(5)

# COMMAND ----------

cd_1.dtypes

# COMMAND ----------

cd_pred_test=pd.merge(cd_1,X_test_sub,how='left',on='Case Number')
cd_pred_test

# COMMAND ----------

cd_pred_test["XGB_pred"].value_counts()

# COMMAND ----------

cd_pred_test["XGB_pred"].min(), cd_pred_test["XGB_pred"].max()

# COMMAND ----------

###
CQC_SPR_RMA_flag=pd.read_csv("/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/CQC_SPR_RMA_flag/CQC_SPR_RMA_flag.csv")

# COMMAND ----------

CQC_SPR_RMA_flag

# COMMAND ----------

CQC_SPR_RMA_flag_1=CQC_SPR_RMA_flag[["Case Number", "CQC_SPR_RMA_flag"]]
CQC_SPR_RMA_flag_1

# COMMAND ----------

CQC_SPR_RMA_flag_1=CQC_SPR_RMA_flag_1.drop_duplicates()
CQC_SPR_RMA_flag_1

# COMMAND ----------

CQC_SPR_RMA_flag_1["Case Number"]=CQC_SPR_RMA_flag_1["Case Number"].astype(str)
CQC_SPR_RMA_flag_1.dtypes

# COMMAND ----------

cd_pred = pd.merge(cd_pred_test,CQC_SPR_RMA_flag_1,how='left',on='Case Number')
cd_pred

# COMMAND ----------

cd_pred['XGB_pred']=cd_pred['XGB_pred'].fillna('NaN')
cd_pred['CQC_SPR_RMA_flag']=cd_pred['CQC_SPR_RMA_flag'].fillna('NaN')
cd_pred

# COMMAND ----------

def flag_df(cd_pred):

    if (((cd_pred['XGB_pred'] == 'NaN') & (cd_pred['CQC_SPR_RMA_flag'] == 'Others'))):
        return 'No email conversation available'
    else:
        return cd_pred['CQC_SPR_RMA_flag']

cd_pred['Remarks'] = pd.DataFrame(cd_pred.apply(flag_df, axis = 1))

# COMMAND ----------

cd_pred

# COMMAND ----------

cd_pred['Remarks']=cd_pred['Remarks'].replace('NaN', 'No email conversation available')
cd_pred

# COMMAND ----------

cd_pred=cd_pred.drop(columns={"CQC_SPR_RMA_flag"})
cd_pred

# COMMAND ----------

cd_pred['Case_Duration_days']=cd_pred['Case_Duration_days'].fillna(0)
cd_pred['Total_Sentiment_Score']=cd_pred['Total_Sentiment_Score'].fillna(0)
cd_pred['total_score_intesity']=cd_pred['total_score_intesity'].fillna(0)
cd_pred['count_of_email']=cd_pred['count_of_email'].fillna(0)
cd_pred['duration_weight']=cd_pred['duration_weight'].fillna(0)
cd_pred['Num_of_Email_weight']=cd_pred['Num_of_Email_weight'].fillna(0)
cd_pred['Total_Sentiment_Score_0_100']=cd_pred['Total_Sentiment_Score_0_100'].fillna(0)
cd_pred['total_score_intesity_0_100']=cd_pred['total_score_intesity_0_100'].fillna(0)
cd_pred['Weightage_Escalation_Score']=cd_pred['Weightage_Escalation_Score'].fillna(0)
cd_pred['Escalation_Class']=cd_pred['Escalation_Class'].fillna(0)
cd_pred['Negative_Word_count']=cd_pred['Negative_Word_count'].fillna(0)
cd_pred['Positive_Word_count']=cd_pred['Positive_Word_count'].fillna(0)
cd_pred['Escalation_Word_count']=cd_pred['Escalation_Word_count'].fillna(0)
cd_pred['XGB_pred']=cd_pred['XGB_pred'].replace('NaN',0)
cd_pred

# COMMAND ----------

# cd_pred["Remarks"].value_counts()

# COMMAND ----------

# cd_pred["XGB_pred"].value_counts()

# COMMAND ----------

cd_pred["XGB_pred"].min(), cd_pred["XGB_pred"].max()

# COMMAND ----------

# cd_pred.loc[cd_pred['Case Origin'] != "TS No Reply"]

# COMMAND ----------

cd_pred['Closed Date'].nunique()

# COMMAND ----------

#writing output
# today_date=pd.Timestamp("today").strftime("%m%d%Y")
today_date=pd.Timestamp("today").strftime("%Y%m%d")
pred_path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_final/'
cd_pred.to_csv(pred_path+'prediction_set_caseinfo_'+str(today_date)+'.csv',index=False)

# COMMAND ----------

pred_path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_final_temp/'
X_test_sub.to_csv(pred_path+'prediction_set_'+str(today_date)+'.csv',index=False)

# COMMAND ----------

cd_pred['esc_or_not']=np.where(cd_pred.XGB_pred <0.5,0,1)

# COMMAND ----------

cd_pred

# COMMAND ----------

cd_pred["Case Categorization"].unique()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt


cd_pred1 = cd_pred[cd_pred.esc_or_not==1]
x = cd_pred.count_of_email
y = cd_pred.Case_Duration_days
colors = cd_pred['esc_or_not']
#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y,c=colors, alpha=0.5)
plt.title("Count of emails vs Case duration days ")
plt.xlabel('Count of emails')
plt.ylabel('Case duration days')
plt.legend(['esc','not_esc'], bbox_to_anchor = (1 , 1))
plt.xlim([0,20])
plt.ylim([0,100])
plt.rcParams["figure.figsize"] = (12,15)
plt.show()