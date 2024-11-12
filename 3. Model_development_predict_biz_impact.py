# Databricks notebook source
!pip install pandas-gbq google-cloud-storage google-cloud-bigquery
!pip install xlrd
!pip install openpyxl
!pip install gcsfs
!pip install gsutil

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

gst = spark.sql("select pos_prod_family, so_net_extd_value_lc, so_net_unit_value_lc, pos_invoice_date from silver_gst_prod.gst_sales_out_data_factory where pos_invoice_date >= '2018-01-01T00:00:00.000+0000'")
# gst.display()

# COMMAND ----------

gst_data=gst.toPandas()
gst_data

# COMMAND ----------

gst_data.dtypes

# COMMAND ----------

gst_data['so_net_extd_value_lc']=gst_data['so_net_extd_value_lc'].astype(float)
gst_data['so_net_unit_value_lc']=gst_data['so_net_unit_value_lc'].astype(float)
gst_data

# COMMAND ----------

gst_data['pos_prod_family'].nunique()

# COMMAND ----------

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# COMMAND ----------

gst_data_1=pd.DataFrame(gst_data.groupby(['pos_prod_family']).agg({'so_net_extd_value_lc':'sum', 'so_net_unit_value_lc':'sum'})).reset_index()
gst_data_1

# COMMAND ----------

total=gst_data_1['so_net_unit_value_lc'].sum().round(1)
total

# COMMAND ----------

gst_data_1["biz_imp"]=((gst_data_1["so_net_unit_value_lc"]/total)*100)
gst_data_1

# COMMAND ----------

# client = storage.Client()
# bucket = client.bucket("its-managed-dbx-ds-01-p-goas-scientists-workarea")
# print(pd.datetime.now())

# COMMAND ----------

fs = gcsfs.GCSFileSystem(project='its-managed-dbx-ds-01-p')

# COMMAND ----------

# blobs = bucket.list_blobs(prefix="escalation_detection_project/output_final/")

# COMMAND ----------

# file = []
# for blob in blobs:
#         file.append(blob.name)

# COMMAND ----------

# file

# COMMAND ----------

# latest_file=file[-1]
# latest_file

# COMMAND ----------

# with fs.open('gs://its-managed-dbx-ds-01-p-goas-scientists-workarea/'+latest_file) as f:
#     pred = pd.read_csv(f, keep_default_na=False)

# COMMAND ----------

import glob 
import os 

for file in glob.glob("/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_final/*"):
    print(file)

# COMMAND ----------

file

# COMMAND ----------

pred = pd.read_csv(file)
pred

# COMMAND ----------

pred["Case Number"]=pred["Case Number"].astype(str)
pred.dtypes

# COMMAND ----------

pred.Remarks.value_counts()

# COMMAND ----------

p_2=pred.query('Remarks == "No email conversation available"')
p_2

# COMMAND ----------

p_1=pred.query('Remarks == "SPR" | Remarks == "RMA" | Remarks == "CQC" | Remarks == "Others"')
p_1

# COMMAND ----------

pred_biz_imp_1 = pd.merge(p_1, gst_data_1, how='left', left_on=['Product Family'], right_on=['pos_prod_family'])
pred_biz_imp_1

# COMMAND ----------

pred_biz_imp_1["biz_imp"]=pred_biz_imp_1["biz_imp"].fillna('NaN')
pred_biz_imp_1["pos_prod_family"]=pred_biz_imp_1["pos_prod_family"].fillna('NaN')

# COMMAND ----------

pred_biz_imp_1["biz_imp"]=np.where(((pred_biz_imp_1["pos_prod_family"] == 'NaN')), 'NaN', pred_biz_imp_1["biz_imp"])
pred_biz_imp_1

# COMMAND ----------

pred_biz_imp_1['esc_or_not']=np.where(pred_biz_imp_1.XGB_pred <0.5,0,1)
pred_biz_imp_1

# COMMAND ----------

pred_biz_imp=pred_biz_imp_1.append(p_2)
pred_biz_imp

# COMMAND ----------

# today_date=pd.Timestamp("today").strftime("%m%d%Y")
today_date=pd.Timestamp("today").strftime("%Y%m%d")
pred_path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_biz_imp/'
pred_biz_imp.to_csv(pred_path+'prediction_set_caseinfo_bizimpact_'+str(today_date)+'.csv',index=False)

# COMMAND ----------

pred_bizimp_alldata = pd.read_csv('/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_alldata/escalation_detection_output.csv')
pred_bizimp_alldata

# COMMAND ----------

pred_bizimp_alldata=pred_bizimp_alldata.head(1)
pred_bizimp_alldata

# COMMAND ----------

pred_bizimp_alldata['Case Number'] = pred_bizimp_alldata['Case Number'].replace(7011731, np.NaN)
pred_bizimp_alldata

# COMMAND ----------

pred_bizimp_alldata['Escalated_c'] = pred_bizimp_alldata['Escalated_c'].replace({'No':'N', 'Yes':'Y'})
pred_bizimp_alldata

# COMMAND ----------

pred_bizimp_alldata["PRODUCT FAMILY"]=pred_bizimp_alldata["PRODUCT FAMILY"].fillna('NaN')

# COMMAND ----------

pred_bizimp_alldata["biz_impact"]=np.where(((pred_bizimp_alldata["PRODUCT FAMILY"] == 'NaN')), 'NaN', pred_bizimp_alldata["biz_impact"])
pred_bizimp_alldata

# COMMAND ----------

pred_bizimp_alldata["Case Number"]=pred_bizimp_alldata["Case Number"].astype(str)
pred_bizimp_alldata.dtypes

# COMMAND ----------

pred_bizimp_alldata["Opened Date"]=pd.to_datetime(pred_bizimp_alldata["Opened Date"])
pred_bizimp_alldata["Closed Date"]=pd.to_datetime(pred_bizimp_alldata["Closed Date"])
pred_bizimp_alldata

# COMMAND ----------

pred_bizimp_alldata["Remarks"]="Others"
pred_bizimp_alldata

# COMMAND ----------

pred_bizimp_alldata.columns

# COMMAND ----------

# blobs1 = bucket.list_blobs(prefix="escalation_detection_project/output_biz_imp/")

# COMMAND ----------

# file1 = []
# for blob in blobs1:
#         file1.append(blob.name)
# file1

# COMMAND ----------

# latest_file1=file1[-1]
# latest_file1

# COMMAND ----------

# pred_bizimp_recent = pd.DataFrame()
# for files in file1:
#     with fs.open('gs://its-managed-dbx-ds-01-p-goas-scientists-workarea/' + files) as f:
#             df = pd.read_csv(f,keep_default_na=False)       
#     pred_bizimp_recent = pred_bizimp_recent.append(df)

# COMMAND ----------

# with fs.open('gs://its-managed-dbx-ds-01-p-goas-scientists-workarea/'+latest_file1) as f:
#     pred_bizimp_recent = pd.read_csv(f, keep_default_na=False)
# pred_bizimp_recent

# COMMAND ----------

import glob 
import os 

for file1 in glob.glob("/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_biz_imp/*"):
    print(file1)

# COMMAND ----------

file1

# COMMAND ----------

pred_bizimp_recent = pd.read_csv(file1)
pred_bizimp_recent

# COMMAND ----------

pred_bizimp_recent["Closed Date"]=pd.to_datetime(pred_bizimp_recent["Closed Date"])
pred_bizimp_recent

# COMMAND ----------

pred_bizimp_recent.columns

# COMMAND ----------

pred_bizimp_recent=pred_bizimp_recent.rename(columns={'pos_prod_family':'PRODUCT FAMILY', 'so_net_extd_value_lc':'Report_Rate_Revenue_total', 'so_net_unit_value_lc':'Plan_Rate_Revenue_total', 'biz_imp': 'biz_impact', 'Line of Business': 'Line Of Business'})
pred_bizimp_recent

# COMMAND ----------

pred_bizimp_recent["Case Number"]=pred_bizimp_recent["Case Number"].astype(str)
pred_bizimp_recent.dtypes

# COMMAND ----------

pred_bizimp_recent_1=pred_bizimp_recent[['Case Number', 'Case Reason', 'Line Of Business', 'Product Family','Product Line', 'Abandoned', 'Account Country', 'Account Name', 'Alert Quality', 'Affected Region', 'Affected Quantity','Account Type', 'Account Number', 'Case Origin', 'Entitlement Status','Escalation Status', 'Escalated_c', 'Closed Date','Case Priority','Case Owner Name', 'Case Status', 'Case Categorization', 'Opened Date','Case Owner Role', 'end_user_account_c', 'End User/Partner', 'Case Age In Business Days','Case Closed Same Day', 'BusinessImpactAVP', 'Bug or Enhancement','Case_Duration_days', 'Total_Sentiment_Score','total_score_intesity', 'count_of_email', 'duration_weight','Num_of_Email_weight', 'Total_Sentiment_Score_0_100','total_score_intesity_0_100', 'Weightage_Escalation_Score','Escalation_Class', 'Negative_Word_count', 'Positive_Word_count','Escalation_Word_count', 'XGB_pred', 'Remarks', 'PRODUCT FAMILY', 'Report_Rate_Revenue_total', 'Plan_Rate_Revenue_total', 'biz_impact','esc_or_not']]
pred_bizimp_recent_1

# COMMAND ----------

biz_imp_final=pred_bizimp_alldata.append(pred_bizimp_recent_1)
biz_imp_final

# COMMAND ----------

biz_imp_final=biz_imp_final[biz_imp_final['Case Number']!='nan']
biz_imp_final

# COMMAND ----------

biz_imp_final["Case Number"].nunique()

# COMMAND ----------

biz_imp_final=biz_imp_final.drop_duplicates(subset="Case Number")
biz_imp_final

# COMMAND ----------

biz_imp_final["Case Number"].nunique()

# COMMAND ----------

biz_imp_final.dtypes

# COMMAND ----------

# biz_imp_final[biz_imp_final["Case Origin"] != "TS No Reply"]

# COMMAND ----------

biz_imp_final = biz_imp_final.replace(np.NaN, '')
biz_imp_final = biz_imp_final.replace('NaN', '')
biz_imp_final

# COMMAND ----------

biz_imp_final=biz_imp_final.sort_values(by=['Opened Date'])
biz_imp_final

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS escalation_detection

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType
from pyspark.sql import functions as F
from dateutil.relativedelta import relativedelta

# COMMAND ----------

# spark = SparkSession.builder \
#     .master("local") \
#     .appName("escalation_detection") \
#     .enableHiveSupport() \
#     .getOrCreate()

# COMMAND ----------

# biz_imp_final = biz_imp_final.replace(np.NaN, '')
# biz_imp_final = biz_imp_final.replace('NaN', '')
# biz_imp_final

# COMMAND ----------

biz_imp_final["Escalated_c"]=biz_imp_final["Escalated_c"].astype(str)
biz_imp_final["Entitlement Status"]=biz_imp_final["Entitlement Status"].astype(str)
biz_imp_final["PRODUCT FAMILY"]=biz_imp_final["PRODUCT FAMILY"].astype(str)
biz_imp_final["Account Country"]=biz_imp_final["Account Country"].astype(str)
biz_imp_final["Account Name"]=biz_imp_final["Account Name"].astype(str)
biz_imp_final["Affected Quantity"]=biz_imp_final["Affected Quantity"].astype(str)
biz_imp_final["Account Type"]=biz_imp_final["Account Type"].astype(str)
biz_imp_final["Case Owner Role"]=biz_imp_final["Case Owner Role"].astype(str)
biz_imp_final["end_user_account_c"]=biz_imp_final["end_user_account_c"].astype(str)
biz_imp_final["End User/Partner"]=biz_imp_final["End User/Partner"].astype(str)
biz_imp_final["Line Of Business"]=biz_imp_final["Line Of Business"].astype(str)
biz_imp_final["Affected Region"]=biz_imp_final["Affected Region"].astype(str)
biz_imp_final["Case Origin"]=biz_imp_final["Case Origin"].astype(str)
biz_imp_final["Product Family"]=biz_imp_final["Product Family"].astype(str)
biz_imp_final["Product Line"]=biz_imp_final["Product Line"].astype(str)
biz_imp_final["Line of Business"]=biz_imp_final["Line of Business"].astype(str)
biz_imp_final["Case Status"]=biz_imp_final["Case Status"].astype(str)
biz_imp_final["Case Reason"]=biz_imp_final["Case Reason"].astype(str)
biz_imp_final["Account Number"]=biz_imp_final["Account Number"].astype(str)
biz_imp_final["Case Priority"]=biz_imp_final["Case Priority"].astype(str)
biz_imp_final["Case Owner Name"]=biz_imp_final["Case Owner Name"].astype(str)
biz_imp_final["Case Categorization"]=biz_imp_final["Case Categorization"].astype(str)
biz_imp_final["Escalation Status"]=biz_imp_final["Escalation Status"].astype(str)
biz_imp_final["BusinessImpactAVP"]=biz_imp_final["BusinessImpactAVP"].astype(str)
biz_imp_final["Bug or Enhancement"]=biz_imp_final["Bug or Enhancement"].astype(str)
biz_imp_final["count_of_email"]=biz_imp_final["count_of_email"].astype(str)

# COMMAND ----------

biz_imp_final=biz_imp_final.rename(columns={'Line of Business':'Line_of_Business_1', 'PRODUCT FAMILY':'PRODUCT_FAMILY_1'})
biz_imp_final=biz_imp_final.rename(columns={'End User/Partner':'End User or Partner'})
biz_imp_final["Opened Date"]=pd.to_datetime(biz_imp_final["Opened Date"])
biz_imp_final["Created_date"]=pd.to_datetime(datetime.date.today())
biz_imp_final

# COMMAND ----------

biz_imp_final["duration_weight"]=biz_imp_final["duration_weight"].astype(str)
biz_imp_final["Num_of_Email_weight"]=biz_imp_final["Num_of_Email_weight"].astype(str)
biz_imp_final["Total_Sentiment_Score_0_100"]=biz_imp_final["Total_Sentiment_Score_0_100"].astype(str)
biz_imp_final["total_score_intesity_0_100"]=biz_imp_final["total_score_intesity_0_100"].astype(str)
biz_imp_final["Weightage_Escalation_Score"]=biz_imp_final["Weightage_Escalation_Score"].astype(str)
biz_imp_final["Escalation_Class"]=biz_imp_final["Escalation_Class"].astype(str)
biz_imp_final["Negative_Word_count"]=biz_imp_final["Negative_Word_count"].astype(str)
biz_imp_final["Positive_Word_count"]=biz_imp_final["Positive_Word_count"].astype(str)
biz_imp_final["Escalation_Word_count"]=biz_imp_final["Escalation_Word_count"].astype(str)
biz_imp_final["PRODUCT_FAMILY_1"]=biz_imp_final["PRODUCT_FAMILY_1"].astype(str)
biz_imp_final["Report_Rate_Revenue_total"]=biz_imp_final["Report_Rate_Revenue_total"].astype(str)
biz_imp_final["Plan_Rate_Revenue_total"]=biz_imp_final["Plan_Rate_Revenue_total"].astype(str)
biz_imp_final["biz_impact"]=biz_imp_final["biz_impact"].astype(str)
biz_imp_final["esc_or_not"]=biz_imp_final["esc_or_not"].astype(str)

# COMMAND ----------

biz_imp_final.dtypes

# COMMAND ----------

biz_imp_final

# COMMAND ----------

biz_imp_final["Case Categorization"].unique()

# COMMAND ----------

biz_imp_final["Case Categorization"] = biz_imp_final["Case Categorization"].replace('Closed Case', '')
biz_imp_final

# COMMAND ----------

biz_imp_final["Case Categorization"].unique()

# COMMAND ----------

def flag_df(biz_imp_final):

    if (((biz_imp_final['XGB_pred'] != 0) & (biz_imp_final['Remarks'] == 'No email conversation available'))):
        return 'Others'
    else:
        return biz_imp_final['Remarks']

biz_imp_final['Remarks_1'] = pd.DataFrame(biz_imp_final.apply(flag_df, axis = 1))

# COMMAND ----------

biz_imp_final["Remarks_1"].unique()

# COMMAND ----------

def flag_df(biz_imp_final):

    if (((biz_imp_final['XGB_pred'] == 0) & (biz_imp_final['Remarks_1'] == 'RMA'))):
        return 'No email conversation available'
    else:
        return biz_imp_final['Remarks_1']

biz_imp_final['Remarks_2'] = pd.DataFrame(biz_imp_final.apply(flag_df, axis = 1))

# COMMAND ----------

biz_imp_final["Remarks_2"].unique()

# COMMAND ----------

def flag_df(biz_imp_final):

    if (((biz_imp_final['XGB_pred'] == 0) & (biz_imp_final['Remarks_2'] == 'SPR'))):
        return 'No email conversation available'
    else:
        return biz_imp_final['Remarks_2']

biz_imp_final['Remarks_3'] = pd.DataFrame(biz_imp_final.apply(flag_df, axis = 1))

# COMMAND ----------

biz_imp_final["Remarks_3"].unique()

# COMMAND ----------

def flag_df(biz_imp_final):

    if (((biz_imp_final['XGB_pred'] == 0) & (biz_imp_final['Remarks_3'] == 'CQC'))):
        return 'No email conversation available'
    else:
        return biz_imp_final['Remarks_3']

biz_imp_final['Remarks_4'] = pd.DataFrame(biz_imp_final.apply(flag_df, axis = 1))
biz_imp_final

# COMMAND ----------

biz_imp_final["Remarks_4"].unique()

# COMMAND ----------

biz_imp_final=biz_imp_final.drop(["Remarks", "Remarks_1", "Remarks_2", "Remarks_3"],axis=1)
biz_imp_final=biz_imp_final.rename(columns={"Remarks_4":"Remarks"})
biz_imp_final

# COMMAND ----------

biz_imp_final=biz_imp_final[['Case Number', 'Case Reason', 'Line Of Business', 'Product Family',
       'Product Line', 'Abandoned', 'Account Country', 'Account Name',
       'Alert Quality', 'Affected Region', 'Affected Quantity', 'Account Type',
       'Account Number', 'Case Origin', 'Entitlement Status',
       'Escalation Status', 'Escalated_c', 'Closed Date', 'Case Priority',
       'Case Owner Name', 'Case Status', 'Case Categorization', 'Opened Date',
       'Case Age In Business Days', 'Case Closed Same Day',
       'BusinessImpactAVP', 'Bug or Enhancement', 'Line_of_Business_1',
       'Case_Duration_days', 'Total_Sentiment_Score', 'total_score_intesity',
       'count_of_email', 'duration_weight', 'Num_of_Email_weight',
       'Total_Sentiment_Score_0_100', 'total_score_intesity_0_100',
       'Weightage_Escalation_Score', 'Escalation_Class', 'Negative_Word_count',
       'Positive_Word_count', 'Escalation_Word_count', 'XGB_pred', 'EU_NAME',
       'PRODUCT_FAMILY_1', 'Report_Rate_Revenue', 'Plan_Rate_Revenue',
       'Report_Rate_Revenue_total', 'Plan_Rate_Revenue_total', 'biz_impact',
       'esc_or_not', 'Remarks', 'Case Owner Role', 'end_user_account_c',
       'End User or Partner', 'Created_date']]
biz_imp_final

# COMMAND ----------

# today_date=pd.Timestamp("today").strftime("%m%d%Y")
today_date=pd.Timestamp("today").strftime("%Y%m%d")
pred_path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_to_zdl/'
biz_imp_final.to_csv(pred_path+'bizimpact_final_output_'+str(today_date)+'.csv',index=False)

# COMMAND ----------

biz_imp_final["XGB_pred"].value_counts()

# COMMAND ----------

biz_imp_final['Opened Date'].min(), biz_imp_final['Opened Date'].max()

# COMMAND ----------

biz_imp_final["Case Number"].nunique()

# COMMAND ----------

# Convert the Pandas Dataframe to Spark Dataframe
final_df_sp = spark.createDataFrame(biz_imp_final)

# COMMAND ----------

# Write to the Delta Table
final_df_sp.write.mode("overwrite").format("delta").saveAsTable("escalation_detection.escalation_detection_biz_impact")