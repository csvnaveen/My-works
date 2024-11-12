# Databricks notebook source
!pip install pandas-gbq google-cloud-storage google-cloud-bigquery
!pip install xlrd
!pip install openpyxl
!pip install gcsfs
!pip install gsutil

# COMMAND ----------

!pip install nltk

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

pq_frame = spark.sql("SELECT id, parent_id, activity_id, created_by_id, created_date, last_modified_date, last_modified_by_id, system_modstamp, text_body, headers, subject, from_name, from_address, validated_from_address, to_address, cc_address, bcc_address, incoming, has_attachment, status, message_date, is_deleted, reply_to_email_message_id, is_private_draft, is_externally_visible, message_identifier, thread_identifier, is_client_managed, related_to_id, is_tracked, is_opened, first_opened_date, last_opened_date, is_bounced, email_template_id, case_account_id_c, case_contact_id_c, case_origin_c, case_previous_record_type_c, case_product_model_c, case_record_type_c, case_status_c, has_attachments_present_c, is_zsb_series_email_c FROM silver_sfdc_prod.sfdc_email_messages WHERE created_date >= '2020-01-01T00:00:00' order by created_date")

# COMMAND ----------

# rename headers for pq_frame1 as per OAC report
newCols_pq_frame = ['ID', 'PARENTID', 'ACTIVITYID', 'CREATEDBYID', 'CREATEDDATE', 'LASTMODIFIEDDATE', 'LASTMODIFIEDBYID', 'SYSTEMMODSTAMP', 'TEXTBODY', 'HEADERS', 'SUBJECT', 'FROMNAME', 'FROMADDRESS', 'VALIDATEDFROMADDRESS', 'TOADDRESS', 'CCADDRESS', 'BCCADDRESS', 'INCOMING', 'HASATTACHMENT', 'STATUS', 'MESSAGEDATE', 'ISDELETED', 'REPLYTOEMAILMESSAGEID', 'ISPRIVATEDRAFT', 'ISEXTERNALLYVISIBLE', 'MESSAGEIDENTIFIER', 'THREADIDENTIFIER', 'ISCLIENTMANAGED', 'RELATEDTOID', 'ISTRACKED', 'ISOPENED', 'FIRSTOPENEDDATE', 'LASTOPENEDDATE', 'ISBOUNCED', 'EMAILTEMPLATEID', 'CASE_ACCOUNT_ID__C', 'CASE_CONTACT_ID__C', 'CASE_ORIGIN__C', 'CASE_PREVIOUS_RECORD_TYPE__C', 'CASE_PRODUCT_MODEL__C', 'CASE_RECORD_TYPE__C', 'CASE_STATUS__C', 'HAS_ATTACHMENTS_PRESENT__C', 'IS_ZSB_SERIES_EMAIL__C']
pq_frame=pq_frame.toDF(*newCols_pq_frame)

# COMMAND ----------

# pq_frame.display()

# COMMAND ----------

case_data_1 = spark.sql("select case_number, reason, line_of_business_c, zeb_product_family_c, zeb_product_line_c, Decode(sfdc_cases.abandoned_c, 1,'Y','N') abandoned_c, account_name_c, csr_account_number_c, zeb_account_type_c, account_country_c, zeb_affected_quantity_c, zeb_affected_region_c, Decode(sfdc_cases.zeb_alert_quality_c, 1,'Y','N') zeb_alert_quality_c, origin, entitlement_status_c, escalation_status_c, Decode(sfdc_cases.is_escalated,1,'Y','N') is_escalated, closed_date as closed_date_cst, priority, case_previous_owner_name_c, previous_case_record_type_c, zeb_case_age_in_business_days_c, Decode(sfdc_cases.case_closed_same_day_c, 1,'Y','N') case_closed_same_day_c, business_impact_avp_c, zeb_bug_or_enhancement_c, zeb_case_summary_isv_c, description, date_opened_c, closed_date, status, engineering_system_c, entitlement_name_c, sr_number_c, zeb_case_owner_role_c, id, previous_case_record_type_c, email_message_id_closed_case_c from silver_sfdc_prod.sfdc_cases")

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

# rename headers for case_data as per OAC report
newCols_case_data = ['Case Number', 'Case Reason', 'Line of Business', 'Product Family', 'Product Line', 'Abandoned', 'Account Name', 'Account Number', 'Account Type', 'Account Country', 'Affected Quantity', 'Affected Region', 'Alert Quality', 'Case Origin', 'Entitlement Status', 'Escalation Status', 'Escalated_c', 'Closed Date CST', 'Case Priority', 'Case Owner Name', 'Case Categorization', 'Case Age In Business Days', 'Case Closed Same Day', 'BusinessImpactAVP', 'Bug or Enhancement', 'Case Summary', 'Description', 'date_opened_c', 'Closed Date', 'Case Status', 'Engineering System', 'Escalation Title', 'SPR #', 'Case Owner Role', 'Case Id', 'Case Previous Record Type', 'Email Message ID', 'Opened Date']
case_data=case_data.toDF(*newCols_case_data)

# COMMAND ----------

# case_data.display()

# COMMAND ----------

# join case_data and pq_frame to get cmm data
cmm_pq = case_data.join(pq_frame, case_data["Case ID"] == pq_frame['PARENTID'],"left")

# COMMAND ----------

cmm_pq_1=cmm_pq.dropDuplicates(['Case Number'])

# COMMAND ----------

cmm_pq_2=cmm_pq_1.select('Opened Date', 'Closed Date', 'SUBJECT', 'ID', 'Case Id', 'Case Number','CREATEDDATE','CASE_RECORD_TYPE__C','CASE_STATUS__C', 'LASTMODIFIEDBYID','LASTMODIFIEDDATE','MESSAGEDATE','TOADDRESS','FROMADDRESS')
# cmm_pq_2=cmm_pq_2.select(sorted(cmm_pq_2.columns))

# COMMAND ----------

# rename headers for cmm as per OAC report
newCols_cmm = ['Opened Date', 'Closed Date', 'Subject', 'Email Message ID', 'Case Id', 'Case Number', 'Create Date', 'Case Record Type', 'Case Status', 'Last Modified By', 'Last Modified Date', 'Message Date', 'To Address', 'From Address']

cmm=cmm_pq_2.toDF(*newCols_cmm)

# COMMAND ----------

# cmm.display()

# COMMAND ----------

from pyspark.sql.functions import *
cmm.select(max(to_date(col("Opened Date"))),min(to_date(col("Opened Date"))),max(to_date(col("Closed Date"))),min(to_date(col("Closed Date")))).show()

# COMMAND ----------

###
pq_frame1=pq_frame.select('ID','TEXTBODY')
pq_frame1

# COMMAND ----------

cmm = cmm.withColumnRenamed('Subject', 'Subject_cmm')
cmm_pq = pq_frame1.join(cmm,pq_frame1.ID == cmm['Email Message ID'],"inner")

# COMMAND ----------

###
from pyspark.sql.functions import *
cmm_pq.select(max(to_date(col("Opened Date"))),min(to_date(col("Opened Date"))),max(to_date(col("Closed Date"))),min(to_date(col("Closed Date")))).show()

# COMMAND ----------

from pyspark.sql.functions import when,col

cmm_pq_cd_esc=cmm_pq.withColumn('CQC_SPR_RMA_flag', when(col('Subject_cmm').like('%RMA%'), 'RMA').when(col('Subject_cmm').like('%CQC%'), 'CQC').when(col('Subject_cmm').like('%SPR%'), 'SPR').otherwise('Others'))

# COMMAND ----------

from pyspark.sql import functions as f
cmm_pq_cd_esc_flag=cmm_pq_cd_esc.withColumn("Create Date", f.from_unixtime(f.unix_timestamp(cmm_pq_cd_esc["Create Date"]), "yyyy-MM-dd")) \
                .withColumn("Closed Date", f.from_unixtime(f.unix_timestamp(cmm_pq_cd_esc["Closed Date"]), "yyyy-MM-dd")) \
                .withColumn("Message Date", f.from_unixtime(f.unix_timestamp(cmm_pq_cd_esc["Message Date"]), "yyyy-MM-dd"))

# COMMAND ----------

df1, df2, df3, df4 = cmm_pq_cd_esc_flag.randomSplit(weights=[0.3,0.3,0.3,0.1])

# COMMAND ----------

# df1.count(), df2.count(), df3.count(), df4.count()

# COMMAND ----------

df11=df1.toPandas()

# COMMAND ----------

df22=df2.toPandas()

# COMMAND ----------

df33=df3.toPandas()

# COMMAND ----------

df44=df4.toPandas()

# COMMAND ----------

cmm_pq_cd_esc_flag_1=df11.append([df22,df33,df44])
cmm_pq_cd_esc_flag_1

# COMMAND ----------

# today_date=pd.Timestamp("today").strftime("%m%d%Y")
path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/CQC_SPR_RMA_flag/'
cmm_pq_cd_esc_flag_1.to_csv(path+'CQC_SPR_RMA_flag.csv',index=False)

# COMMAND ----------

# cmm_pq_cd_esc_1 = cmm_pq_cd_esc.filter((cmm_pq_cd_esc.CQC_SPR_RMA_flag != 'RMA') & (cmm_pq_cd_esc.CQC_SPR_RMA_flag != 'SPR'))
# cmm_pq_cd_esc_1 = cmm_pq_cd_esc_1.dropDuplicates()

# including RMA and SPR in the data
cmm_pq_cd_esc_1 = cmm_pq_cd_esc.dropDuplicates()

# COMMAND ----------

from pyspark.sql.functions import *
cmm_pq_cd_esc_1 = cmm_pq_cd_esc_1.filter(col("Closed Date").isNull())
cmm_pq_cd_esc_1=cmm_pq_cd_esc_1.withColumn('closed_date_updated',to_date(col("Closed Date")))
# cmm_pq_cd_esc_1 = cmm_pq_cd_esc_1.filter(cmm_pq_cd_esc_1.closed_date_updated == '1901-01-01')

# COMMAND ----------

cmm_pq_cd_esc_1=cmm_pq_cd_esc_1.fillna({'closed_date_updated':'1901-01-01'})
cmm_pq_cd_esc_1=cmm_pq_cd_esc_1.fillna({'Closed Date':'1901-01-01'})

# COMMAND ----------

# cmm_pq_cd_esc_1.count()

# COMMAND ----------

############################################################
# creating function to store message body from EMAIL THREAD
#-----------------------------------------------------------
import re
def extract_messagebody(text):
    # extracting message body from email thread

    groups = re.findall(r'From:(.*?)Sent:(.*?)To:(.*?)Subject:(.*?)$(.*?)(?=^From:|\Z)', text, flags=re.DOTALL|re.M)
    emails = []
    msgbody= []
    for g in groups:
        d = {}
        d['from'] = g[0].strip()
        d['sent'] = g[1].strip()
        d['to'] = g[2].strip()
        d['subject'] = g[3].strip()
        d['message'] = g[4].strip()
        emails.append(d)
    return(emails)

# COMMAND ----------

prep_email_data=cmm_pq_cd_esc_1
prep_email_data1 = prep_email_data.cache()
prep_email_data2 = prep_email_data1.select('TEXTBODY','Case Id','ID','Case Number','Opened Date','Create Date','Closed Date','Case Record Type','Case Status','Last Modified By','Last Modified Date','Message Date','To Address','From Address','CQC_SPR_RMA_flag')

# COMMAND ----------

from pyspark.sql import functions as f
prep_email_data2=prep_email_data2.withColumn("Create Date", f.from_unixtime(f.unix_timestamp(prep_email_data2["Create Date"]), "yyyy-MM-dd")) \
                .withColumn("Closed Date", f.from_unixtime(f.unix_timestamp(prep_email_data2["Closed Date"]), "yyyy-MM-dd")) \
                .withColumn("Message Date", f.from_unixtime(f.unix_timestamp(prep_email_data2["Message Date"]), "yyyy-MM-dd"))

# COMMAND ----------

prep_email_data3 = prep_email_data2.toPandas()

# COMMAND ----------

prep_email_data3

# COMMAND ----------

############################
# replacing None values with 'NAs'
#================================
prep_email_data=prep_email_data3
prep_email_data['TEXTBODY'] = prep_email_data['TEXTBODY'].replace(np.nan, 'NA', regex=True)

# COMMAND ----------

prep_email_data['Create Date']=pd.to_datetime(prep_email_data['Create Date'])
prep_email_data['Opened Date']=pd.to_datetime(prep_email_data['Opened Date'], errors="coerce")
prep_email_data['Closed Date']=pd.to_datetime(prep_email_data['Closed Date'])

# COMMAND ----------

prep_email_data['Opened Date'].min(),prep_email_data['Opened Date'].max()

# COMMAND ----------

prep_email_data

# COMMAND ----------

prep_email_data['Case Number'].nunique()

# COMMAND ----------

prep_email_data[prep_email_data['Opened Date'].dt.year==2022]['Case Number'].nunique()

# COMMAND ----------

# removing few disclaimers from the mail
d0 ='''is legally privileged, proprietary, strictly confidential and exempt from disclosure, which is not waived or lost by mis-transmission or error. If you are not the original intended recipient of this message, it may be unlawful and illegal for you to read, print, retain, copy, disseminate, disclose or otherwise use this message, or take any action in reliance on it, and the same is prohibited and forbidden. If you have received this email in error, please notify the sender immediately by return reply and delete the message from your system without printing or making a copy. Neither the sender nor we are liable for any loss or damage as a result of this message, or for any delay, interception, corruption, virus, error, omission, improper or incomplete transmission thereof'''
d1 = '''is legally privileged, proprietary, strictly confidential and exempt from disclosure, which is not waived or lost by mis-transmission or error. If you are not the original intended recipient of this message, it may be unlawful and illegal for you to read, print, retain, copy, disseminate, disclose or otherwise use this message, or take any action in reliance on it, and the same is prohibited and forbidden'''
d2 = '''This email and any files transmitted with it are confidential, and may also be legally privileged. If you are not the intended recipient, you may not review, use, copy, or distribute this message. If you receive this email in error, please notify the sender immediately by reply email and then delete this email'''
d3 ='''If you have received this email in error, please notify the sender immediately by return reply and delete the message from your system without printing or making a copy. Neither the sender nor we are liable for any loss or damage as a result of this message, or for any delay, interception, corruption, virus, error, omission, improper or incomplete transmission'''
d4 = '''Please do not reply to this email'''
d5= '''To contact our Support Team, please follow the instructions below'''
d6 ='''For faster responses and to view status updates any time, please visit our online portals'''
d7 ='''do-not-delete-this-line'''

prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY'].str.replace(d0,'',regex=True)
prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY_2'].str.replace(d1,'',regex=True)
prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY_2'].str.replace(d2,'',regex=True)
prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY_2'].str.replace(d3,'',regex=True)
prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY_2'].str.replace(d4,'',regex=True)
prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY_2'].str.replace(d5,'',regex=True)
prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY_2'].str.replace(d6,'',regex=True)
prep_email_data['TEXTBODY_2'] =prep_email_data['TEXTBODY_2'].str.replace(d7,'',regex=True)

# COMMAND ----------

#creating a dataframe to combine all the textbody for every case id
import time
df = prep_email_data
start = time.time()

listofcaseIds = df['Case Number'].unique()
columns = ['from','sent','to','subject','message','ID','CaseID']
data = [['NA','NA','NA','NA','NA','NA','NA']]
message_body =pd.DataFrame(data,columns=columns)
temp3 =pd.DataFrame(data,columns=columns)
count = 0
for id in listofcaseIds:
    listofIds = df[df['Case Number']==id]['ID'].unique()
    for i in listofIds:
        temp = df[df['ID']==i]['TEXTBODY_2']
        temp = ''.join(temp)
        try:
            if ('From' in temp) & ('To' in temp) & ('Subject' in temp):
                temp1 = extract_messagebody(temp)
                temp2 = pd.DataFrame(temp1)
                temp2['ID'] = i
                temp2['CaseID'] = id
                message_body = message_body.append(temp2)


            else: 
                temp3['from'] = 'NA'
                temp3['sent'] = 'NA'
                temp3['to'] = 'NA'
                temp3['subject'] = 'NA'
                temp3['message'] = temp
                temp3['ID'] = i
                temp3['CaseID'] = id
                message_body = message_body.append(temp3)
        except:
             pass
    count = count+1
    print(count)
#     end = time.time()
#     print('Single ID Execution Time: ',end - start)

end = time.time()
print('Complete Execution Time: ',end - start)

# COMMAND ----------

# count of case id from message body
len(message_body['CaseID'].unique()), len(prep_email_data['Case Number'].unique())

# COMMAND ----------

#===================================++++++++++++++++
# Subset data frame
#===================================++++++++++++++++
prep_email_data_sub =prep_email_data[['ID','Case Number','Opened Date','Create Date','Closed Date','Case Record Type','Case Status','Last Modified By','Last Modified Date','Message Date','To Address','From Address','CQC_SPR_RMA_flag']]

# COMMAND ----------

#===================================++++++++++++++++
# Merging Message body with email details sub table
#===================================++++++++++++++++

msg_body_prep_email_data = pd.merge(message_body,prep_email_data_sub,left_on='ID', right_on='ID', how='left')

# COMMAND ----------

msg_body_prep_email_data.shape, message_body.shape, prep_email_data_sub.shape, len(msg_body_prep_email_data['Case Number'].unique()),len(message_body['CaseID'].unique()),len(prep_email_data_sub['Case Number'].unique())

# COMMAND ----------

msg_body_prep_email_data

# COMMAND ----------

msg_body_prep_email_data.dtypes

# COMMAND ----------

######################################
# customer & agent identification based on email

msg_body_prep_email_data['Agent_OR_Customer'] = np.where(msg_body_prep_email_data['From Address'].str.contains('zebra.com'),'Agent', 'Customer')



#===================================++++++++++++++++
#Calculating case duration
#===================================++++++++++++++++
if_old_sent_date = pd.to_datetime(msg_body_prep_email_data['Message Date'],utc=True,errors = 'coerce') - pd.to_datetime(msg_body_prep_email_data['Opened Date'], utc=True,errors = 'coerce')
msg_body_prep_email_data['Case_Duration'] = np.where( pd.to_datetime(msg_body_prep_email_data['Closed Date'],utc=True,errors = 'coerce') < '2019-01-01 12:03:00+00:00', if_old_sent_date, pd.to_datetime(msg_body_prep_email_data['Closed Date'],utc=True,errors = 'coerce') - pd.to_datetime(msg_body_prep_email_data['Opened Date'], utc=True,errors = 'coerce'))


# # #===================================++++++++++++++++
# # #Calculating case duration
# # #===================================++++++++++++++++
# if_old_sent_date = pd.to_datetime(msg_body_prep_email_data['Message Date'],utc=True) - pd.to_datetime(msg_body_prep_email_data['Opened Date'], utc=True)
# msg_body_prep_email_data['Case_Duration'] = np.where( pd.to_datetime(msg_body_prep_email_data['sent_date'],utc=True) < '2019-01-01 12:03:00+00:00', if_old_sent_date,  
#                                                      pd.to_datetime(msg_body_prep_email_data['sent_date'],utc=True) - pd.to_datetime(msg_body_prep_email_data['Opened Date'], utc=True))

# COMMAND ----------

msg_body_prep_email_data['Opened Date'].isnull().sum()

# COMMAND ----------

msg_body_prep_email_data.shape,message_body.shape,prep_email_data_sub.shape

# COMMAND ----------

#================================================
# Testing of removal of duplicates message body
#------------------------------------------------
chk_dupl = msg_body_prep_email_data.head(70)
chkno_dupl = chk_dupl.drop_duplicates(subset='message', keep="first")
chkno_dupl[chkno_dupl['ID']=='02s0H000016gfJvQAI']

# validated for IDs '02s0H000016gfJvQAI' and '02s0H000016fCxHQAU'


#==============================
# Applying for all data frame
#------------------------------

msg_body_prep_email_dataNoDupl = msg_body_prep_email_data.drop_duplicates(subset='message', keep="first")




# COMMAND ----------

#---------------------------------
# considering only customer reply
#----------------------------------
#data = msg_body_prep_email_dataNoDupl[msg_body_prep_email_dataNoDupl['Agent_OR_Customer']=='Customer']

data =msg_body_prep_email_dataNoDupl.copy()

# COMMAND ----------

####################################################
# Creating label using Case Age Duration
####################################################

def create_label(age):
    
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

#######################################################################
# Creating label using Case Age Duration : Running overall dataframe
#######################################################################


data['Case_Duration_days'] = data['Case_Duration'].dt.days

# replacing NANs with 0  days duration
data['Case_Duration_days'] =data['Case_Duration_days'].replace(np.nan, 0, regex=True)

# converting to int 
data['Case_Duration_days'] = (data['Case_Duration_days'].astype(int)).abs()
#data['Case_Duration_days'] = abs(lit(data['Case_Duration_days']).astype(int))

# general rule of escalation (for all cases) ( irrespective of Lee's file)
data['label'] = data['Case_Duration_days'].apply(lambda x: create_label(x))

# COMMAND ----------

data

# COMMAND ----------

data['Case_Duration_days'].unique()

# COMMAND ----------

# cleaning Message body:
 
SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

def clean(text, stem_words=True):
    import re
    from string import punctuation
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    
    def pad_str(s):
        return ' '+s+' '
    
    if pd.isnull(text):
        return ''

    
    if type(text) != str or text=='':
        return ''

    # Clean the text
    text = re.sub("\'s", " ", text) 
    text = re.sub("@[\w]*", "", text)
    text = re.sub("[^a-zA-Z#]", " " , text)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    # add padding to punctuations and special chars, we still need them later
    
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    

        
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) # replace non-ascii word with special word
    
    # indian dollar
    
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rt", " ", text, flags=re.IGNORECASE)
    
    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^www?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    text = re.sub('[0-9]+\.[0-9]+', " NUMBER ", text)
  
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
       # Return a list of words
    return text

# COMMAND ----------

disc= '''CONFIDENTIAL-\n\nThis email and any files transmitted with it are confidential, and may also be legally privileged. If you are not the intended recipient, you may not review, use, copy, or distribute this message. If you receive this email in error, please notify the sender immediately by reply email and then delete this email.\n________________________________\n- CONFIDENTIAL-\nThis email and any files transmitted with it are confidential, and may also be legally privileged. If you are not the intended recipient, you may not review, use, copy, or distribute this message. If you receive this email in error, please notify the sender immediately by reply email and then delete this email.'''
data['clean_message1'] = data['message'].str.replace(disc, '')


# applying above Clean function to clean
data['clean_message'] = data['clean_message1'].apply(clean)
data['clean_message'] = data['clean_message1']

# COMMAND ----------

###############################++++++++++++++
#removing non-english words eg. manoel dos anjos
#################################++++++++++++++

import nltk
nltk.download('words')
meaningful_words = []
def remove_nonenglish(text):
    words = set(nltk.corpus.words.words())
    meaningful_words = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    return(meaningful_words)


# COMMAND ----------

###############################++++++++++++++
#removing non-english words eg. manoel dos anjos
#################################++++++++++++++
data['clean_message_2'] =data['clean_message'].apply(remove_nonenglish)

# COMMAND ----------

#============================================================
# Calculating email count per case from agent and customer
#===========================================================

cust_email_count = pd.DataFrame(data[data['Agent_OR_Customer']=='Customer'].groupby(['Case Number','Agent_OR_Customer']).size()).reset_index()
cust_email_count['customer_email_count'] = cust_email_count[0]


agent_email_count = pd.DataFrame(data[data['Agent_OR_Customer']=='Agent'].groupby(['Case Number','Agent_OR_Customer']).size()).reset_index()
agent_email_count['agent_email_count'] = agent_email_count[0]

del cust_email_count[0]
del agent_email_count[0]

agent_cust_email_count = pd.merge(cust_email_count,agent_email_count, left_on='Case Number', right_on='Case Number' , how ='left')

# COMMAND ----------

cust_email_count.shape, agent_email_count.shape, len(cust_email_count['Case Number'].unique()),len(agent_email_count['Case Number'].unique())

# COMMAND ----------

# filling NAn values with 0 ( NO email from Agent)
agent_cust_email_count['agent_email_count'] = agent_cust_email_count['agent_email_count'].fillna(0)

# Calculating email count difference betweent agent and customer
agent_cust_email_count['Email_count_diff'] = agent_cust_email_count['customer_email_count'] - agent_cust_email_count['agent_email_count']

# COMMAND ----------

data

# COMMAND ----------

# save clean data in parquet file
data['CaseID'] =data['CaseID'].astype(str)
data['Case_Duration'] =data['Case_Duration'].astype(str)
# today_date=pd.Timestamp("today").strftime("%m%d%Y")
today_date=pd.Timestamp("today").strftime("%Y%m%d")
path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_predict/'
#data.to_parquet(path+'clean_message_sub_data_train_'+str(today_date)+'.parquet')

# COMMAND ----------

data.to_csv(path+'clean_message_sub_data_predict_'+str(today_date)+'.csv',index=False)

# COMMAND ----------

# save clean data in parquet file
data['CaseID'] =data['CaseID'].astype(str)
#data['Case_Duration'] =data['Case_Duration'].astype(str)
# today_date=pd.Timestamp("today").strftime("%m%d%Y")
today_date=pd.Timestamp("today").strftime("%Y%m%d")
path = '/Volumes/prod_catalog/shared_volume/gscr_ds/escalation_detection_project/output_predict_temp/'
#data.to_parquet(path+'clean_message_sub_data_train_'+str(today_date)+'.parquet')
data.to_csv(path+'clean_message_sub_data_predict_temp_'+str(today_date)+'.csv',index=False)

# COMMAND ----------

pd.Timestamp("today").strftime("%Y%m%d")