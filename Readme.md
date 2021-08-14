

Predicting Hospital Readmission

for Diabetes Patients

BY MACHINE LEARNING APPROACHES

Capstone Report

Mentored by:

**Srikar Muppidi**

Submitted by:

**Manoj Kumar. T.N**

**Prasanth A.B**

**RamNivash. B.S**

**Shiva. C.R**

**Sivakarthikayan. B**

1





Table of Contents

[**ABSTRACT**](#br5)

[**5**](#br5)

[**5**](#br5)

[**6**](#br6)

[**INTRODUCTION**](#br5)

[**DATASET**](#br6)[** ](#br6)[AND**](#br6)[** ](#br6)[DOMAIN**](#br6)

[**V**](#br6)[ARIABLE**](#br6)[** ](#br6)[C**](#br6)[ATEGORIZATION**](#br6)

[**D**](#br6)[ATA**](#br6)[** ](#br6)[D**](#br6)[ICTIONARY**](#br6)

[**6**](#br6)

[**6**](#br6)

[**DATA**](#br8)[** ](#br8)[PRE-PROCESSING**](#br8)

[**8**](#br8)

[**N**](#br8)[ULL**](#br8)[/M**](#br8)[ISSING**](#br8)[** ](#br8)[V**](#br8)[ALUE**](#br8)[** ](#br8)[D**](#br8)[ETECTION**](#br8)

[**R**](#br9)[EMOVING**](#br9)[** ](#br9)[U**](#br9)[NIQUELY**](#br9)[** ](#br9)[R**](#br9)[ANDOM**](#br9)[** ](#br9)[V**](#br9)[ALUED**](#br9)[** ](#br9)[C**](#br9)[OLUMNS**](#br9)

[**R**](#br9)[EMOVING**](#br9)[** ](#br9)[D**](#br9)[EGENERATE**](#br9)[** ](#br9)[F**](#br9)[EATURES**](#br9)

[**F**](#br9)[EATURE**](#br9)[** ](#br9)[E**](#br9)[NGINEERING**](#br9)

[**8**](#br8)

[**9**](#br9)

[**9**](#br9)

[**9**](#br9)

[D](#br9)[IAGNOSIS](#br9)[-1,](#br9)[ ](#br9)[D](#br9)[IAGNOSIS](#br9)[-2](#br9)[ ](#br9)[D](#br9)[IAGNOSIS](#br9)[-3](#br9)

[A](#br10)[DMISSION](#br10)[ ](#br10)[T](#br10)[YPE](#br10)[ ](#br10)[ID](#br10)

[D](#br11)[ISCHARGE](#br11)[ ](#br11)[D](#br11)[ISPOSITION](#br11)[ ](#br11)[ID](#br11)

[A](#br11)[DMISSION](#br11)[ ](#br11)[S](#br11)[OURCE](#br11)[ ](#br11)[ID](#br11)

[**F**](#br12)[EATURE**](#br12)[** ](#br12)[E**](#br12)[XTRACTION**](#br12)

[P](#br12)[ATIENT](#br12)[ ](#br12)[NBR](#br12)[:](#br12)

[**P**](#br12)[ATIENT**](#br12)[** ](#br12)[D**](#br12)[EMOGRAPHIC**](#br12)

[G](#br12)[ENDER](#br12)

[R](#br12)[ACE](#br12)

[A](#br13)[GE](#br13)

[**D**](#br13)[RUGS**](#br13)

[9](#br9)

[10](#br10)

[11](#br11)

[11](#br11)

[**12**](#br12)

[12](#br12)

[**12**](#br12)

[12](#br12)

[12](#br12)

[13](#br13)

[**13**](#br13)

[13](#br13)

[13](#br13)

[14](#br14)

[14](#br14)

[14](#br14)

[15](#br15)

[15](#br15)

[15](#br15)

[16](#br16)

[16](#br16)

[16](#br16)

[16](#br16)

[17](#br17)

[**17**](#br17)

[**17**](#br17)

[17](#br17)

[**18**](#br18)

[18](#br18)

[I](#br13)[NSULIN](#br13)

[METFORMIN](#br13)[-](#br13)[PIOGLITAZONE](#br13)[,](#br13)[ ](#br13)[GLIMEPIRIDE](#br13)[-](#br13)[PIOGLITAZONE](#br13)[,](#br13)[ ](#br13)[ACETOHEXAMIDE](#br13)

[METFORMIN](#br14)[-](#br14)[ROSIGLITAZONE](#br14)

[T](#br14)[ROGLITAZONE](#br14)

[M](#br14)[ETFORMIN](#br14)[,](#br14)[ ](#br14)[G](#br14)[LIMEPIRIDE](#br14)[,](#br14)[ ](#br14)[G](#br14)[LIPIZIDE](#br14)

[G](#br15)[LYBURIDE](#br15)[,](#br15)[ ](#br15)[C](#br15)[HLORPROPAMIDE](#br15)[,](#br15)[ ](#br15)[T](#br15)[OLBUTAMIDE](#br15)

[R](#br15)[EPAGLINIDE](#br15)[,](#br15)[ ](#br15)[N](#br15)[ATEGLINIDE](#br15)

[P](#br15)[IOGLITAZONE](#br15)

[R](#br16)[OSIGLITAZONE](#br16)

[T](#br16)[OLAZAMIDE](#br16)

[G](#br16)[LYBURIDE](#br16)[-](#br16)[METFORMIN](#br16)[,](#br16)[ ](#br16)[G](#br16)[LIPIZIDE](#br16)[-](#br16)[METFORMIN](#br16)

[A](#br16)[CARBOSE](#br16)

[M](#br17)[IGLITOL](#br17)

[**I**](#br17)[NITIAL**](#br17)[** ](#br17)[O**](#br17)[BSERVATION**](#br17)[** ](#br17)[FROM**](#br17)[** ](#br17)[ABOVE**](#br17)[** ](#br17)[FEATURES**](#br17)

[**P**](#br17)[RELIMINARY**](#br17)[** ](#br17)[T**](#br17)[EST**](#br17)[** ](#br17)[FOR**](#br17)[** ](#br17)[D**](#br17)[IABETIC**](#br17)[** ](#br17)[P**](#br17)[ATIENTS**](#br17)

[M](#br17)[AXIMUM](#br17)[ ](#br17)[G](#br17)[LUCOSE](#br17)[ ](#br17)[S](#br17)[ERUM](#br17)[ ](#br17)[AND](#br17)[ ](#br17)[A1C](#br17)[RESULT](#br17)

[**C**](#br18)[HANGE**](#br18)[** ](#br18)[IN**](#br18)[** ](#br18)[MEDICATION**](#br18)

[D](#br18)[IABETES](#br18)[ ](#br18)[M](#br18)[ED](#br18)[,](#br18)[ ](#br18)[CHANGE](#br18)

2





[**PROJECT**](#br18)[** ](#br18)[JUSTIFICATION**](#br18)

[**18**](#br18)

[**19**](#br18)

[**DATA**](#br18)[** ](#br18)[EXPLORATION**](#br18)[** ](#br18)[(EDA)**](#br18)

[**U**](#br19)[NIVARIATE**](#br19)[** ](#br19)[A**](#br19)[NALYSIS**](#br19)

[**D**](#br19)[ISTRIBUTION**](#br19)[** ](#br19)[OF**](#br19)[** ](#br19)[N**](#br19)[UMERICAL**](#br19)[** ](#br19)[V**](#br19)[ARIABLES**](#br19)

[N](#br19)[UMBER](#br19)[ ](#br19)[OF](#br19)[ ](#br19)[M](#br19)[EDICATIONS](#br19)

[N](#br20)[UMBER](#br20)[ ](#br20)[OF](#br20)[ ](#br20)[L](#br20)[AB](#br20)[ ](#br20)[T](#br20)[ESTS](#br20)

[N](#br20)[UMBER](#br20)[ ](#br20)[OF](#br20)[ ](#br20)[D](#br20)[AYS](#br20)[ ](#br20)[IN](#br20)[ ](#br20)[H](#br20)[OSPITAL](#br20)

[**19**](#br19)

[**19**](#br19)

[19](#br19)

[20](#br20)

[20](#br20)

[21](#br21)

[**21**](#br21)

[21](#br21)

[22](#br22)

[**22**](#br22)

[22](#br22)

[23](#br23)

[23](#br23)

[24](#br24)

[**24**](#br24)

[24](#br24)

[25](#br25)

[25](#br25)

[26](#br26)

[26](#br26)

[**27**](#br27)

[27](#br27)

[27](#br27)

[**28**](#br28)

[P](#br21)[ATIENT](#br21)[ ](#br21)[F](#br21)[REQUENCY](#br21)

[**D**](#br21)[ISTRIBUTION**](#br21)[** ](#br21)[OF**](#br21)[** ](#br21)[C**](#br21)[ATEGORICAL**](#br21)[** ](#br21)[V**](#br21)[ARIABLES**](#br21)

[H](#br21)[OSPITAL](#br21)[ ](#br21)[F](#br21)[ORMALITIES](#br21)[ ](#br21)[D](#br21)[ISTRIBUTION](#br21)

[D](#br22)[IAGNOSIS](#br22)[ ](#br22)[D](#br22)[ISTRIBUTION](#br22)

[**B**](#br22)[IVARIATE**](#br22)[** ](#br22)[A**](#br22)[NALYSIS**](#br22)

[I](#br22)[NSULIN](#br22)[ ](#br22)[VS](#br22)[ ](#br22)[A](#br22)[GE](#br22)

[A](#br23)[DMISSION](#br23)[ ](#br23)[T](#br23)[YPE](#br23)[ ](#br23)[VS](#br23)[ ](#br23)[N](#br23)[UMBER](#br23)[ ](#br23)[OF](#br23)[ ](#br23)[LAB](#br23)[ ](#br23)[P](#br23)[ROCEDURE](#br23)

[A](#br23)[DMISSION](#br23)[ ](#br23)[T](#br23)[YPE](#br23)[ ](#br23)[VS](#br23)[ ](#br23)[N](#br23)[UMBER](#br23)[ ](#br23)[OF](#br23)[ ](#br23)[M](#br23)[EDICATIONS](#br23)

[D](#br24)[ISCHARGE](#br24)[ ](#br24)[D](#br24)[ISPOSITION](#br24)[ ](#br24)[VS](#br24)[ ](#br24)[T](#br24)[IME](#br24)[ ](#br24)[IN](#br24)[ ](#br24)[H](#br24)[OSPITAL](#br24)

[**B**](#br24)[IVARIATE**](#br24)[** ](#br24)[A**](#br24)[NALYSIS**](#br24)[** ](#br24)[WITH**](#br24)[** ](#br24)[RESPECT**](#br24)[** ](#br24)[TO**](#br24)[** ](#br24)[TARGET**](#br24)[** ](#br24)[(**](#br24)[READMITTED**](#br24)[)**](#br24)

[T](#br24)[IME](#br24)[ ](#br24)[IN](#br24)[ ](#br24)[H](#br24)[OSPITAL](#br24)[ ](#br24)[VS](#br24)[ ](#br24)[R](#br24)[EADMITTED](#br24)

[M](#br25)[AXIMUM](#br25)[ ](#br25)[G](#br25)[LUCOSE](#br25)[ ](#br25)[S](#br25)[ERUM](#br25)[ ](#br25)[VS](#br25)[ ](#br25)[R](#br25)[EADMITTED](#br25)

[I](#br25)[NSULIN](#br25)[ ](#br25)[VS](#br25)[ ](#br25)[R](#br25)[EADMITTED](#br25)

[A1C](#br26)[RESULT](#br26)[ ](#br26)[VS](#br26)[ ](#br26)[R](#br26)[EADMITTED](#br26)

[P](#br26)[ATIENT](#br26)[ ](#br26)[F](#br26)[REQUENCY](#br26)[ ](#br26)[C](#br26)[ATEGORY](#br26)[ ](#br26)[VS](#br26)[ ](#br26)[R](#br26)[EADMITTED](#br26)

[**M**](#br27)[ULTIVARIATE**](#br27)[** ](#br27)[P**](#br27)[LOT**](#br27)

[D](#br27)[IAGNOSIS](#br27)[ ](#br27)[VS](#br27)[ ](#br27)[N](#br27)[UMBER](#br27)[ ](#br27)[OF](#br27)[ ](#br27)[M](#br27)[EDICATIONS](#br27)[ ](#br27)[VS](#br27)[ ](#br27)[R](#br27)[EADMITTED](#br27)

[D](#br27)[IAGNOSIS](#br27)[ ](#br27)[VS](#br27)[ ](#br27)[P](#br27)[ATIENT](#br27)[ ](#br27)[F](#br27)[REQUENCY](#br27)[ ](#br27)[VS](#br27)[ ](#br27)[R](#br27)[EADMITTED](#br27)

[**M**](#br28)[ULTICOLLINEARITY**](#br28)[** ](#br28)[C**](#br28)[HECK**](#br28)

[**PRELIMINARY**](#br28)[** ](#br28)[EDA**](#br28)[** ](#br28)[INSIGHTS**](#br28)

[**28**](#br28)

[**E**](#br28)[XPIRED**](#br28)[** ](#br28)[AND**](#br28)[** ](#br28)[H**](#br28)[OSPICE**](#br28)[** ](#br28)[P**](#br28)[ATIENTS**](#br28)

[**E**](#br28)[XPIRED**](#br28)[** ](#br28)[P**](#br28)[ATIENTS**](#br28)

[**R**](#br29)[ATE**](#br29)[** ](#br29)[OF**](#br29)[** ](#br29)[E**](#br29)[XPIRATION**](#br29)[** ](#br29)[WITH**](#br29)[** ](#br29)[RESPECT**](#br29)[** ](#br29)[TO**](#br29)[** ](#br29)[D**](#br29)[ISEASE**](#br29)

[**T**](#br30)[RANSFERRED**](#br30)[** ](#br30)[P**](#br30)[ATIENTS**](#br30)[** ](#br30)[WITH**](#br30)[** ](#br30)[RESPECT**](#br30)[** ](#br30)[TO**](#br30)[** ](#br30)[DISORDER**](#br30)

[**I**](#br31)[NSULIN**](#br31)[** ](#br31)[VS**](#br31)[** ](#br31)[A**](#br31)[GE**](#br31)

[**I**](#br32)[NSULIN**](#br32)[** ](#br32)[VS**](#br32)[** ](#br32)[D**](#br32)[IAGNOSIS**](#br32)

[**I**](#br32)[NSULIN**](#br32)[** ](#br32)[VS**](#br32)[** ](#br32)[P**](#br32)[ATIENT**](#br32)[** ](#br32)[F**](#br32)[REQUENCY**](#br32)

[**A**](#br33)[GE**](#br33)[** ](#br33)[VS**](#br33)[** ](#br33)[D**](#br33)[IAGNOSIS**](#br33)

[**A**](#br34)[GE**](#br34)[** ](#br34)[VS**](#br34)[** ](#br34)[D**](#br34)[ISCHARGE**](#br34)[** ](#br34)[D**](#br34)[ISPOSITION**](#br34)

[**A**](#br36)[GE**](#br36)[** ](#br36)[VS**](#br36)[** ](#br36)[P**](#br36)[ATIENT**](#br36)[** ](#br36)[F**](#br36)[REQUENCY**](#br36)

[**D**](#br37)[IAGNOSIS**](#br37)[** ](#br37)[VS**](#br37)[** ](#br37)[P**](#br37)[ATIENT**](#br37)[** ](#br37)[F**](#br37)[REQUENCY**](#br37)[** ](#br37)[C**](#br37)[ATEGORY**](#br37)

[**R**](#br39)[EADMITTED**](#br39)[** ](#br39)[VS**](#br39)[** ](#br39)[P**](#br39)[ATIENT**](#br39)[** ](#br39)[F**](#br39)[REQUENCY**](#br39)[** ](#br39)[C**](#br39)[ATEGORY**](#br39)

[**28**](#br28)

[**28**](#br28)

[**29**](#br29)

[**30**](#br30)

[**31**](#br31)

[**32**](#br32)

[**32**](#br32)

[**33**](#br33)

[**34**](#br34)

[**36**](#br36)

[**37**](#br37)

[**39**](#br39)

[**INCONSISTENCIES**](#br40)[** ](#br40)[IN**](#br40)[** ](#br40)[DATA**](#br40)

[**40**](#br40)

3





[**FEATURE**](#br40)[** ](#br40)[SELECTION**](#br40)[** ](#br40)[USING**](#br40)[** ](#br40)[FILTER**](#br40)[** ](#br40)[METHODS:**](#br40)

[**40**](#br40)

[**F**](#br40)[EATURE**](#br40)[** ](#br40)[S**](#br40)[ELECTION**](#br40)[** ](#br40)[USING**](#br40)[** ](#br40)[V**](#br40)[ARIANCE**](#br40)[** ](#br40)[T**](#br40)[HRESHOLD**](#br40)[:**](#br40)

[**F**](#br41)[EATURE**](#br41)[** ](#br41)[S**](#br41)[ELECTION**](#br41)[** ](#br41)[USING**](#br41)[** ](#br41)[S**](#br41)[TATISTICAL**](#br41)[** ](#br41)[S**](#br41)[IGNIFICANCE**](#br41)[** ](#br41)[OF**](#br41)[** ](#br41)[VARIABLES**](#br41)[** ](#br41)[(U**](#br41)[NIVARIATE**](#br41)[** ](#br41)[F**](#br41)[EATURE**](#br41)[** ](#br41)[S**](#br41)[ELECTION**](#br41)[)**](#br41)

[C](#br41)[HI](#br41)[-](#br41)[SQUARE](#br41)[ ](#br41)[TEST](#br41)

[A](#br42)[NOVA](#br42)[ ](#br42)[OR](#br42)[ ](#br42)[K](#br42)[RUSKAL](#br42)[ ](#br42)[T](#br42)[EST](#br42)

[**40**](#br40)

[**41**](#br41)

[41](#br41)

[42](#br42)

[**BASE**](#br43)[** ](#br43)[MODEL**](#br43)[** ](#br43)[PERFORMANCE**](#br43)

[**43**](#br43)

[**L**](#br43)[OGISTIC**](#br43)[** ](#br43)[R**](#br43)[EGRESSION**](#br43)

[**D**](#br43)[ECISION**](#br43)[** ](#br43)[T**](#br43)[REE**](#br43)[** ](#br43)[C**](#br43)[LASSIFIER**](#br43)

[**R**](#br43)[ANDOM**](#br43)[** ](#br43)[F**](#br43)[OREST**](#br43)[** ](#br43)[C**](#br43)[LASSIFIER**](#br43)

[**A**](#br44)[DA**](#br44)[B**](#br44)[OOST**](#br44)[** ](#br44)[C**](#br44)[LASSIFIER**](#br44)

[**G**](#br44)[RADIENT**](#br44)[** ](#br44)[B**](#br44)[OOSTING**](#br44)[** ](#br44)[C**](#br44)[LASSIFIER**](#br44)

[**M**](#br44)[ETRICS**](#br44)

[**43**](#br43)

[**43**](#br43)

[**43**](#br43)

[**44**](#br44)

[**44**](#br44)

[**44**](#br44)

[**FEATURE**](#br45)[** ](#br45)[SELECTION**](#br45)[** ](#br45)[USING**](#br45)[** ](#br45)[EMBEDDED**](#br45)[** ](#br45)[METHOD**](#br45)

[**L**](#br45)[OGISTIC**](#br45)[** ](#br45)[L**](#br45)[ASSO**](#br45)[** ](#br45)[R**](#br45)[EGRESSION**](#br45)

[**45**](#br45)

[**45**](#br45)

[**45**](#br45)

[**45**](#br45)

[**46**](#br46)

[**46**](#br46)

[**47**](#br47)

[**FEATURE**](#br45)[** ](#br45)[SELECTION**](#br45)[** ](#br45)[USING**](#br45)[** ](#br45)[WRAPPER**](#br45)[** ](#br45)[METHOD**](#br45)

[**R**](#br45)[ECURSIVE**](#br45)[** ](#br45)[F**](#br45)[EATURE**](#br45)[** ](#br45)[E**](#br45)[LIMINATION**](#br45)

[**CLASS**](#br46)[** ](#br46)[IMBALANCE**](#br46)

[**FINAL**](#br46)[** ](#br46)[BEST**](#br46)[** ](#br46)[MODEL:**](#br46)

[**BUSINESS**](#br47)[** ](#br47)[INSIGHTS**](#br47)

4





**Abstract**

➢ Hospital readmission is considered a key metric to assess health center performances.

Indeed, readmission involves different consequences such as the patient’s health

condition, hospital operational efficiency but also cost burden from a wider

perspective. Prediction of readmission for diabetes patients is of prime importance.

➢ Current practice to find at-risk diabetic patients are subjective: a clinician will assess

the patient and decide what the proper care plan is for that person. This is the current

practice which is a time-consuming process. To overcome this, we build machine

learning models which are fast.

➢ Machine Learning models are more complex but may be able to create

more correct risk predictions that should lead to improved diabetic patient outcomes.

➢ So, we are using various Machine Learning algorithms, for predicting whether the

patient will be re-admitted within 30 days or not, with the various input parameters

available.

**Keywords:** 30-day readmission, Machine Learning, diabetes

**Introduction**

➢ Diabetes is one of the chronic non-communicable diseases that are on the rise with

massive urbanization and a drastic change of lifestyle in many countries. It is

expected to turn into the seventh most prevalent mortality caused by 2030 and

millions of deaths could be prevented each year through better analytics.

➢ This study investigates the hypothesis that advanced machine learning techniques can

make use of a wide set of clinical features to improve diabetic readmission risk

prediction over simpler objective measures like LACE (Length of stay, Acuity of

admission, Charlson comorbidity index, and Emergency visits) while reducing

hospitalization cost. An existing dataset and algorithms are used to test this

hypothesis.

➢ This work covers methods to show potentially modifiable risk factors leading to

readmission rates. Machine learning identification of the likelihood of readmission is

the foundational step to understand and develop protocols for better inpatient

diabetic care.

5





**Dataset and Domain**

This study uses the Health Facts National Database (Cerner Corporation, Kansas City, MO),

gathering extensive clinical records across hundreds of hospitals throughout the US [18]. The

data subset used for analysis covers 10 years of diabetes patient encounter data (1999 – 2008)

among 130 US hospitals with over 100,000 diabetes patients. The Healthcare industry collects

and processes diabetes patient medical data in huge volumes, diverse structures, and real-

time flow of data. When assessing the quality of care delivered by a health center,

readmission is the metric of choice. It measures the number of patients that need to come

back to the hospital after their initial discharge. Hospital readmission of diabetic patients is

expensive as hospitals face penalties if their readmission rate is higher than expected and

reflects the inadequacies in the health care system. For these reasons, the hospitals need to

improve focus on reducing readmission rates. Identify the key factors that influence

readmission for diabetes and predict the probability of patient readmission.

**Variable Categorization**

There are about 101,766 rows and 50 columns in the dataset of which there are about 13

numerical columns and 37 categorical columns. The output variable is the column labeled

“readmitted” which is encoded a 3-class classifier including “<30 days”, “>30 days”, “NO”.

**Data Dictionary**

**Feature Name**

**Feature Description**

**Data Type Missing**

**Value**

**%**

Encounter Id

Patient Number

Race

Unique identifier of an encounter.

Unique identifier of a patient.

Numerical

Numerical

Categorical

0

0

2

Nominal Values: Caucasian, Asian, African

American, Hispanic, and other.

Gender

Age

Nominal Values: male, female, and

unknown/invalid.

Categorical

Categorical

0

0

Nominal Grouped in 10-year intervals.

0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-

70, 70-80, 80-90, 90-100.

Weight

Weight of a patient in pounds.

Numerical

97

0

Admission type

Id

Nominal Integer identifier corresponding to Numerical

9 distinct values.

1- Emergency, 2- Urgent, 3- Elective,

4- New-born, 5- Not Available, 6- NULL,

6





7- Trauma Center, 8- Not Mapped

Discharge

Nominal Integer identifier corresponding to Numerical

29 distinct values.

E.g., 1- Discharged to home

0

0

disposition Id

Admission

source

Nominal Integer identifier corresponding to Numerical

21 distinct values.

E.g., 1- Physician Referral

Time in hospital

Payer code

Numeric Integer number of days between

admission and discharge.

Numerical

0

Nominal Integer identifier corresponding to Numerical

23 distinct values.

53

0

Medical specialty Nominal Integer identifier of a specialty of

the admitting physician, corresponding to 84

distinct values.

Numerical

Number of lab

procedures

Number of lab tests performed during the

encounter.

Numerical

0

0

0

0

0

0

0

Number of

procedures

Number of procedures (other than lab tests) Numerical

performed during the encounter.

Number of

medications

Number of distinct generic names

administered during the encounter.

Numerical

Number of

Number of outpatient visits of the patient in Numerical

outpatient visits the year preceding the encounter.

Number of

Number of emergency visits of the patient in Numerical

emergency visits the year preceding the encounter.

Number of

inpatient visits

Number of inpatient visits of the patient in

the year preceding the encounter.

Numerical

Categorical

Diagnosis 1

Diagnosis 2

Diagnosis 3

Nominal data. Primary diagnosis (coded as

first three digits of ICD9); 848 distinct

values.

Nominal data. Secondary diagnosis (coded

as first three digits of ICD9); 923 distinct

values.

Categorical

Categorical

0

1

Nominal data. Additional secondary

diagnosis (coded as first three digits of

ICD9); 954 distinct values

Number of

diagnoses

Number of diagnoses entered to the system. Numerical

0

0

Glucose serum

test result

Nominal data. Indicates the range of the

result or if the test was not taken. Values:

“>200,” “>300,” “normal,” and “none” if not

measured

Categorical

A1c test result

Nominal data. It indicates the range of the

result or if the test was not taken. Values:

“>8” if the result was greater than 8%, “>7”

if the result was greater than 7% but less

Categorical

0

7





than 8%, “normal” if the result was less than

7%, and “none” if not measured

Change of

medications

Nominal data. It indicates if there was a

change in diabetic medications (either

dosage or generic name). Values: “change”

and “no change”

Categorical

0

Diabetes

Nominal data. It indicates if there was any

diabetic medication prescribed. Values:

“yes” and “no”

Categorical

Categorical

0

0

medications

Readmitted

Nominal data. Days to inpatient

readmission.

Values: “<30” if the patient was readmitted

in less than 30 days,

“>30” if the patient was readmitted in more

than 30 days, and

“No” for no record of readmission

24 features for medications. This is all Nominal data. For the generic names: metformin,

repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide,

tolbutamide, pioglitazone, rosiglitazone, carbose, miglitol, troglitazone, tolazamide, examide,

sitagliptin, insulin, glyburide-metformin, glipizide-metformin, glimepiride- pioglitazone,

metformin-rosiglitazone, and metformin-pioglitazone, the feature indicates whether the drug

was prescribed or there was a change in the dosage.

Values: “up” if the dosage was increased during the encounter, “down” if the dosage was

decreased, “steady” if the dosage did not change, and “no” if the drug was not prescribed.

**Data Pre-Processing**

As real-world medical data are often clumsy, a particular focus will be led on pre-processing

tasks handling missing data, duplicates, redundant columns, outliers, and data pollution

thereby able to reduce the dataset features and optimizing it for further before model

deployment.

**Null/Missing Value Detection**

➢ The first step in pre-processing the data consist of handling missing values. Missing

values refer to the absence, voluntary or not, of data in a record. While the initial step

is to identify and encode missing values, the second step consists of addressing the

missing values. Each variable comprising missing values was independently analysed,

and depending upon the percentage of the missing values we either drop the column

or impute it with values based on some statistical criteria.

➢ Our dataset features such as “weight and payer code” are having more than 50% of

missing data and hence we drop these columns.

8





➢ Medical Speciality feature also contains many missing values and no other feature

helps us in determining the missing values for Medical Speciality, So we drop the

Medical Speciality.

➢ There are three columns for diagnosis. When a patient is not diagnosed with any

illness in the first diagnosis, He or She will not go through further diagnosis, But in our

dataset there are missing values in the first diagnosis and illnesses in the second and

third diagnoses. These records are dropped.

➢ There are three entries in the gender column as “invalid”, These three records are

dropped.

**Removing Uniquely Random Valued Columns**

➢ Feature “encounter\_id” has all the records unique and hence this feature doesn’t give

much meaningful insight for our prediction. We can drop this column for further

analysis.

➢ The significance of the Feature “patient\_nbr “will be tested after model creation using

the feature\_importance parameter.

➢ In particular, we removed the “encounter ID” to avoid overfitting the model.

**Removing Degenerate Features**

➢ Two features, namely, “examide”, “citoglipton” having the same observation (“No”)

for every record in the dataset. Such features will, as a result, be dropped from the

analysis.

**Feature Engineering**

**Diagnosis-1, Diagnosis-2 Diagnosis-3**

➢ Features diag\_1, diag\_2, and diag\_3 are having numerical entries combined with

characters. Using the domain knowledge, we have mapped the ICD-9 (International

Classification of Diseases, Ninth Revision) Codes in the above columns with their

9





respective disease name. For a particular record if the value is not present i.e., Nan we

are again keeping the value as Nan. So below table shows the value for diagnosis in

the dataset.

**Group Names**

Circulatory

Respiratory

Digestive

**ICD-9 Codes**

390-459,785

460-519,786

520-579,787

250.xx

Diabetes

Injury

800-999

Musculoskeletal 710-739

Genitourinary

Neoplasms

580-629,788

➢ A patient goes through at most three diagnoses. If a patient Is not diagnosed for the

first time, then there is no need for the patient to undergo further diagnoses. So,

these rows can be removed.

➢ There are anomalies in the dataset wherein the patient has not gone through a

second diagnosis but is diagnosed with some side effects in the third one. These are

meaningless entries hence removed.

**Admission Type ID**

➢ The code indicating the type and priority of an inpatient admission associated with the

service on an intermediary submitted claim. We have encoded the code with the value.

**Code**

**Encoded Value**

Emergency

Urgent

1

2

3

Elective

4

7

Newborn

Trauma Cancer

Other

5, 6, 8

10





**Discharge Disposition ID**

➢ Discharge Disposition is the person’s anticipated location or status following the

encounter. We have 29 distinct Integer values for this and we have done feature

engineering with our domain knowledge. We have encoded the code with the value.

**Code**

11, 19, 20, 21

**Encoded Value**

Expired

7

Left AMA

9, 12

Inpatient / Stillpatient /

Outpatient

13, 14

1, 6, 8

15

Hospice

Discharged Home

Transferred Within

2, 3, 4, 5, 10, 16, 17, 22, Transferred / Referred

23, 24, 27, 28, 29

15, 18, 25

Other

**Admission Source ID**

➢ The code indicating the source of the referral for the admission or visit. We have 25

distinct Integer values for this and we have done feature engineering with our domain

knowledge. We have encoded the code with the value.

**Code**

**Encoded Value**

Referral

1, 2, 3

4, 5, 6, 10, 18, 22, 25, 26

Transfer

7

Emergency Room

Law Enforcement

Normal Delivery

Premature Delivery

Sick Baby

8

11

12

13

14

Extramural Birth

Readmission

19

23

Born In This Hospital

Born Elsewhere

Other

24

9, 15, 17, 21

11





**Feature Extraction**

**Patient nbr:**

➢ Patient nbr gives the information about the assigned random numbers for each

patient. Using this we have mapped the frequency of each patient in terms of a

number of visits as patient\_frequency.

➢ We have extracted one more feature patient\_frequency\_categorized in which we

have categorized patients based on the number of visits (1-time, 2-5 times, 6-10

times, more than 10 times).

**Patient Demographic**

**Gender**

➢ The gender column has 3 different values. From the data, it is clear that most of the

patients are Females. There is one more value named Unknown/Invalid which can be

removed because it consists of only 3 records.

**Race**

➢ The below screenshot shows the distribution of Race. The majority of the patients are

Caucasian.

12





**Age**

➢ The below screenshot shows the number of patients in various groups.

**Drugs**

**Insulin**

➢ Insulin is a hormone made in your pancreas; a gland located behind your stomach. It

allows your body to use glucose for energy. Glucose is a type of sugar found in many

carbohydrates.

➢ Diabetes occurs when your body doesn’t use insulin properly or doesn’t make enough

insulin.

**metformin-pioglitazone, glimepiride-pioglitazone, acetohexamide**

➢ Metformin/pioglitazone is used to improve blood sugar control in adults with type 2

diabetes.

➢ This medication is a combination of 2 drugs, pioglitazone and glimepiride. It is used

along with a proper diet and exercise program to control high blood sugar in patients

with type 2 diabetes.

➢ Acetohexamide is a first-generation sulfonylurea medication used to treat diabetes

mellitus type 2, particularly in people whose diabetes cannot be controlled by diet

alone.

All the above features in our dataset have value “Steady” for one record and all the

remaining records have value “No”.

13





**metformin-rosiglitazone**

➢ Rosiglitazone and metformin combination is used to treat a type of diabetes mellitus

called type 2 diabetes. It is used together with a proper diet and exercise to help

control blood sugar levels.

In our dataset this feature has a value “Steady” for two record and all the other records have

a value “No”.

**Troglitazone**

➢ Troglitazone is the first of a new group of oral antidiabetic drugs, the

thiazolidinediones, and is indicated for the treatment of patients with type 2 (non-

insulin-dependent) diabetes mellitus.

➢ In our dataset this feature has a value “Steady” for two record and all the other

records have a value “No”.

**Metformin, Glimepiride, Glipizide**

➢ Metformin/ Glimepiride/ Glipizide is used with a proper diet and exercise program and

possibly with other medications to control high blood sugar. It is used in patients with

type 2 diabetes.

14





**Glyburide, Chlorpropamide, Tolbutamide**

➢ Glyburide/ Chlorpropamide/ Tolbutamide is used in the treatment of non-insulin

dependent diabetes mellitus. It used to treat patients with diabetes type II.

**Repaglinide, Nateglinide**

➢ Repaglinide/ Natglinide is used alone or with other medications to control high blood

sugar along with a proper diet and exercise program. It is used in people with type 2

diabetes.

**Pioglitazone**

➢ Pioglitazone is a diabetes drug (thiazolidinedione-type, also called "glitazones") used

along with a proper diet and exercise program to control high blood sugar in patients

with type 2 diabetes.

15





**Rosiglitazone**

➢ Rosiglitazone is an insulin sensitizing agent and thiazolidinedione that is indicated for

the treatment of type 2 diabetes.

**Tolazamide**

➢ Tolazamide is an oral blood glucose lowering drug of the sulfonylurea class.

Tolazamide appears to lower the blood glucose acutely by stimulating the release of

insulin from the pancreas. It is used in the treatment of type 2 diabetes.

**Glyburide-metformin, Glipizide-metformin**

➢ This combination medication is used with a proper diet and exercise program to

control high blood sugar in people with type 2 diabetes.

**Acarbose**

➢ Acarbose is an alpha glucosidase inhibitor which decreases intestinal absorption of

carbohydrates and is used as an adjunctive therapy in the management of type 2

diabetes.

16





**Miglitol**

➢ Miglitol is an oral alpha-glucosidase inhibitor used to improve glycaemic control by

delaying the digestion of carbohydrates.

**Initial Observation from above features**

➢ From the above one, we can infer that many patients are not diagnosed with the

drugs.

**Preliminary Test for Diabetic Patients**

**Maximum Glucose Serum and A1Cresult**

➢ The A1C test measures the percentage of your red blood cells that have sugar-

coated haemoglobin.

➢ A blood glucose test is a blood test that screens for diabetes by measuring the level

of glucose (sugar) in a person’s blood.

➢ Features max\_glu\_serum and A1Cresult have a value None for a majority of the

records. Since these are very significant ones for determining the readmitted status,

we are keeping this feature.

17





**Change in medication**

➢ The feature change is either dosage or generic name. We do not have any feature that

relates to the feature “change”.

➢ The feature diabetes med signifies if atleast one of the diabetic drug was provided to

the patient or not.

**Diabetes Med, change**

**Project Justification**

➢ The dataset that we are going to use is the real-time dataset which was collected from

Health Facts National Database, US.

➢ This is a Multi-Class Classification problem. The dependent variable is **Readmission**.

➢ We can use Classification model algorithms like Multinomial Logistic Regression, Gradient

Boosting, Ada Boost, XG Boost, Stacking Classifier, Decision Tree and Random Forest. We

use these techniques to improve the accuracy and performance of the model.

18





**Data Exploration (EDA)**

**Univariate Analysis**

**Distribution of Numerical Variables**

From the below table we could say that the features number\_emergency and

number\_outpatient are highly right-skewed.

**Number of Medications**

➢ From the below plot, it is evident that an average of 13-15 distinct drugs has been administered

per visit.

19





**Number of Lab Tests**

➢ This feature tells us about the number of lab tests performed during the encounter.

➢ From the below plot we could say that on average 43 lab tests are taken on a patient,

except for 1.

**Number of Days in Hospital**

➢ From the below plot it could be evident that on average patients are admitted to the

hospital for nearly 3-4 days.

20





**Patient Frequency**

➢ Most of the patients visit the hospital only once. On average, patients are likely to visit

the hospital twice. There are few exceptions where patients visited the hospital nearly

38 times.

**Distribution of Categorical Variables**

**Hospital Formalities Distribution**

➢ From the below plot, Fig-1 shows the distribution of different admission types. So,

from figure - 1 we could say that most of the patients are admitted in either

Emergency or Urgent conditions. Around 20,000 patients are admitted with prior

formalities verified (Elective).

➢ Fig-2 shows the distribution of different discharge disposition. Discharge dispositions

gives the detail on whether the patient was discharged home after treatment or was

transferred to other hospitals for various reasons. A considerably good number of

patients were discharged back home.

➢ Fig-3 shows the distribution of different admission sources. It is evident from the

graph that many patients are admitted to the Emergency room.

21





**Diagnosis Distribution**

➢ From the below 3 graphs it is evident that many people are prone to diabetes disorder.

**Bivariate Analysis**

**Insulin vs Age**

➢ The below plot shows how the count of insulin intake varies concerning different age

groups.

22





**Admission Type vs Number of lab Procedure**

➢ From the below graph we could say that the average number of lab procedures

performed for the patients in Emergency room is more than those patients who are

admitted in another type.

**Admission Type vs Number of Medications**

➢ From the below graph we could say that the average number of medications

performed for the patients in Elective is more than those patients who are admitted

in another type.

23





**Discharge Disposition vs Time in Hospital**

➢ Before the patients are transferred to other facilities or within or sent to hospice, they

spend more time in the hospital concerning the rest.

**Bivariate Analysis with respect to target (readmitted)**

**Time in Hospital vs Readmitted**

➢ The below box plot shows the relationship between various readmission factors and

the number of days the patient was admitted to the hospital.

➢ From the below plot it is evident that readmitted patients within 30 days spend

more time in hospital than those patients who are not readmitted, but the

difference between them is only a small margin.

24





**Maximum Glucose Serum vs Readmitted**

➢ From the below plot we can infer that the patients whose glucose serum level is >300

are at a higher risk of being readmitted.

**Insulin vs Readmitted**

➢ The below plots show the how readmitted factor is related to insulin. From that, we

could say that people whose insulin doses were not administered are more likely to

get re-admitted.

25





**A1Cresult vs Readmitted**

➢ Most of the patients have A1C results of more than 6 or7.

**Patient Frequency Category vs Readmitted**

➢ From the below plot we can infer those patients who visit the hospital more than once

are more likely to get readmitted.

26





**Multivariate Plot**

**Diagnosis vs Number of Medications vs Readmitted**

**Diagnosis vs Patient Frequency vs Readmitted**

27





**Multicollinearity Check**

➢ In our dataset, there are 19 numerical variables. We need to check the

multicollinearity between them. From the below plot we could say that a maximum

correlation value of 0.59 between patient frequency and the number of inpatients.

**Preliminary EDA Insights**

**Expired and Hospice Patients**

➢ In the dataset of diabetes, 264 patients are expired and 96 patients are Hospice.

**Expired Patients**

➢ From the below plot we can conclude that many patients expired have circulatory

and Respiratory disorders. But the ratio of circulatory and respiratory patients is

considerably high.

28





**Rate of Expiration with respect to Disease**

➢ The below plot shows the ratio of deceased people due to a specific disorder to the

number of patients admitted in the hospital with the specific disorder.

➢ From the below plot we could say that on average 4% of patients suffering from

Neoplasm may expire.

29





**Transferred Patients with respect to disorder**

➢ Many patients who were transferred had a circulatory disorder and other disorders

with diabetes.

➢ From the below plot we could say that the rate of transfer of patients who were

injured and the patients who had musculoskeletal disorder are high.

30





**Insulin vs Age**

➢ From the above plot, we can infer that many children less than the age of 10 suffer

from diabetes and a steady dose of insulin was administered to many of them.

31





**Insulin vs Diagnosis**

**Insulin vs Patient Frequency**

➢ From the above, we can say that, although few patients have visited the hospital more

than 10 times they were not administered with insulin.

32





**Age vs Diagnosis**

➢ From the above, we can infer that people in the age group of 1-20 are more prone to

diabetes than any other disease.

➢ At a young age, patients are more prone to diabetes. As age increases the diabetic

patients are also prone to circulatory disorders. Until the age of 50 patients is more

prone to diabetes.

33





➢ From the above, we can say that until the age of 50 patients is more prone to diabetes.

**Age vs Discharge Disposition**

34





➢ From the above plot, we can infer that many of the in-patients are in the age group of

10-20 and 60-70.

➢ From the above plot, we can infer that as age increases the Transfer rate of the patient

also increases.

35





**Age vs Patient Frequency**

➢ From the above plot, we can infer those patients between the age of 20-50 tend to

visit the hospital more often.

36





➢ From the above plot, we could say that many patients in all age groups visit the

hospital once.

**Diagnosis vs Patient Frequency Category**

➢ From the above one, we can say that circulatory and respiratory disorder patients tend

to visit the hospital most frequently.

37





➢ From the above one, we could say that patients with respiratory and other illnesses

tend to visit the hospital often.

38





**Readmitted vs Patient Frequency Category**

➢ From the above, we could say that nearly 4200 patients were readmitted in the first

instance itself.

➢ From the above, we could say that patients who visit the hospital more than once are

more likely to be readmitted.

39





**Inconsistencies in data**

\1. Few of the records in diagnoses 1,2 and 3 had Nan values. All the records in

diagnosis-1 which had Nan values are removed. The records where diagnosis 2 was

not specified but diagnosis 3 was specified were dropped because the patient will

not be diagnosed further without being affected by a particular illness.

\2. Few patients have expired and none of them was readmitted hence those patients can

be dropped before our further analysis because they won’t contribute to the

readmission.

\3. Many patients who visited the hospital for the first time were readmitted. This

doesn’t make the data meaningful. Assuming those records were not captured in the

dataset we are moving ahead with the flaws.

\4. There are anomalies in the dataset wherein the patient has not gone through a

second diagnosis but is diagnosed with some side effects in the third one. These are

meaningless entries hence removed.

**Feature Selection using Filter Methods:**

**Feature Selection using Variance Threshold:**

➢ Selecting features with a variance threshold of 99.8%. Below screenshot shows the

feature which is not satisfying the above condition.

40





**Feature Selection using Statistical Significance of variables (Univariate**

**Feature Selection)**

We want to analyze the variables in this dataset to understand any relationships between

them and their overall effects.

To do this, we will perform either

\* `Chi-square test` for categorical variables relationship.

\* `Analysis Of Variance` or `ANOVA test` for categorical variable and Numerical variable

relationship.

The purpose of these tests is to determine whether there is a statistically significant

relationship between the target variable and independent variables.

The significance level or α is a measure of the strength of the evidence that must be present

in your sample before you will reject the null hypothesis and conclude that the effect is

statistically significant. Here we are considering alpha=0.05 for our Hypothesis testing.

**Chi-square test**

➢ We are performing the chi-square test between the independent categorical variable

with the dependent variable. For the input variables ['race', 'gender', 'age', 'diag\_1',

'diag\_2', 'diag\_3', 'max\_glu\_serum', 'A1Cresult', 'change', 'diabetesMed'] we are doing

the test with the readmission column.

➢ From the below screenshot, it is evident for all the variables the p-value is less than

0.05. Hence, we can conclude that all the variables are significant i.e., there is a

relationship between those variables (they will help predict the readmission).

➢ Now we will check the relationship between the medications (i.e., drugs) with the

target variable i.e., readmission. The list of drugs we are going to consider are

['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide',

'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',

'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',

'metformin-pioglitazone'].

41





➢ From the below screenshot we could say that few of the drugs do not help predict the

readmission rate the p-value is greater than 0.05, which means they failed to reject

the null hypothesis which indirectly tells us that there is no strong influence of them

on readmission rate i.e., readmission target variable.

**Anova or Kruskal Test**

➢ We are performing either Anova or Kruskal test between the independent

numerical variable and the response variable. The numerical variables we are

considering are [‘time\_in\_hospital’, ’num\_lab\_procedures’, ’num\_procedures’,

’num\_medications’,’number\_outpatients’,’number\_emergency’,’number\_inpatie

nt’ , ’number\_diagnose].

➢ From the test results, we could say that all the variables have a p-value less than

0.05, which means they rejected the null hypothesis which indirectly tells us that

there is a strong influence of them on readmission rate i.e., readmission target

variable.

42





**Base Model Performance**

**Logistic Regression**

**Decision Tree Classifier**

**Random Forest Classifier**

43





**AdaBoost Classifier**

**Gradient Boosting Classifier**

**Metrics**

➢ From the above models we could say that Random Forest and Gradient Boosting

classifier are performing well both in terms of accuracy as well as f1\_score.

➢ Since it is an imbalanced multiclass classification, our main focus is to predict the more

vulnerable patients (i.e., whose readmission value is ‘< 30’ days.) hence we consider

the F1 score as the metric.

44





**Feature Selection using Embedded method**

**Logistic Lasso Regression**

➢ We have kept a penalty of l1 which is Lasso and solver as saga.

➢ We kept the magnitude of penalty C=0.015 which selected 27 features, using which

we built logistic regression model which gave an accuracy of 0.64 and f1\_score of

0.44.

**Feature Selection using Wrapper method**

**Recursive Feature Elimination**

➢ Using Recursive Feature Elimination (RFE) we have eliminated nearly 33 features,

using which we built a Random Forest Classifier which gave an accuracy of 0.66 and

f1\_score of 0.47.

45





**Class Imbalance**

➢ In our dataset, the readmission (i.e., Target variable Readmission) is almost equally

distributed, and hence there is no need for any imbalance treatment to be done on

our data.

**Final Best model:**

➢ Depending on the business need i.e., if business wants results from a realistic data, we

will go with the gradient boosting classifier whose features are selected using Random

Feature Elimination method.

➢ Since there is a class imbalance, we have also done a SMOTE analysis where Random

Forest classifier model is the best performing one.

46





**Business Insights**

➢ Our work suggests that applying a machine learning approach to a larger feature set as

well as novel approaches to model diversity and model blending can improve on simpler

readmission models such as LACE, potentially improving patient outcomes and lowering

inpatient costs to hospitals.

➢ 30-day hospital readmission of diabetes patients is of prime importance for health centres

and is found very stressful due to the current model’s limit in terms of performance and

generalizability. To cope with this challenge, this study implemented a comprehensive

pre-processing framework to improve the initial data quality, hence empowering the

model’s efficiency. The suggested pre-processing framework included comprehensive

data cleaning, data reduction, and transformation aiming at better optimizing and

selecting prominent features for 30-day unplanned readmission among diabetes patients.

Random Forest algorithm for feature selection and SMOTE algorithm for data balancing

are some examples of methods during pre-processing.

➢ This in turn helps in prioritizing patients for treatment and also provides an opportunity

for the hospital to optimize their resources well before the patient gets readmitted.

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

47

