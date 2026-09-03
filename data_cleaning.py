import pandas as pd
df=pd.read_csv("D:/ds project/glassdoor_jobs.csv")

#salary parsing
#company name text only
#state field
#age of company
#parsing of job description(python,etc.)

#1 salary parsing
df['hourly']=df['Salary Estimate'].apply(lambda x:1 if 'per hour' in x.lower() else 0) #returns 1 if there is per hour in salary estimate column
df['employer_provided']=df['Salary Estimate'].apply(lambda x:1 if 'employer provided salary:' in x.lower() else 0) #returns 1 if there is employer provided salary in salary estimate column

df= df[df['Salary Estimate']!='-1'] #removes all the rows of column salary estimate which have value -1.
salary=df['Salary Estimate'].apply(lambda x:x.split('(')[0]) #removes all the text of the row after '(' symbol.
minus_Kdollar=salary.apply(lambda x:x.replace('K','').replace('$','')) #removes K and $ sign from row

min_hr=minus_Kdollar.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df['min_salary']=min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary']=min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary']=(df.min_salary+df.max_salary)/2


#2 company name text only
df['company_txt']=df.apply(lambda x: x['Company Name'] if x['Rating']<0 else x['Company Name'][:-3],axis=1)


#3 state field
df['job_state']=df['Location'].apply(lambda x: x.split(',')[1])
# print(df.job_state.value_counts())
df['same_state']=df.apply(lambda x: 1 if x.Location==x.Headquarters else 0,axis=1)


#4 age of company
df['age']=df.Founded.apply(lambda x:x if x<1 else 2026-x)


#5 parsing of job description(python,etc.)

#python
df['python_yn']=df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
# print(df.python_yn.value_counts())

#R studio
df['R_yn']=df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
# print(df.R_yn.value_counts())

#spark
df['spark_yn']=df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
# print(df.spark_yn.value_counts())

#aws
df['aws_yn']=df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
# print(df.aws_yn.value_counts())

#excel
df['excel_yn']=df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
# print(df.excel_yn.value_counts())

df_out=df.drop(['Unnamed: 0'],axis=1)

df_out.to_csv('salary_data_cleaned.csv',index=False)