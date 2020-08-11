import pandas as pd
import numpy as np

df=pd.read_csv("glassdoor_jobs.csv")
#salary estimate cleaning
df["hourly"]=df["Salary Estimate"].apply(lambda x: 1 if "per hour" in x.lower() else 0)
df["emp_provided"]=df["Salary Estimate"].apply(lambda x: 1 if "employer provided salary:" in x.lower() else 0)
df["Salary Estimate"]=df["Salary Estimate"].apply(lambda x: x.lower().replace("per hour",'').replace("employer provided salary:","").replace("(employer",""))
df=df[df["Salary Estimate"]!="-1"]
df["Salary Estimate"]=df["Salary Estimate"].apply(lambda x: x.split(" ")[0])
df["Salary Estimate"]=df["Salary Estimate"].apply(lambda x: x.lower().replace("$",'').replace("k",""))
df["min_sal"]=df["Salary Estimate"].apply(lambda x: int(x.split("-")[0]))
df["max_sal"]=df["Salary Estimate"].apply(lambda x: int(x.split("-")[1]))
df["avg_sal"]=(df["min_sal"]+df["max_sal"])/2

#company name cleaning
df["company_text"]=df.apply(lambda x: x["Company Name"] if x["Rating"]<0 else x["Company Name"][0:-3],axis=1)

#state 
df["job_state"]=df.Location.apply(lambda x: x.split(",")[1])
df["same_loc_head"]=df.apply(lambda x: 1 if x["Location"]==x["Headquarters"] else 0,axis=1)
#year establishment
df['Years_of_estd']=df["Founded"].apply(lambda x: 2020-x if x>0 else x)
#companies with specific skill requirements
#python
df["Job Description"][1]
df["python"]=df["Job Description"].apply(lambda x: 1 if "python" in x.lower() else 0)
#excel
df["excel"]=df["Job Description"].apply(lambda x: 1 if "excel" in x.lower() else 0)
#spark
df["spark"]=df["Job Description"].apply(lambda x: 1 if "spark" in x.lower() else 0)
#tableau
df["tableau"]=df["Job Description"].apply(lambda x: 1 if "tableau" in x.lower() else 0)
#r studio
df["rstudio"]=df["Job Description"].apply(lambda x: 1 if "r studio" in x.lower() or "r-studio" in x.lower() else 0)
#sas
df["sas"]=df["Job Description"].apply(lambda x: 1 if "sas" in x.lower() else 0)
df.sas.value_counts()


#lets drop unnecessary columns
df.columns
df_out=df.drop(["Unnamed: 0"],axis=1)
df_out.to_csv("cleaned_data",index=False)

