#!/usr/bin/env python
# coding: utf-8

# Loading necessary Liabraries 

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt


# Reading the dataset for analysis 

# In[2]:


df= pd.read_csv('YourCabs_training.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


d=df.drop(['user_id','from_long' , 'from_area_id' , 'from_lat' , 'from_date' ,'from_city_id',  'to_lat' , 'to_area_id' , 'to_city_id' , 'to_date'  , 'to_long' , 'package_id'],axis=1)


# In[9]:


corrmat = d.corr()
k = 20
cols = corrmat.nlargest(k, 'Car_Cancellation')['Car_Cancellation'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Looking for Outliers and Null values in the data set and filling them with Median 

# In[10]:


df['vehicle_model_id'].value_counts(normalize=True)*100


# In[11]:


df['travel_type_id'].value_counts(normalize=True)*100


# In[12]:


df['online_booking'].value_counts(normalize=True)*100


# In[13]:


df['mobile_site_booking'].value_counts(normalize=True)*100


# In[14]:


df


# In[15]:


df


# In[16]:


df.isna().sum()


# In[17]:


df


# In[18]:


df['package_id'].describe()


# In[19]:


df['package_id'].value_counts(normalize=True)*100


# In[20]:


df.drop(df[df['package_id']>7].index,axis=0,inplace=True)


# In[21]:


df.fillna(df['package_id'].median(), inplace=True)


# In[22]:


df


# In[23]:


df['to_area_id'].describe()


# In[24]:


df.fillna(df['to_area_id'].median(), inplace=True)


# In[25]:


df['from_area_id'].describe()


# In[26]:


df.fillna(df['from_area_id'].median(), inplace=True)


# In[27]:


df['from_city_id'].describe()


# In[28]:


df.fillna(df['from_city_id'].median(), inplace=True)


# In[29]:


df['to_city_id'].describe()


# In[30]:


df.fillna(df['to_city_id'].median(), inplace=True)


# In[31]:


df['to_date'].describe()


# In[32]:


df.fillna(df['to_date'].median(), inplace=True)


# In[33]:


print(df[df['from_lat'] == df['from_lat'].median()]['from_area_id'].max())
print(df['from_lat'].median())
print(df['from_long'].median())


# In[34]:


df[df['from_lat'] == df['from_lat'].median()]['from_area_id']


# In[35]:


df[df['from_lat'] == df['from_lat'].median()].shape


# In[36]:


df['from_lat'].fillna(df['from_lat'].median(), inplace=True)


# In[37]:


df['from_long'].fillna(df['from_long'].median(), inplace=True)


# In[38]:


df.info()


# In[39]:


df.isnull().sum()


# Extracting Time , Date ,Month , Weekday and Booking Created time from data set

# In[40]:


df['from_date'] = pd.to_datetime(df['from_date']).dt.strftime('%m/%d/%Y')
df['from_time_tm'] = pd.to_datetime(df['from_date']).dt.strftime('%H:%M')
df['booking_created_date'] = pd.to_datetime(df['booking_created']).dt.strftime('%m/%d/%Y')
df['booking_created_time'] = pd.to_datetime(df['booking_created']).dt.strftime('%H:%M')


# In[41]:


df['from_date_day'] = pd.to_datetime(df['from_date']).dt.day_name()
df['booking_created_day'] = pd.to_datetime(df['booking_created_date']).dt.day_name()
df['from_date_month'] = pd.to_datetime(df['from_date']).dt.month_name()
df['booking_created_month'] = pd.to_datetime(df['booking_created_date']).dt.month_name()
df['from_date_week'] = np.where((df['from_date_day']=='Saturday') | (df['from_date_day']=='Sunday'),'Weekend','Weekday')
df['booking_created_week'] = np.where((df['booking_created_day']=='Saturday') | (df['booking_created_day']=='Sunday'),'Weekend','Weekday')


# In[42]:


cond = [(pd.to_datetime(df['from_time_tm']).dt.hour.between(5, 8)),
        (pd.to_datetime(df['from_time_tm']).dt.hour.between(9, 12)),
        (pd.to_datetime(df['from_time_tm']).dt.hour.between(13, 16)),
        (pd.to_datetime(df['from_time_tm']).dt.hour.between(17, 20)),
        ((pd.to_datetime(df['from_time_tm']).dt.hour.between(21, 24)) | (pd.to_datetime(df['from_time_tm']).dt.hour==0)),
        (pd.to_datetime(df['from_time_tm']).dt.hour.between(1, 4))]
values = ['Early Morning','Morning','Afternoon','Evening','Night','Late Night']
df['from_date_session'] = np.select(cond,values)


# In[43]:



cond = [(pd.to_datetime(df['booking_created_time']).dt.hour.between(5, 8)),
        (pd.to_datetime(df['booking_created_time']).dt.hour.between(9, 12)),
        (pd.to_datetime(df['booking_created_time']).dt.hour.between(13, 16)),
        (pd.to_datetime(df['booking_created_time']).dt.hour.between(17, 20)),
        ((pd.to_datetime(df['booking_created_time']).dt.hour.between(21, 24)) | (pd.to_datetime(df['booking_created_time']).dt.hour==0)),
        (pd.to_datetime(df['booking_created_time']).dt.hour.between(1, 4))]
values = ['Early Morning','Morning','Afternoon','Evening','Night','Late Night']
df['booking_created_session'] = np.select(cond,values)


# In[44]:


df.info()


# In[45]:


df.head()


# In[46]:


df['time_difference'] = (pd.to_datetime(df['from_date']) - pd.to_datetime(df['booking_created'])).astype('timedelta64[m]')


# In[47]:


df


# In[48]:


df[df['time_difference'] < 0]['time_difference'].count()


# In[49]:


df['package_id'].value_counts(normalize=True)*100


# In[50]:


df


# Droping few Columns and storing data into a new Variable 

# In[51]:


data=df.drop(['from_date_day', 'from_date_month', 'from_date_week', 'from_date_session','from_date' , 'from_time_tm' , ],axis=1)


# In[52]:


data


# Extracting and Transforming Latitude and Longitude 

# In[53]:


data["from_lat_long"] = list(zip(data["from_lat"], data["from_long"]))


# In[54]:


data["to_lat_long"] = list(zip(data["to_lat"], data["to_long"]))


# In[55]:


data


# Calculating Distance travelled by the car 

# In[56]:


import math

def distance(from_lat_long, to_lat_long):
    lat1, lon1 = from_lat_long
    lat2, lon2 = to_lat_long
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


# In[57]:


data['distance_travelled'] = data.apply(lambda row: distance(row['from_lat_long'],row['to_lat_long']),axis=1)


# In[58]:


data


# Transforming From area id and To area id columns for analysis 

# In[59]:


data['from_area_id'] = round(data.groupby('from_area_id')['Car_Cancellation'].sum()/data.groupby('from_area_id')['Car_Cancellation'].count(),2)
data['from_area_id'].replace(np.nan,0,inplace=True)


# In[60]:


cond = [(data['from_area_id'].astype('float').between(0,0.33)),
        (data['from_area_id'].astype('float').between(0.34,0.66)),
        (data['from_area_id'].astype('float').between(0.67,1.0))]
values = ['Low Cancellation','Medium Cancellation','High Cancellation']
data['from_area_id'] = np.select(cond,values)


# In[61]:


data['to_area_id'] = round(data.groupby('to_area_id')['Car_Cancellation'].sum()/data.groupby('to_area_id')['Car_Cancellation'].count(),2)
data['to_area_id'].replace(np.nan,0,inplace=True)


# In[62]:


cond = [(data['to_area_id'].astype('float').between(0,0.33)),
        (data['to_area_id'].astype('float').between(0.34,0.66)),
        (data['to_area_id'].astype('float').between(0.67,1.0))]
values = ['Low Cancellation','Medium Cancellation','High Cancellation']
data['to_area_id'] = np.select(cond,values)


# In[63]:


data


# In[64]:


data=data.drop(['id', 'user_id', 'vehicle_model_id', 'to_date'  ],axis=1)


# In[65]:


data


# In[66]:


data.info()


# In[67]:


data['from_area_id'].value_counts()


# Corelation Check for the full Dataset

# In[68]:


corrmat = data.corr()
k = 240
cols = corrmat.nlargest(k, 'Car_Cancellation')['Car_Cancellation'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Droping columns for analysis

# In[69]:


data=data.drop(['to_city_id', 'from_city_id', 'from_lat','from_long' , 'to_lat' ,'to_long','time_difference' ],axis=1)


# In[70]:


data


# In[71]:


corrmat = data.corr()
k = 240
cols = corrmat.nlargest(k, 'Car_Cancellation')['Car_Cancellation'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Dropping columns with corelation 

# In[72]:


data=data.drop(['travel_type_id', 'booking_created','booking_created_time', "booking_created_date",'from_lat_long', 'to_lat_long', 'distance_travelled'] , axis=1)


# In[73]:


data


# In[74]:


data.info()


# In[75]:


corrmat = data.corr()
k = 240
cols = corrmat.nlargest(k, 'Car_Cancellation')['Car_Cancellation'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[77]:


data


# In[78]:


data=data.drop(['from_area_id','to_area_id' ] , axis=1)


# In[79]:


data


# Preprocessing Data For Machine Learning 

# In[80]:


from sklearn import preprocessing


# In[93]:


def preprocessor(data):
    res_data = data.copy()
    le = preprocessing.LabelEncoder()
    
   
    res_data['booking_created_day'] = le.fit_transform(res_data['booking_created_day'])
    res_data['booking_created_month'] = le.fit_transform(res_data['booking_created_month'])
    res_data['booking_created_week'] = le.fit_transform(res_data['booking_created_week'])
    res_data['booking_created_session'] = le.fit_transform(res_data['booking_created_session'])
    res_data['package_id'] = le.fit_transform(res_data['package_id'])
    res_data['mobile_site_booking'] = le.fit_transform(res_data['mobile_site_booking'])
    res_data['online_booking'] = le.fit_transform(res_data['online_booking'])



    return res_data


# In[94]:


data=preprocessor(data)


# In[95]:


data


# In[96]:


feature_cols = ['package_id','online_booking','mobile_site_booking','booking_created_day','booking_created_month','booking_created_session','booking_created_week']
x = data[feature_cols]
y = data.Car_Cancellation


# In[97]:


from sklearn.preprocessing import StandardScaler


# In[98]:


scaler = StandardScaler()


# In[99]:


scaled = scaler.fit_transform(x)


# In[100]:


from sklearn.model_selection import train_test_split


# In[101]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,train_size=0.75, stratify=y)


# In[102]:


get_ipython().system('pip install imblearn')

from imblearn.over_sampling import RandomOverSampler

over=RandomOverSampler()

x_train_over, y_train_over = over.fit_resample(x_train, y_train)



# In[103]:


x_train


# In[104]:


x_test


# In[105]:


y_train


# In[106]:


y_test


# Model Bulinding and Evaluation 

# In[107]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=4)
lr.fit(x_train_over,y_train_over)


# In[108]:


y_pred_lr=lr.predict(x_test)


# In[109]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[110]:


print('Logistic Regression Metrics')
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr))
print('Precision:', metrics.precision_score(y_test, y_pred_lr))
print('Recall:', metrics.recall_score(y_test, y_pred_lr))
print('f1_score:', metrics.f1_score(y_test, y_pred_lr))


# In[111]:


metrics.plot_confusion_matrix(lr,x_test,y_test)


# In[112]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


# In[113]:


from sklearn.model_selection import cross_val_score


# In[114]:


def run_cross_validation_on_trees(x, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, x, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(x, y).score(x, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores


# In[115]:


def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()


# In[116]:


sm_tree_depths = range(1,20)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(x_train_over,y_train_over, sm_tree_depths)


# In[117]:



plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree depth on training data')


# In[118]:



idx_max = sm_cv_scores_mean.argmax()
sm_best_tree_depth = sm_tree_depths[idx_max]
sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
print('The depth{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
      sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))


# In[119]:


model_dt= DecisionTreeClassifier(random_state=42,max_depth=19)

model_dt.fit(x_train_over,y_train_over)

model_dt_score_tarin = model_dt.score(x_train_over,y_train_over)

model_dt_score_test = model_dt.score(x_test,y_test)

print('Trining Score',model_dt_score_tarin)

print('Testing Score',model_dt_score_test)


# In[120]:


from io import StringIO
get_ipython().run_line_magic('pip', 'install pydotplus')
get_ipython().run_line_magic('conda', 'install graphviz')


# In[121]:


from IPython.display import Image  
from sklearn.tree import export_graphviz


# In[122]:


from six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus


# In[123]:


plt.figure(figsize=(10,8))
dot_data = StringIO()
export_graphviz(model_dt,out_file=dot_data,
               filled=True,rounded=True,
               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini',random_state=4)


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_dist = {'max_depth': [3, 5, 6, 7,9,11], 'min_samples_split': [50, 100, 150, 200, 250]}
gscv_dtc = GridSearchCV(dtc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_dtc.fit(x_train_over,y_train_over)


# In[ ]:


gscv_dtc.best_params_


# In[ ]:


dtc=DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=11,min_samples_split=50)
dtc.fit(x_train,y_train)
dtc_score_tarin = dtc.score(x_train_over,y_train_over)

dtc_score_test = dtc.score(x_test,y_test)

print('Trining Score',dtc_score_tarin)

print('Testing Score',dtc_score_test)


# In[125]:


y_pred_dtc=dtc.predict(x_test)


# In[ ]:


print('Decision Tree Metrics')
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_dtc))
print('Precision:', metrics.precision_score(y_test, y_pred_dtc))
print('Recall:', metrics.recall_score(y_test, y_pred_dtc))
print('f1_score:', metrics.f1_score(y_test, y_pred_dtc))


# In[ ]:


metrics.plot_confusion_matrix(dtc,x_test,y_test)


# In[ ]:


fpr_dt,tpr_dt,_=roc_curve(y_test,y_pred_dtc)
roc_auc_dt = auc(fpr_dt,tpr_dt)






plt.figure(1)
lw=2
plt.plot(fpr_dt,tpr_dt,color='orange',lw=lw,label='Decision Tree(AUC = %0.2f)'%roc_auc_dt)
plt.plot([0,1],[0,1],color='blue',lw=lw,linestyle='--')

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("area Under the Curve")
# plt.legend(loc="upper left")
plt.legend(loc="lower right")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini',random_state=42)


# In[ ]:


param_dist = {'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 200, 250]}
gscv_rfc = GridSearchCV(rfc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_rfc.fit(x_train_over, y_train_over)


# In[ ]:


gscv_rfc.best_params_


# In[ ]:


rfc=RandomForestClassifier(criterion='gini',random_state=4,max_depth=7,min_samples_split=50)
rfc.fit(x_train_over,y_train_over)


# In[ ]:


y_pred_rfc=rfc.predict(x_test)


# In[ ]:


print('Random Forest Metrics')
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rfc))
print('Precision:', metrics.precision_score(y_test, y_pred_rfc))
print('Recall:', metrics.recall_score(y_test, y_pred_rfc))
print('f1_score:', metrics.f1_score(y_test, y_pred_rfc))


# In[ ]:


metrics.plot_confusion_matrix(rfc,x_test,y_test)


# In[ ]:


fpr_dt,tpr_dt,_=roc_curve(y_test,y_pred_rfc)
roc_auc_dt = auc(fpr_dt,tpr_dt)


# In[ ]:


plt.figure(1)
lw=2
plt.plot(fpr_dt,tpr_dt,color='orange',lw=lw,label='Decision Tree(AUC = %0.2f)'%roc_auc_dt)
plt.plot([0,1],[0,1],color='blue',lw=lw,linestyle='--')

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("area Under the Curve")
# plt.legend(loc="upper left")
plt.legend(loc="lower right")


# In[ ]:




