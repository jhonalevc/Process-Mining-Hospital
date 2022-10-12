import pandas as pd
import numpy as np
import pm4py
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

eventlog = pm4py.read_xes(r'C:\Users\AlejandroVelez\OneDrive - Accelirate, Inc\Documents\Accelirate\Python-Process Mining\12705113\Hospital Billing - Event Log.xes.gz')
eventlog_df = pm4py.convert_to_dataframe(eventlog)
eventlog_df['time:timestamp'] = pd.to_datetime(eventlog_df['time:timestamp'],utc=True)
years = eventlog_df['time:timestamp'].dt.year.unique()
#Total

time_df_total = pd.read_csv(r'Exploration\time_eventlog.csv')
time_df_total['time:timestamp'] = pd.to_datetime(time_df_total['time:timestamp'])
#### Variants 

variants_total = pm4py.get_variants_as_tuples(eventlog)
var = []
dat = []
for variant, data in variants_total.items():
    var.append(variant)
    dat.append(len(data))
variants_total_df = pd.DataFrame({
    'variant':var,
    'len_Data':dat})
variants_total_df = variants_total_df.sort_values('len_Data',ascending=False).reset_index().drop('index',axis=1)
v_name = ['varinat_' + str(i) for i in np.arange(len(variants_total_df))]
variants_total_df['variant_name'] = v_name
variants_total_df = variants_total_df[['variant_name','len_Data','variant']]

#### Events
len(eventlog_df)
#### Activities
activities_ = eventlog_df['concept:name'].unique()
### Cases
cases_ = eventlog_df['case:concept:name'].unique()
#### States 
states_= eventlog_df['state'].unique()

# ----------------------Consilidating the Data 

events = len(eventlog_df)
total_variants = len(variants_total_df)
activities = len(activities_)
cases = len(cases_)
states = len(states_)
header_total_df_total = pd.DataFrame({
    'Cases':[cases],
    'Events': [events],
    'Activities': [activities],
    'Variants': [total_variants],
    'States':[states],
    'year': ['Total']
})



ssd = []
xxd = []
for year in years:
    traces_contained = pm4py.filter_time_range(
        eventlog,
        dt1 = datetime.datetime(year-1,12,31),
        dt2= datetime.datetime(year,12,31),
        mode='traces_contained'
    )
    df_traces_contained = pm4py.convert_to_dataframe(traces_contained)
    variants__ = pm4py.get_variants_as_tuples(traces_contained)
    var_ = []
    data_ = []
    for variant,data in variants__.items():
        var_.append(variant)
        data_.append(data)
    df_vars = pd.DataFrame({
        'variant':var_,
        'len_data':data_
    })
    xc = ['variant_' + str(h) for h in np.arange(len(df_vars))]
    #df_vars = df_vars.sort_values('len_data', ascending=False).reset_index().drop('index',axis=1)
    df_vars['variant_name'] = xc
    xxd.append(df_vars)
    events = len(df_traces_contained)
    total_variants = len(df_vars)
    activities = len(df_traces_contained['concept:name'].unique())
    cases = len(df_traces_contained['case:concept:name'].unique())
    states = len(df_traces_contained['state'].unique())


    df_final = pd.DataFrame({
        'Cases': [cases],
        'Events': [events],
        'Activities': [activities],
        'Variants':[total_variants],
        'States': [states],
        'year': [year]
    })

    ssd.append(df_final)



header_total_df = pd.concat(ssd)
header_total_df = pd.concat([header_total_df, header_total_df_total])
header_total_df.to_csv('header_info.csv') # ---------------------------------------------


import plotly.express as plx
month = eventlog_df[['time:timestamp','case:concept:name']]
month['month_year'] = month['time:timestamp'].dt.to_period('M')
month_ = month.groupby('month_year')['case:concept:name'].count().to_frame().reset_index()
month_['month_year'] =  month_['month_year'].astype(str)
month_['month_year'] = pd.to_datetime(month_['month_year'])
month_.to_csv('count_month.csv')

variants_total_df.to_csv('variants_total_df.csv')


# Events_per Case
events_per_case_df = eventlog_df.groupby(['case:concept:name'])['time:timestamp'].count().to_frame().reset_index()
events_per_case_df['t'] = 'a'
events_per_case_df = events_per_case_df.groupby('time:timestamp')['t'].count().to_frame().reset_index()
events_per_case_df.columns = ['Events per case','Count']
events_per_case_df = events_per_case_df.sort_values('Count',ascending=False)
events_per_case_df.to_csv('events_per_case_df.csv')


activities = eventlog_df['concept:name'].unique().tolist()
numbers = []
for activity in activities:
    _ = eventlog_df[eventlog_df['concept:name'] == activity]
    numbers.append(len(_['case:concept:name'].unique()))

g_df = pd.DataFrame({
    'activities':activities,
    'count':numbers
})

g_df['%'] = g_df['count'] / len(eventlog_df['case:concept:name'].unique()) * 100
g_df.to_csv('activities per case.csv')

eventlog_df


df_cancelled = eventlog_df.groupby(['case:concept:name','isCancelled'])['time:timestamp'].count().to_frame().reset_index()
df_cancelled = df_cancelled.drop_duplicates(subset=['case:concept:name'],keep ='last')
df_cancelled =  df_cancelled.groupby('isCancelled')['time:timestamp'].count().to_frame().reset_index()
df_cancelled.to_csv('df_canceled.csv')


df_closed = eventlog_df.groupby(['case:concept:name','isClosed'])['time:timestamp'].count().to_frame().reset_index()
df_closed = df_closed.drop_duplicates(subset=['case:concept:name'],keep ='last')
df_closed =  df_closed.groupby('isClosed')['time:timestamp'].count().to_frame().reset_index()
df_closed.to_csv('df_closed.csv')

#Heuristic Net

map_heu = pm4py.discover_heuristics_net(eventlog)
#pm4py.view_heuristics_net(map_heu)

k_a = []
v_b = []
for a,b in pm4py.get_start_activities(eventlog).items():
    k_a.append(a)
    v_b.append(b)

starts =  pd.DataFrame({'Activity':k_a,' # of Cases':v_b})
starts.to_csv('start_Activities_all_variants.csv')

# --------------------------------------------


ends = list(pm4py.get_end_activities(eventlog_df).keys())
value_ends = list(pm4py.get_end_activities(eventlog).values())
ends_df = pd.DataFrame({
    'ends':ends,
    'value_ends':value_ends
})
ends_df['Percentage'] = (ends_df['value_ends']/100000) * 100
ends_df = ends_df.T
ends_df.columns = ends_df.loc['ends']
ends_df = ends_df.drop('ends',axis=0)
ends_df.T.reset_index().to_csv('end_activities_allvariants.csv')

bpm_inductive_all_variants = pm4py.discover_bpmn_inductive(eventlog)
#pm4py.view_bpmn(bpm_inductive_all_variants)

variants_total_df.to_csv('allvariants-details.csv')

net_ind, initial_ind, final_ind = pm4py.discover_petri_net_inductive(eventlog)
#pm4py.view_petri_net(net_ind,initial_marking= initial_ind,final_marking= final_ind)


net, initial, final = pm4py.discover_petri_net_alpha_plus(eventlog)
#pm4py.view_petri_net(net,initial_marking= initial,final_marking= final)


net, initial, final = pm4py.discover_petri_net_alpha(eventlog)
#pm4py.view_petri_net(net,initial_marking= initial,final_marking= final)
#

#Filter with the most comon Variants

variants_total_df["%"] = variants_total_df['len_Data']/variants_total_df['len_Data'].sum() * 100
variants_total_df_subset = variants_total_df[variants_total_df["%"]>30]
event_log_filtered = pm4py.filter_variants(eventlog,variants_total_df_subset['variant'].to_list(),retain=True)

#map_heu_best = pm4py.discover_heuristics_net(event_log_filtered)
#pm4py.view_heuristics_net(map_heu_best)

#bpm_inductive_best = pm4py.discover_bpmn_inductive(event_log_filtered)
#pm4py.view_bpmn(bpm_inductive_best)

#net, initial, final = pm4py.discover_petri_net_alpha_plus(event_log_filtered)
#pm4py.view_petri_net(net,initial_marking= initial,final_marking= final)

#net, initial, final = pm4py.discover_petri_net_alpha(event_log_filtered)
#pm4py.view_petri_net(net,initial_marking= initial,final_marking= final)

#net_ind, initial_ind, final_ind = pm4py.discover_petri_net_inductive(event_log_filtered)
#pm4py.view_petri_net(net_ind,initial_marking= initial_ind,final_marking= final_ind)

best_df = pm4py.convert_to_dataframe(event_log_filtered)
start_best = pm4py.get_start_activities(event_log_filtered)
_s = []
_x = []
for s,x in start_best.items():
    _s.append(s)
    _x.append(x)
start_best_df = pd.DataFrame({'Activity':_s,'# of cases':_x})
#start_best_df.to_csv('start_activities_best.csv')
end_best = pm4py.get_end_activities(event_log_filtered)
_a = []
_q = []
for a,q in end_best.items():
    _a.append(a)
    _q.append(q)
end_best_df = pd.DataFrame({'Activity':_a,'ends':_q})
end_best_df['Percentage'] = end_best_df['ends'] /  len(best_df['case:concept:name'].unique()) * 100
#end_best_df.to_csv('end_activities_best.csv')

#variants_total_df_subset.to_csv("best_variants_details.csv")


eventlog = pm4py.read_xes(r'C:\Users\AlejandroVelez\OneDrive - Accelirate, Inc\Documents\Accelirate\Python-Process Mining\12705113\Hospital Billing - Event Log.xes.gz')


#Best Path Average Duration
#All variants  /  Average Duration 



zx = eventlog_df[eventlog_df['case:concept:name'] =='A'][['time:timestamp','concept:name','case:concept:name']]
zx = zx.sort_values('time:timestamp')
zx['inteval'] = zx['time:timestamp'].diff().shift(-1).fillna(datetime.timedelta(0)).apply(lambda x :x.total_seconds() /3600)

data_frames = []
for case in eventlog_df['case:concept:name'].drop_duplicates().to_list():
    df_ = eventlog_df[eventlog_df['case:concept:name'] == case][['time:timestamp','isCancelled','isClosed','concept:name','case:concept:name']]
    df_ = df_.sort_values('time:timestamp')
    df_['interval'] = df_['time:timestamp'].diff().shift(-1).fillna(datetime.timedelta(0)).apply(lambda x: x.total_seconds() / 3600)
    data_frames.append(df_)

time_df = pd.concat(data_frames)
time_df.to_csv('time_eventlog.csv')


#variants_df


variants_tuple = pm4py.get_variants_as_tuples(eventlog)
kk = []
vv = []
cases_len = []
for k,v in variants_tuple.items():
    kk.append(k)
    vv.append(v)

for list_in in vv:
    cases_len.append(len(list_in))
r = []
for list_ in vv:
    cases_outer = []
    for value in list_:
        internal_cases = []
        for internal_dict in value:
            cases_list = []
            for key_,value_ in dict(internal_dict).items():
                if key_ == 'case:concept:name':
                    cases_list.append(value_)
            for case__ in cases_list:
                internal_cases.append(case__)
        for internal_case in internal_cases:
            cases_outer.append(internal_case)
    cases_outer = pd.Series(cases_outer)
    cases_outer = cases_outer.unique().tolist()
    r.append(cases_outer)
    
variants_Dataframe = pd.DataFrame({'Variant':kk,'Data':vv,'Cases':r,'case_len':cases_len})
variants_Dataframe['%'] = variants_Dataframe['case_len']/variants_Dataframe['case_len'].sum() * 100
variants_Dataframe = variants_Dataframe.sort_values("%",ascending =False)
variant_name = ['Variant_'+str(i) for i in np.arange(len(variants_Dataframe))]
variants_Dataframe['Variant_Name'] = variant_name


dataframes_d = []
for Variant, Cases in zip(variants_Dataframe['Variant_Name'].values,variants_Dataframe['Cases']):
    for case in Cases:
        xc = pd.DataFrame({'Case':[case],'Variant':[Variant]})
        dataframes_d.append(xc)

dataframes_d = pd.concat(dataframes_d)

merged = pd.merge(dataframes_d,variants_Dataframe, left_on='Variant', right_on='Variant_Name')
merged = merged.drop(['Variant_x'],axis=1)
merged = merged.drop(['Cases','Data'],axis =1)
#merged.to_csv('case_name_variant.csv',index =False)
df_traces_ = merged.copy()


time_df_total_merged_aa = time_df_total.merge(merged, left_on ='case:concept:name', right_on = 'Case' )
all_variants_ = time_df_total_merged_aa.groupby('case:concept:name')['interval'].agg(['sum']).reset_index()
_min = [all_variants_['sum'].min()]
_mean = [all_variants_['sum'].mean()]
_max = [all_variants_['sum'].max()]
df_time_all_t = pd.DataFrame({
    'min':_min,
    'mean':_mean,
    'max':_max
})
df_time_all_t = df_time_all_t.T
df_time_all_t['days'] = df_time_all_t[0]/24
df_time_all_t['years'] = df_time_all_t['days']/365
df_time_all_t = df_time_all_t.rename(columns={0:'hours'})
#df_time_all_t.to_csv('summary_time_all.csv',index = False)

dataframe_all_variant_act = time_df_total_merged_aa.groupby('concept:name')['interval'].agg(['min','mean','max','std',lambda x:x.quantile(0.25),lambda x:x.quantile(0.75)]).reset_index()
dataframe_all_variant_act.columns = ['concept:name','min','mean','max','std','q25','q75']
dataframe_all_variant_act = dataframe_all_variant_act.T
dataframe_all_variant_act.columns= dataframe_all_variant_act.loc['concept:name',:]
dataframe_all_variant_act = dataframe_all_variant_act.drop('concept:name')
dataframe_all_variant_act= dataframe_all_variant_act.round(1)
dataframe_all_variant_act_copy = dataframe_all_variant_act.copy().divide(24)
dataframe_all_variant_act['time'] ='Hours'
dataframe_all_variant_act_copy['time'] ='days'
dataframe_all_variant_act_full = pd.concat([dataframe_all_variant_act_copy,dataframe_all_variant_act])
#dataframe_all_variant_act_full.to_csv('dataframe_all_variant_act_full.csv')
df_time_all_t


# relevants_variant
time_df_total_merged_aa_relevant = time_df_total_merged_aa.loc[(time_df_total_merged_aa['%']>2),:]
time_df_total_merged_aa_relevant['Variant_Name'].unique()
time_df_total_merged_aa_relevant_grouped = time_df_total_merged_aa_relevant.groupby('case:concept:name')['interval'].agg(['sum']).reset_index()
_min = [time_df_total_merged_aa_relevant_grouped['sum'].min()]
_mean = [time_df_total_merged_aa_relevant_grouped['sum'].mean()]
_max = [time_df_total_merged_aa_relevant_grouped['sum'].max()]
df_time_all_t_grp = pd.DataFrame({
    'min':_min,
    'mean':_mean,
    'max':_max
})
df_time_all_t_grp = df_time_all_t_grp.T
df_time_all_t_grp['days'] = df_time_all_t_grp[0]/24
df_time_all_t_grp['years'] = df_time_all_t_grp['days']/365
df_time_all_t_grp = df_time_all_t_grp.rename(columns={0:'hours'})
#df_time_all_t_grp.to_csv('summary_time_relevant.csv',index = False)
#df_time_all_t_grp



dataframe_all_relevant_act = time_df_total_merged_aa_relevant.groupby('concept:name')['interval'].agg(['min','mean','max','std',lambda x:x.quantile(0.25),lambda x:x.quantile(0.75)]).reset_index()
dataframe_all_relevant_act.columns = ['concept:name','min','mean','max','std','q25','q75']
dataframe_all_relevant_act = dataframe_all_relevant_act.T
dataframe_all_relevant_act.columns = dataframe_all_relevant_act.loc['concept:name',:]
dataframe_all_relevant_act = dataframe_all_relevant_act.drop('concept:name')
dataframe_all_relevant_act = dataframe_all_relevant_act.round(1)
dataframe_all_relevant_act_copy = dataframe_all_relevant_act.divide(24)
dataframe_all_relevant_act['time'] ='hours'
dataframe_all_relevant_act_copy['time'] ='days'
dataframe_all_relevant_act_full = pd.concat([dataframe_all_relevant_act,dataframe_all_relevant_act_copy])
#dataframe_all_relevant_act_full.to_csv('dataframe_all_relevant_act_full.csv')


# Best?path
time_df_best = time_df_total_merged_aa.loc[(time_df_total_merged_aa['%']>30),:]
time_df_best['Variant_Name'].unique()
time_df_best_gropued = time_df_best.groupby('case:concept:name')['interval'].agg(['sum']).reset_index()
_min = [time_df_best_gropued['sum'].min()]
_mean = [time_df_best_gropued['sum'].mean()]
_max = [time_df_best_gropued['sum'].max()]
time_df_best_gropued_total = pd.DataFrame({
    'min':_min,
    'mean':_mean,
    'max':_max
})


time_df_best_gropued_total = time_df_best_gropued_total.T
time_df_best_gropued_total['days'] = time_df_best_gropued_total[0]/24
time_df_best_gropued_total['years'] = time_df_best_gropued_total['days']/365
time_df_best_gropued_total = time_df_best_gropued_total.rename(columns={0:'hours'})
#time_df_best_gropued_total.to_csv('summary_time_best.csv',index = False)


dataframe_best_act = time_df_best.groupby('concept:name')['interval'].agg(['min','mean','max','std',lambda x:x.quantile(0.25),lambda x:x.quantile(0.75)]).reset_index()
dataframe_best_act.columns = ['concept:name','min','mean','max','std','q25','q75']
dataframe_best_act = dataframe_best_act.T
dataframe_best_act.columns = dataframe_best_act.loc['concept:name',:]
dataframe_best_act = dataframe_best_act.drop('concept:name')
dataframe_best_act_copy = dataframe_best_act.divide(24)
dataframe_best_act['time']= 'Hour'
dataframe_best_act_copy['time'] ='Day'
dataframe_best_act_full = pd.concat([dataframe_best_act,dataframe_best_act_copy])
dataframe_best_act_full.to_csv('dataframe_best_act_full.csv')


s1 = 'A'
s2 =  'WD'
s3 =  'BXQ'

s1_df = time_df_total_merged_aa[time_df_total_merged_aa['Case'] == s1]
s2_df = time_df_total_merged_aa[time_df_total_merged_aa['Case'] == s2]
s3_df = time_df_total_merged_aa[time_df_total_merged_aa['Case'] == s3]

df_final = pd.concat([s1_df,s2_df,s3_df])
df_final = df_final[['time:timestamp','concept:name','case:concept:name']]

#time_df_total_merged_aa
plx.line(df_final,x = 'concept:name',y = 'time:timestamp',color= 'case:concept:name',markers=True)