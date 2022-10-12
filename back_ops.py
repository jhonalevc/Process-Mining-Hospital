import pandas as pd
import numpy as np
import pm4py
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


eventlog = pm4py.read_xes(r'C:\Users\AlejandroVelez\OneDrive - Accelirate, Inc\Documents\Accelirate\Python-Process Mining\12705113\Hospital Billing - Event Log.xes.gz')
eventlog_df = pm4py.convert_to_dataframe(eventlog)
eventlog_df['time:timestamp'] = pd.to_datetime(eventlog_df['time:timestamp'],utc=True)
eventlog_df.head(5)


plt.style.use('dark_background')

#eventlog_df.info()


min_time = eventlog_df['time:timestamp'].min()
max_time = eventlog_df['time:timestamp'].max()
print(' - The time elapsed in the eventlog is : {}'.format(max_time-min_time))
#Min and max time_each_day
eventlog_df_copy = eventlog_df.copy()
eventlog_df_copy['date'] = eventlog_df_copy['time:timestamp'].dt.date
eventlog_df_copy['hour'] = eventlog_df_copy['time:timestamp'].dt.hour
eventlog_df_copy['weekday'] = eventlog_df_copy['time:timestamp'].dt.day_name()


eventlog_df_copy_grp = eventlog_df_copy.groupby('date')['hour'].agg(['min','max','mean','count']).reset_index()
print(' - The avg minimum hour of activities in the billing dept is {: 2f}'.format(eventlog_df_copy_grp['min'].mean()))
print(' - The minimum hour of activities in the billing dept is {:}'.format(eventlog_df_copy_grp['min'].min()))
print(' - The avg maximum hour of activities in the billing dept is {: 2f}'.format(eventlog_df_copy_grp['max'].mean()))
print(' - The maximum hour of activities in the billing dept is {:}'.format(eventlog_df_copy_grp['max'].max()))


plt.rcParams['figure.figsize'] = (21, 5)
plot_time = sns.histplot(data= eventlog_df_copy,x ='hour', color='blue')
plot_time.set_title('Number of ticket actions per time of the day')
plt.show()
plt.clf()
weekplot = sns.histplot(data=eventlog_df_copy, x = 'weekday',color='green')
weekplot.set_title('Number of ticket actions per day of the week ')
plt.show()
plt.clf()

eventlog_df_copy_grp_2 = eventlog_df_copy.groupby('case:concept:name')['time:timestamp'].agg(['min','max']).reset_index()
eventlog_df_copy_grp_2['diff'] = eventlog_df_copy_grp_2['max'] - eventlog_df_copy_grp_2['min']
eventlog_df_copy_grp_2['diff_day_hours'] = eventlog_df_copy_grp_2['diff'].dt.days*24
eventlog_df_copy_grp_2['diff_second_hours'] = eventlog_df_copy_grp_2['diff'].dt.seconds/3600
eventlog_df_copy_grp_2['total_diff_hours'] = eventlog_df_copy_grp_2['diff_day_hours'] + eventlog_df_copy_grp_2['diff_second_hours']

ticket_duration_hours_plot = sns.boxplot(
    data=eventlog_df_copy_grp_2,
    x = 'total_diff_hours',
    color='orange'
)
plt.show()
plt.clf()


print(
    ' - The average duration of a ticket is : {:2f}'.format(eventlog_df_copy_grp_2['total_diff_hours'].mean()/24) + ' days'
)
print(' - The longest a ticket has taken to be resolved is {}'.format(eventlog_df_copy_grp_2['total_diff_hours'].max()/24) + ' days')
print(' - The shortest a ticket has taken to be resolved is {}'.format(eventlog_df_copy_grp_2['total_diff_hours'].min()) +' days')


print('Total number of cases in this eventlog (Patients): {}'.format(len(eventlog)))
print('Total number of activities peformed in this eventlog: {}'.format(len(eventlog_df)))
print('\n')
print('Start Activities:')
print(pm4py.get_start_activities(eventlog_df))
print('End Activities Activities:')
print(pm4py.get_end_activities(eventlog_df))
print('\n')
ends = list(pm4py.get_end_activities(eventlog_df).keys())
value_ends = list(pm4py.get_end_activities(eventlog).values())
value_ends_plot = sns.barplot(x =ends, y=value_ends)
plt.show()

ends_df = pd.DataFrame({
    'ends':ends,
    'value_ends':value_ends
})
ends_df['Percentage'] = (ends_df['value_ends']/100000) * 100
ends_df = ends_df.T
ends_df.columns = ends_df.loc['ends']
ends_df = ends_df.drop('ends',axis=0)
ends_df

#Variants
variants = pm4py.get_variants_as_tuples(eventlog)
traces_ = []
len_data = []
for trace, data in variants.items():
    traces_.append(trace)
    len_data.append(len(data))

df_traces = pd.DataFrame({
    'trace':traces_,
    'len':len_data
})
print('There are {}'.format(len(traces_)) + ' Traces')
idx = []
for a,b in zip(df_traces['trace'].values,np.arange(len(traces_))):
    if a[-1] == 'NEW':
        idx.append(b)
# Filter on the indices where the end activity is 'NEW'12705113
df_traces_filrered = df_traces.loc[idx,:]
print('There are {}'.format(len(df_traces_filrered)) + ' Where the trace ends with "NEW" ')


eventolg_filtered_1 = pm4py.filter_variants(eventlog,variants= df_traces_filrered['trace'].to_list(),retain = False)
variants_1 = [] 
len_data_1 = []
for key,value in pm4py.get_variants_as_tuples(eventolg_filtered_1).items():
    variants_1.append(key)
    len_data_1.append(len(value))
df_variants_2 = pd.DataFrame({'variant':variants_1,'len':len_data_1})

# ----------------------------------------------------------

ends_2 = pm4py.get_end_activities(eventolg_filtered_1)
_x = list(ends_2.keys())
_y = list(ends_2.values())

plot_2 = sns.barplot(x=_x,y=_y)
plot_2.set_title('Ends removing New')
plt.show(plot_2)
plt.clf()


df_ends_2 = pd.DataFrame(
    {
        'ends':_x,
        'len': _y
    }
)

print('The new amount of case_ids is {}'.format(len(eventolg_filtered_1)))


df_ends_2['%'] = df_ends_2['len']/len(eventolg_filtered_1)
df_ends_2 = df_ends_2.T
df_ends_2.columns = df_ends_2.loc['ends',:]
df_ends_2 = df_ends_2.drop('ends',axis=0)
df_ends_2

# -------------------------------------------------------------


target_ends =['BILLED','DELETE','FIN']
IDX_2 = []
for a,b in zip(df_variants_2['variant'].to_list(),np.arange(len(df_variants_2))):
    if a[-1] not in target_ends:
        IDX_2.append(b)

traces_to_del = df_variants_2.loc[IDX_2,:]['variant'].to_list()

event_log_final = pm4py.filter_variants(
    eventolg_filtered_1,
    traces_to_del,
    retain=False
)

traces_final = pm4py.get_variants_as_tuples(event_log_final)

keys_final = []
lens_final = []
for key,value in traces_final.items():
    keys_final.append(key)
    lens_final.append(len(value))

traces_df_final = pd.DataFrame(
    {
        'trace':keys_final,
        'len':lens_final
    }
)

traces_df_final['perc. total'] = traces_df_final['len']/len(pm4py.convert_to_dataframe(event_log_final)) * 100
traces_df_final = traces_df_final.sort_values('perc. total',ascending=False).reset_index().drop('index',axis=1)

traces_df_final = traces_df_final.loc[(traces_df_final['len']>100),:]
print('There are {} traces that have more than 100 patients attended'.format(len(traces_df_final)) + ' ,We will work with these')
end_event_log = pm4py.filter_variants(
    event_log_final,
    traces_df_final['trace'].to_list())


bpm_inductive = pm4py.discover_bpmn_inductive(end_event_log)
pm4py.view_bpmn(bpm_inductive)


net, initial, final = pm4py.discover_petri_net_alpha(end_event_log)
#pm4py.view_petri_net(net,initial_marking= initial,final_marking= final)

net_ind, initial_ind, final_ind = pm4py.discover_petri_net_inductive(end_event_log)
pm4py.view_petri_net(net_ind,initial_marking= initial_ind,final_marking= final_ind)

map_heu = pm4py.discover_heuristics_net(end_event_log)
pm4py.view_heuristics_net(map_heu)