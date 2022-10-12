from base64 import encode
import pandas as pd
import numpy as np
import plotly.express as plx
import streamlit as st
#import pm4py
import streamlit_nested_layout
from PIL import Image
from io import BytesIO


#xes_data = pm4py.read_xes('12705113\Hospital Billing - Event Log.xes.gz')
#eventlog_df = pm4py.convert_to_dataframe(xes_data)

st.set_page_config(layout='wide')

selectbox = st.sidebar.selectbox(
    "Available Options for you to explore",
    (   
        'Intro',
        'overview',
        'timing',
        'process',
        'data'
    ))

with st.sidebar:
    
    st.markdown("<br>",unsafe_allow_html=True)
    st.info("If assistance is required, contact  Alejandro Velez")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.info("Updated : October 2022")

if selectbox == 'Intro':
    st.markdown("""<h1 style='text-align: center'>Introduction  - Process Mining Project Net</h1>""",unsafe_allow_html=True)

    y1,y2 = st.columns(2)
    with y2:
        st.image(r'Images\1-j3dTgXjyaYPo9XkM6UMZ3g-removebg-preview.png',use_column_width=True )
    with y1:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(
            """
            <p style="text-align:justify">
                Definition: Process mining is a family of techniques relating the fields of data science and process management 
                to support the analysis of operational processes based on event logs. The goal of process mining is to turn event
                data into insights and actions. Process mining is an integral part of data science, fueled by the availability of event data and 
                the desire to improve processes.[1] Process mining techniques use event data to show what people, machines, and organizations are really doing. 
                Process mining provides novel insights that can be used to identify the executional path taken by operational processes and address their
                performance and compliance problems.
            </p>
            """,unsafe_allow_html=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    with st.expander("Expand to see the basic layout of the project architecture"):
        v1,v2,v3,v4,v5,v6,v7 = st.columns([1,1,1,17,1,1,1]) 
        with v4: 
            st.image(r'Images\Capture-removebg-preview.png',use_column_width=True )
    i1,i2 = st.columns([2,1])
    with i1:
        st.markdown("""
            <p style="text-align:justify">
                Designed to be used in both academia and industry, PM4Py is
                the leading open source process mining platform written in Python, implementing:
                Conformance Checking,Process Discovery, BPMN Support, Open XES/Importing & Exporting.
            </p> """,unsafe_allow_html=True) 
        st.markdown("""    
            <p style="text-align:justify">
                The term "Process mining" was first coined in a research proposal written by the Dutch computer scientist Wil van der Aalst ("Godfather of Process mining"). 
                Thus began a new field of research that emerged under the umbrella of techniques related to data science and process science at the Eindhoven University in 1999. 
                In the early days, process mining techniques were often convoluted with the techniques used for workflow management. In the year 2000, the very first practically a
                pplicable algorithm for process discovery, "Alpha miner" was developed. The very next year, in 2001, a much similar algorithm based on heuristics called "Heuristic miner"
                was introduced in the research papers. Further along the link more powerful algorithms such as inductive miner were developed for process discovery. As the field 
                of process mining began to evolve, conformance checking became an integral part of it.
        """,unsafe_allow_html=True)

        with i2:
            st.markdown("""<h3 style="text-align:center"> Resources </h3>""",unsafe_allow_html=True)

            st.markdown("""<a href="https://www.python.org/" >Python - Official Site</a>""",unsafe_allow_html=True)
            st.markdown("""<a href="https://pm4py.fit.fraunhofer.de/" >PM4PY - Official Site</a>""",unsafe_allow_html=True)
            st.markdown("""<a href="https://streamlit.io/" >Streamlit - Official Site</a>""",unsafe_allow_html=True)
            st.markdown("""<a href="https://azure.microsoft.com/en-us/" >Azure - Official Site</a>""",unsafe_allow_html=True)
            st.markdown("""<a href="https://github.com/jhonalevc/Process-Mining-Hospital" >Github repo with the code </a>""",unsafe_allow_html=True)



       
 

if selectbox == 'overview':
    # ---------------------- Title --------------------------
    
    st.markdown(
        """
        <h1 style='text-align: center'>
            Overview
        </h1>
        """, unsafe_allow_html= True    )

    # ----------------------- Header Info --------------------

    data_header = pd.read_csv('Dataframes\header_info.csv')
    with st.container():
        with st.expander('Filter Data!'):
            option_header_period = st.selectbox(
                'Select The Desired Period',
                data_header['year'].unique(),
                label_visibility="hidden",
                index=5
            )
        data_header = data_header[data_header['year'] == option_header_period]
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col1:
            st.subheader('Cases')
            st.markdown(data_header['Cases'].to_list()[0])
        with col2:
            st.subheader('Events')
            st.markdown(data_header['Events'].to_list()[0])
        with col3:
            st.subheader('Activities')
            st.markdown(data_header['Activities'].to_list()[0])
        with col4:
            st.subheader('Variants')
            st.markdown(data_header['Variants'].to_list()[0])
        with col5:
            st.subheader('States')
            st.markdown(data_header['States'].to_list()[0])
        with col6:
            st.subheader('Period')
            st.markdown(data_header['year'].to_list()[0])

    st.markdown("<hr>",unsafe_allow_html=True)


    # ---------------------  Main plot ----------------------------- 


    month_ = pd.read_csv(r'Dataframes\count_month.csv')
    month_['month_year'] = pd.to_datetime(month_['month_year'])
    month_ = month_.drop(month_.columns.to_list()[0], axis=1)

    with st.container():
        col1_, col2_ = st.columns(2)
        with col1_:
            global monthly_yearly
            monthly_yearly = st.radio('Year  - Month', ['Year','Month'],index = 0)
        with col2_:
            if monthly_yearly == 'Month':
                    months = month_['month_year'].dt.month.unique()
                    month_ = month_.copy()
                    month_['month'] = month_['month_year'].dt.month
                    month_ = month_.groupby('month')['case:concept:name'].sum().to_frame().reset_index()
                    month_.columns = ['month_year','case:concept:name']
                    #month_selected = st.selectbox(label='Choose Month',options = month_['month_year'])
                    st.error("Montlhy Data")
                    month_d = month_.copy()
            else:
                year_select = st.selectbox('Select Year', options = month_['month_year'].dt.year.unique().tolist() + ['Total'] ,index = 5)
                if year_select == 'Total':
                    month_d = month_.copy()
                else:
                    month_d = month_[month_['month_year'].dt.year == year_select]
        try:
            if year_select == 2012:
                st.error("Only one Month to display")
            elif year_select == 2016:
                st.error("Only one Month to display")
        except:
            pass
        plot_month = plx.area(
            month_d,
            x ='month_year',
            y='case:concept:name'
            )
        st.plotly_chart(plot_month,use_container_width=True)
        st.markdown("<hr>",unsafe_allow_html=True)


        # ---------------------  Three plots together ----------------------------- 

    with st.container():

        variants_total_df = pd.read_csv(r'Dataframes\variants_total_df.csv')
        variants_total_df['%'] = variants_total_df['len_Data'] /variants_total_df['len_Data'].sum() * 100
        events_per_case_df = pd.read_csv(r'Dataframes\events_per_case_df.csv')
        events_per_case_df['Events per case'] = events_per_case_df['Events per case'].astype(str)
        events_per_case_df['%'] = events_per_case_df['Count'] / events_per_case_df['Count'].sum() * 100
        activities_per_case = pd.read_csv(r'Dataframes\activities per case.csv')

        col_1,col_2,col_3 = st.columns(3)
        with col_1:
            st.markdown("""
            <h3 style='text-align: center'>
                Variants
            </h3>
            """,unsafe_allow_html=True)
            with st.container():
                variants_total_plot = plx.bar(variants_total_df.head(25),x = 'variant_name', y='%')
                st.plotly_chart(variants_total_plot,use_container_width=True)
        with col_2:
            st.markdown("""
            <h3 style='text-align: center'>
                Events per Case
            </h3>
            """,unsafe_allow_html=True)
            with st.container():
                events_per_Case_plot = plx.bar(events_per_case_df,x = '%', y='Events per case')
                st.plotly_chart(events_per_Case_plot,use_container_width=True)
        with col_3:
            st.markdown("""
            <h3 style='text-align: center'>
                Activities
            </h3>
            """,unsafe_allow_html=True)
            with st.container():
                activities_plot = plx.bar(activities_per_case, x = 'activities',y='%' )
                st.plotly_chart(activities_plot,use_container_width=True)
    st.markdown("""<hr>""", unsafe_allow_html=True)



    df_canceled = pd.read_csv('Dataframes\df_canceled.csv')
    df_canceled = df_canceled.drop(df_canceled.columns.to_list()[0],axis =1)
    df_closed = pd.read_csv('Dataframes\df_closed.csv')
    df_closed = df_closed.drop(df_closed.columns.to_list()[0],axis=1)
    with st.container():
        column1, column2 = st.columns(2)

        with column1:
            with st.container():
                st.markdown("""<h3 style='text-align: center'>Is Cancelled</h3>""",unsafe_allow_html=True)
                plot_pie_canceled = plx.pie(data_frame=df_canceled,names ='isCancelled',values = 'time:timestamp')
                st.plotly_chart(plot_pie_canceled,use_container_width=True)
                
                with st.expander("Details"):
                    clla, collb = st.columns(2)
                    with clla:
                        st.markdown(
                            """
                            <p align="justify">
                                According to the Dataset documentation : isCancelled -  A flag that indicates whether the billing package  was eventually cancelled.
                            </p>
                            """, unsafe_allow_html=True)
                    with collb:
                        df_canceled

        
        with column2:
            with st.container():
                st.markdown("""<h3 style='text-align: center'>Is Closed</h3>""",unsafe_allow_html=True)
                with st.expander("Details"):
                    colla_ , collb_ = st.columns(2)
                    with colla_:
                        df_closed              
                    with collb_:
                        st.markdown(
                        """
                        <p align="justify">
                            According to the Dataset documentation : isClosed -   A flag that indicates whether the billing package  was eventually closed.
                        </p>
                        """, unsafe_allow_html=True)
                plot_pie_closed = plx.pie(data_frame=df_closed,names ='isClosed',values = 'time:timestamp')
                st.plotly_chart(plot_pie_closed,use_container_width=True)                           

if selectbox == 'timing':
    global time_df_total_merged
    df_timing = pd.read_csv(r'Dataframes\time_eventlog.csv')
    df_timing['time:timestamp'] = pd.to_datetime(df_timing['time:timestamp'])
    df_traces_ = pd.read_csv(r'Dataframes/case_name_variant.csv')
    time_df_total = df_timing.groupby('case:concept:name')['interval'].sum().to_frame().reset_index()
    time_df_total_merged = time_df_total.merge(df_traces_, left_on ='case:concept:name', right_on = 'Case' ) 
    time_df_total_merged_aa = df_timing.merge(df_traces_, left_on ='case:concept:name', right_on = 'Case' )
    time_df_total_merged_ =  time_df_total_merged.groupby('Case')['interval'].sum().to_frame().reset_index()
    st.markdown("""<b><h1 style='text-align:center'> Timing </h1></b>""",unsafe_allow_html =True)
    with st.container():
        with st.container(): # -------------------------------------------------Details
            a1,a2,a3,a4,a5,a6 = st.columns(6)
            with a1:
                init_time = """<b><p style='text-align:center'> """ + df_timing['time:timestamp'].min().strftime("%m/%d/%Y, %H:%M:%S")  + """ </p></b>"""
                st.markdown("""<b><h2 style='text-align:center'> Initial Time </h2></b>""",unsafe_allow_html =True)
                st.markdown(init_time,unsafe_allow_html =True)
            with a2:
                st.markdown("<hr>", unsafe_allow_html=True)
            with a3:
                st.markdown("<hr>", unsafe_allow_html=True)
            with a4:
                st.markdown("<hr>", unsafe_allow_html=True)
            with a5:
                st.markdown("<hr>", unsafe_allow_html=True)
            with a6:
                ennd_time = """<b><p2 style='text-align:center'> """ +  df_timing['time:timestamp'].max().strftime("%m/%d/%Y, %H:%M:%S") + """ </p></b>"""
                st.markdown("""<b><h2 style='text-align:center'> Ending Time </h2></b>""",unsafe_allow_html =True)
                st.markdown(ennd_time,unsafe_allow_html =True)
        with st.expander("Time elapsed Details",expanded=True):
            q1,q2,q3,q4,q5,q6,q7,q8 = st.columns(8)

            diff = df_timing['time:timestamp'].max() - df_timing['time:timestamp'].min()
            days = diff.days
            hours = int(diff.seconds / 3600)
            minutes = int(((diff.seconds/3600) - int(hours)) * 60)
            seconds = round(((((diff.seconds/3600) - int(hours)) * 60)  - minutes) * 60)
            with q4:
                st.markdown("""<p style='text-align:center'> """ + str(days)  + """ Days </p>""", unsafe_allow_html=True)
                st.markdown("""<p style='text-align:center'>""" +  str(hours) + """ Hours </p>""", unsafe_allow_html=True)
            with q5:
                st.markdown("""<p style='text-align:center'>""" + str(minutes) + """ Minutes  </p>""", unsafe_allow_html=True)
                st.markdown("""<p style='text-align:center'> """ + str(seconds) + """ Seconds </p>""", unsafe_allow_html=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    with st.container():
        w1, w2 = st.columns([2,1])
        with w1:
            #global variant_selector
            global case_selector
            #variant_selector = st.selectbox("Select Trace",df_traces_['Variant_Name'].unique())
            case_selector = st.selectbox("Select Case",df_traces_['Case'].unique())
            #df_traces_grouped = df_traces_.groupby(['Variant_Name'])['case_len'].max().to_frame().reset_index()
            #df_traces_grouped = df_traces_grouped.sort_values('case_len')
            #plot_occurance_ = plx.bar(df_traces_grouped,y = 'Variant_Name', x = 'case_len')
            #st.plotly_chart(plot_occurance_, use_container_width=True)
            index_target = time_df_total_merged_[time_df_total_merged_['Case'] == case_selector].index.to_list()
            plus_5 = [ index_target[0] +  i for i in np.arange(5)]
            minus_5 = [ index_target[0] - i for i in np.arange(5)]
            x = plus_5 + index_target
            indexs = x + minus_5
            plot_time_1 = st.plotly_chart(plx.bar(time_df_total_merged_.iloc[indexs,:],x ='Case', y = 'interval'),use_container_width=True)
        with w2:
            with st.container():
                e1,e2,e3 = st.columns(3)
                with e1:
                    st.markdown("""<h3 style='text-align:center'> Variant </h3>""", unsafe_allow_html=True)
                    #string_er = """<p style='text-align:center>""" + str(variant_selector) + "</p>"
                    st.markdown("<br>",unsafe_allow_html=True)                  
                    st.markdown(df_traces_[df_traces_['Case'] == case_selector ]['Variant_Name'].to_list()[0],unsafe_allow_html=True)
                with e2:
                    st.markdown("""<h3 style='text-align:center'> Case </h3>""", unsafe_allow_html=True)
                    st.markdown("<br>",unsafe_allow_html=True)
                    #string_ef = """<p style='text-align:center>""" + str(case_selector) + "</p>"
                    st.markdown(case_selector,unsafe_allow_html=True)
                with e3:
                    global _z_duration
                    st.markdown("""<h3 style='text-align:center'> Time Elapsed in Hours </h3>""", unsafe_allow_html=True)
                    #string_ef = """<p style='text-align:center>""" + str(case_selector) + "</p>"
                    _z_duration = df_timing[df_timing['case:concept:name'] == case_selector][['time:timestamp','concept:name','interval']]
                    str__val = """<p style='text-align:center'> """ + str(round(_z_duration['interval'].sum(),2)) + """</p> """
                    st.markdown(str__val,unsafe_allow_html=True)
            st.markdown("<hr>",unsafe_allow_html=True)
            with st.container():
                st.dataframe(_z_duration,use_container_width=True)
        st.markdown("<hr>",unsafe_allow_html=True)
        with st.container():
            r1,r2 = st.columns([1,3])
            with r1:
                radio_variant_act = st.radio(horizontal=True,label='Analyze variant or activity',options = ['Variant', 'Activity']) # --------------------------------------
            with r2:
                if radio_variant_act == 'Variant':
                    global variant_selector
                    variant_selector = st.selectbox("Select Trace",df_traces_['Variant_Name'].unique(),label_visibility='hidden')
                else:
                    global act_select 
                    act_select = st.selectbox("Select Activity",label_visibility='hidden',options= df_timing['concept:name'].unique())
        st.markdown("<br>",unsafe_allow_html=True)
        with st.container():
            if radio_variant_act == 'Activity':
                global df__filtered_concept
                df__filtered_concept = df_timing[df_timing['concept:name'] == act_select]
                f1,f2 = st.columns([2,6])
                with f1:
                    st.dataframe(df__filtered_concept.describe()[['interval']])
                with f2:
                    with st.expander("Box plot"):
                        st.plotly_chart(plx.box(df__filtered_concept,x = 'interval'),use_container_width=True)
                    st.markdown(unsafe_allow_html=True,body = "<br>")
                    with st.expander("Scatter plot"):
                        st.plotly_chart(plx.scatter(df__filtered_concept, x = 'interval' ,y = 'time:timestamp'),use_container_width=True)
                    st.markdown(unsafe_allow_html=True,body = "<br>")
                    with st.expander("Line plot - Progression"):
                        df__filtered_concept['date'] = df__filtered_concept['time:timestamp'].dt.date
                        st.plotly_chart(plx.line(df__filtered_concept.groupby('date')['concept:name'].count().to_frame().reset_index(), x = 'date', y = 'concept:name'  ),use_container_width=True)
            else:
                o1,o2 = st.columns([2,5])
                with o1:
                    s_df = time_df_total_merged_aa.groupby('Variant_Name')['interval'].mean().to_frame().reset_index()
                    s_df_index =  s_df.loc[(s_df['Variant_Name'] == variant_selector),:].index.to_list()
                    s_df__index_plus = [s_df_index[0] + i for i in np.arange(10)]
                    #s_df__index_minus = [s_df_index[0] -i for i in np.arange(5)]
                    s_df_index_total = pd.concat([pd.Series(s_df_index),pd.Series(s_df__index_plus)]).to_list()
                    st.plotly_chart(plx.bar(s_df.loc[s_df_index_total,:], y ='Variant_Name', x = 'interval' ),use_container_width=True)
                with o2:
                    def quantile_25(x):
                        return x.quantile(0.25)
                    def quantile_75(x):
                        return x.quantile(0.75)
                    st.markdown(unsafe_allow_html=True,body = "<br>")
                    st.markdown(unsafe_allow_html=True,body = "<br>")
                    w_df  = time_df_total_merged_aa[time_df_total_merged_aa['Variant_Name'] == variant_selector]
                    w_df =  w_df.groupby('concept:name')['interval'].agg(['count','min','max','mean','std',quantile_25, quantile_75]).reset_index()
                    st.dataframe(w_df, use_container_width=True)

        st.markdown("""<hr>""",unsafe_allow_html=True)

        with st.expander('Variant and case identifier'):
            f1,f2 = st.columns(2)
            with f1:
                st.markdown("""<p style='text-align:center;font-weight:bold'> Find Variant per case <p>""",unsafe_allow_html =True)
                _sd_ = st.selectbox("Select1", time_df_total_merged_aa['Variant_Name'].unique(),index=13,label_visibility='hidden')
                _sd_1 = time_df_total_merged_aa[time_df_total_merged_aa['Variant_Name'] == _sd_ ]['Case'].unique().tolist()
                _sd_2 = ', '.join(_sd_1)
                st.markdown(unsafe_allow_html=True,body = """<p style='text-align:justify'>""" +_sd_2 + """</p>""" )
            with f2:
                st.markdown("""<p style=text-align:center;font-weight:bold'> Find Case per Variant <p>""",unsafe_allow_html =True)
                _ds_ = st.selectbox("Select12",time_df_total_merged_aa['Case'].unique(),label_visibility='hidden')
                _ds_1 = time_df_total_merged_aa[time_df_total_merged_aa['Case'] == _ds_]['Variant_Name'].unique()[0]
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown(unsafe_allow_html=True,body = """<p style='text-align:center'>""" +_ds_1 + """</p>""" )
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)

        with st.container():
            st.markdown("""
                <h1 style='text-align: center'>
                    Progress Anlysis
                </h1>""",unsafe_allow_html=True)
            select_process_ = st.selectbox('Choose Flow',["All Variants" ,"Best Path","Relevant Variants"], index = 2)
            summary_time_all = pd.read_csv(r'Dataframes\summary_time_all.csv')
            summary_time_relevant = pd.read_csv(r'Dataframes\summary_time_relevant.csv')
            summary_time_best = pd.read_csv(r'Dataframes\summary_time_best.csv')
            if select_process_ == "All Variants":
                g1,g2 = st.columns([1,1])
                with g1:
                    im_all_variants = Image.open('Images\Heuristic-net-allvariants.png')
                    st.image(im_all_variants, use_column_width=True)
                with g2:
                    st.markdown("""<hr>""",unsafe_allow_html=True)
                    t1,t2,t3 = st.columns(3)
                    with t1:
                        st.markdown("""<b><h3 style='text-align:center'> Min time </h3></b>""",unsafe_allow_html =True)
                        _s_ =st.select_d = st.selectbox('select',label_visibility='hidden',options= summary_time_all.columns.to_list())
                        min_hour_s_ = summary_time_all.loc[0,_s_]
                        st.markdown("""<p style='text-align:center'>""" + str(min_hour_s_) + """</p>""",unsafe_allow_html=True) 
                    with t2:
                        st.markdown("""<b><h3 style='text-align:center'> Avg time </h3></b>""",unsafe_allow_html =True)
                        _x_ =st.select_d = st.selectbox('select_!',label_visibility='hidden',options= summary_time_all.columns.to_list())
                        mean_hour_x_ = summary_time_all.loc[1,_x_]
                        st.markdown("""<p style='text-align:center'>""" + str(round(mean_hour_x_,2)) + """</p>""",unsafe_allow_html=True)
                    with t3:
                        st.markdown("""<b><h3 style='text-align:center'> Max time </h3></b>""",unsafe_allow_html =True)
                        _r_ =st.select_d = st.selectbox('select_',label_visibility='hidden',options= summary_time_all.columns.to_list())
                        max_hour_r_ = summary_time_all.loc[2,_r_]
                        st.markdown("""<p style='text-align:center'>""" + str(round(max_hour_r_,2)) + """</p>""",unsafe_allow_html=True)
                    st.markdown("""<hr>""",unsafe_allow_html=True)
                    st.markdown("""<b><h3 style='text-align:center'> Time per activity </h3></b>""",unsafe_allow_html =True)
                    dataframe_all_variant_act = pd.read_csv(r'Dataframes\dataframe_all_variant_act_full.csv')
                    time_selector_1 = st.select_slider('Hours - Days',dataframe_all_variant_act['time'].unique(),label_visibility='hidden')
                    st.dataframe(dataframe_all_variant_act[dataframe_all_variant_act['time'] == time_selector_1],use_container_width=True)
                st.markdown("<hr>",unsafe_allow_html=True)

            if select_process_ == 'Best Path':
                g1,g2 = st.columns([1,2])
                with g1:
                    im_best = Image.open('Images\Heuristic-net-best.png')
                    im_best = im_best.resize((380,670))
                    st.image(im_best)
                with g2:
                    st.markdown("""<hr>""",unsafe_allow_html=True)
                    m1,m2,m3 = st.columns(3)
                    with m1:
                        st.markdown("""<b><h3 style='text-align:center'> Min time </h3></b>""",unsafe_allow_html =True)
                        _m_ =st.select_d = st.selectbox('selectyy',label_visibility='hidden',options= summary_time_best.columns.to_list())
                        min_hour_s_ = summary_time_best.loc[0,_m_]
                        st.markdown("""<p style='text-align:center'>""" + str(round(min_hour_s_,2)) + """</p>""",unsafe_allow_html=True)
                    with m2:
                        st.markdown("""<b><h3 style='text-align:center'> Mean time </h3></b>""",unsafe_allow_html =True)
                        _cv_ =st.select_d = st.selectbox('selectyeey',label_visibility='hidden',options= summary_time_best.columns.to_list())
                        mean_hour_s_ = summary_time_best.loc[1,_cv_]
                        st.markdown("""<p style='text-align:center'>""" + str(round(mean_hour_s_,2)) + """</p>""",unsafe_allow_html=True)
                    with m3:
                        st.markdown("""<b><h3 style='text-align:center'> Max time </h3></b>""",unsafe_allow_html =True)
                        _cd_ =st.select_d = st.selectbox('selectaayy',label_visibility='hidden',options= summary_time_best.columns.to_list())
                        max_hour_s_ = summary_time_best.loc[2,_cd_]
                        st.markdown("""<p style='text-align:center'>""" + str(round(max_hour_s_,2)) + """</p>""",unsafe_allow_html=True)
                    dataframe_best_act_full = pd.read_csv(r'Dataframes\dataframe_best_act_full.csv')
                    d_select = st.select_slider("sdd",label_visibility='hidden',options = dataframe_best_act_full['time'].unique())
                    st.dataframe(dataframe_best_act_full[dataframe_best_act_full['time'] == d_select],use_container_width=True)
            
            if select_process_ == "Relevant Variants":
                g1,g2 = st.columns([1,2])
                with g1:
                    im_relevant = Image.open('Images\Heuristic-net-relevant.png')
                    im_relevant = im_relevant.resize((450,800))
                    st.image(im_relevant)
                with g2:
                    y1,y2,y3 = st.columns(3)
                    with y1:
                        st.markdown("""<b><h3 style='text-align:center'> Min time </h3></b>""",unsafe_allow_html =True)
                        _s_ =st.select_d = st.selectbox('selectqq',label_visibility='hidden',options= summary_time_relevant.columns.to_list())
                        min_hour_s_ = summary_time_relevant.loc[0,_s_]
                        st.markdown("""<p style='text-align:center'>""" + str(min_hour_s_) + """</p>""",unsafe_allow_html=True)
                    with y2: 
                        st.markdown("""<b><h3 style='text-align:center'> Avg time </h3></b>""",unsafe_allow_html =True)
                        _x_ =st.select_d = st.selectbox('select_!qq',label_visibility='hidden',options= summary_time_relevant.columns.to_list())
                        mean_hour_x_ = summary_time_relevant.loc[1,_x_]
                        st.markdown("""<p style='text-align:center'>""" + str(round(mean_hour_x_,2)) + """</p>""",unsafe_allow_html=True)
                    with y3: 
                        st.markdown("""<b><h3 style='text-align:center'> Max time </h3></b>""",unsafe_allow_html =True)
                        _d_ =st.select_d = st.selectbox('select_11!qq',label_visibility='hidden',options= summary_time_relevant.columns.to_list())
                        mean_hour_x_ = summary_time_relevant.loc[2,_d_]
                        st.markdown("""<p style='text-align:center'>""" + str(round(mean_hour_x_,2)) + """</p>""",unsafe_allow_html=True)
                    dataframe_relevant_act_ = pd.read_csv('Dataframes\dataframe_all_relevant_act_full.csv')
                    sele_timer = st.select_slider("Hours - Days",dataframe_relevant_act_['time'].unique(),label_visibility='hidden')
                    st.dataframe(dataframe_relevant_act_[dataframe_relevant_act_['time']==sele_timer], use_container_width=True)
                st.markdown("<hr>",unsafe_allow_html=True)

        st.markdown("<hr>",unsafe_allow_html=True)
        space_plot_final =  st.container()
        with space_plot_final:
            colr1, colr2 = st.columns([1,6])
            with colr1:
                global __s1,__s2,__s3,__s4,__s5
                __s1 = st.text_input("case 1",max_chars=3,value ='B')
                __s2 = st.text_input("case 2",max_chars=3,value ='B')
                __s3 = st.text_input("case 3",max_chars=3,value ='ZAZ')
                __s4 = st.text_input("case 4",max_chars=3,value ='Q')
                __s5 = st.text_input("case 5",max_chars=3,value ='B')
                  
        #df_for_plot_end
            with colr2:
                dff_1 =  time_df_total_merged_aa[time_df_total_merged_aa['Case'] == __s1 ]
                dff_2 =  time_df_total_merged_aa[time_df_total_merged_aa['Case'] == __s2 ]
                dff_3 =  time_df_total_merged_aa[time_df_total_merged_aa['Case'] == __s3 ]
                dff_4 =  time_df_total_merged_aa[time_df_total_merged_aa['Case'] == __s4 ]
                dff_5 =  time_df_total_merged_aa[time_df_total_merged_aa['Case'] == __s5 ]
                df_plot = pd.concat([dff_1,dff_2,dff_3,dff_4,dff_5])
                st.plotly_chart(plx.line(df_plot, x = 'concept:name',y = 'time:timestamp',markers=True,color= 'case:concept:name',height=650),use_container_width=True)       
                 
if selectbox == 'process':
    st.markdown("""
            <h1 style='text-align: center'>
                Process
            </h1>
            """,unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    
    select_process = st.selectbox('Choose Variants',["All Variants" ,"Best Path","Relevant Variants"], index = 0)
    
    if select_process == "All Variants":
        #Container 1  - Porcess all variants 
        all_variants_process_1 = st.container()
        with all_variants_process_1:
            col1__,col2__,  = st.columns([1,1])
            with col1__:
                st.markdown("""<h3 style='text-align: center'>Heurist Net</h3>""",unsafe_allow_html=True)
                st.image('Images\Heuristic-net-allvariants.png',use_column_width=True)
            with col2__:
                st.markdown("""<h3 style='text-align:center'> Starting Activities </h3>""",unsafe_allow_html=True)
                start_Activities_all_variants = pd.read_csv(r'Dataframes\start_Activities_all_variants.csv')
                start_Activities_all_variants = start_Activities_all_variants.drop(start_Activities_all_variants.columns.to_list()[0],axis =1)
                start_Activities_all_variants
                st.markdown("""<h3 style='text-align:center'> End Activities </h3>""",unsafe_allow_html=True)
                end_activities_allvariants = pd.read_csv(r'Dataframes\end_activities_allvariants.csv')
                end_activities_allvariants = end_activities_allvariants.drop(end_activities_allvariants.columns.to_list()[0],axis =1)
                end_activities_allvariants_ = end_activities_allvariants.T
                end_activities_allvariants_.columns = end_activities_allvariants_.loc['ends',:]
                end_activities_allvariants_ = end_activities_allvariants_.drop('ends',axis=0)
                end_activities_allvariants_
                st.markdown("""<h3 style='text-align:center'> Download net </h3>""",unsafe_allow_html=True)
                im = Image.open(r'Images\Heuristic-net-allvariants.png')
                buf = BytesIO()
                im.save(buf,format="png")
                byte_im = buf.getvalue()
                st.download_button("Click to Download", data=byte_im , mime="image/jpeg")
            st.markdown("<hr>",unsafe_allow_html=True)
            with st.expander('Aditional Info'):
                st.markdown("""<h3 style='text-align:center'> % of Ending Activities  </h3>""",unsafe_allow_html=True)
                plot_ends_all_variants = plx.bar(end_activities_allvariants, x = 'Percentage',y='ends')
                st.plotly_chart(plot_ends_all_variants, use_container_width=True)
            st.markdown("<hr>",unsafe_allow_html=True)
            with st.container():
                st.markdown("""<h3 style='text-align:center'>  Business Process Model  </h3>""",unsafe_allow_html=True)
                st.image(r'Images\bpmn_allvariants.png')
                co1,co2,co3,co4,co5 = st.columns(5)
                with co3:
                    im_bp_all = Image.open(r'Images\bpmn_allvariants.png')
                    buf__ = BytesIO()
                    im_bp_all.save(buf__,format='png')
                    bytes_im_bp = buf__.getvalue()
                    st.download_button("Click to Download", data=bytes_im_bp , mime="image/jpeg")
            st.markdown("<hr>",unsafe_allow_html=True)
            st.markdown("""<h3 style='text-align:center'>  Variants  </h3>""",unsafe_allow_html=True)
            variants_total_df__ = pd.read_csv(r'Dataframes\allvariants-details.csv')
            __  = []
            for trace in variants_total_df__['variant']:
                x = list(trace)
                __.append(len(x))
            variants_total_df__['len_variant'] = __
            variants_total_df__['% ocu'] = variants_total_df__['len_Data'] / variants_total_df__['len_Data'].sum() * 100
            with st.container():
                ca,cb,cc = st.columns([2,1,1])
                with ca:
                    st.markdown("""<h4 style='text-align:center'> Select Variant </h4>""",unsafe_allow_html=True)
                    selected_variant = st.selectbox('Select Variant',variants_total_df__['variant_name'],label_visibility='hidden')
                with cb:
                    st.markdown("""<h4 style='text-align:center'> Length of the Trace </h4>""",unsafe_allow_html=True)
                    a = variants_total_df__[variants_total_df__['variant_name'] == selected_variant]['variant'].values[0]
                    st.info("Number of elements of the trace : {}".format(len(list(a.split(",")))))
                    a_ = variants_total_df__[variants_total_df__['variant_name'] == selected_variant]['% ocu'].values[0]                   
                    st.info("Percentage of occurance of this trace: {:.2f}".format(a_))
                with cc:
                    st.markdown("""<h4 style='text-align:center'> Elements of the Trace </h4>""",unsafe_allow_html=True)
                    a = variants_total_df__[variants_total_df__['variant_name'] == selected_variant]['variant'].values[0]
                    string_variant_=  ("".join(str(w) for w in a).replace("(","").replace(")","").replace("'","").replace(","," > "))
                    string_variant = """<p style='text-align:center'>""" + string_variant_ + "</p>"
                    st.markdown(string_variant_,unsafe_allow_html=True)
            with st.container():
                type_radio = st.radio('Choose', options= ['Length of the Variant', 'Ocurrance of the variant'],horizontal=True,label_visibility='hidden')
                if type_radio == 'Length of the Variant':
                    len_plot = plx.bar(variants_total_df__,x = 'variant_name', y = 'len_variant')
                    st.plotly_chart(len_plot,use_container_width=True)
                else:
                    len_m_plot = plx.bar(variants_total_df__, x ='variant_name',y="% ocu")
                    st.plotly_chart(len_m_plot,use_container_width=True)
            st.markdown("<hr>",unsafe_allow_html=True)
            with st.container():
                st.markdown("""<h3 style='text-align:center'>  Aditional Process Maps  </h3>""",unsafe_allow_html=True)
                petri_inductive_img = Image.open(r'Images\petri_net__inductive_all_variants.png')
                buf_pet_ind = BytesIO()
                petri_inductive_img.save(buf_pet_ind,format='png')
                bytes_petri_inductive = buf_pet_ind.getvalue()
                petri_alpha_img = Image.open(r'Images\Petri_net_alpha_allvariants.png')
                buf_pet_alpha = BytesIO()
                petri_alpha_img.save(buf_pet_alpha,format='png')
                bytes_petri_alpha = buf_pet_alpha.getvalue()
                petri_alphaplus_img = Image.open(r'Images\Petri_net_alphaplus_best.png')
                buf_pet_alphaplus = BytesIO()
                petri_alphaplus_img.save(buf_pet_alphaplus,format='png')
                bytes_petri_alphaplus = buf_pet_alphaplus.getvalue()
                s1,s2 = st.columns([8,2])
                with s1:
                    st.markdown("""<h4 style='text-align:center'>  Petri Net - Inductive  </h4>""",unsafe_allow_html=True)
                    st.image(petri_inductive_img,use_column_width=True)
                with s2:
                    Inductive_allvar = st.download_button("Download Petri Net  - Inductive",data = bytes_petri_inductive, mime="image/jpeg")
                    alpha_allvat = st.download_button("Download Petri Net  - Alpha Algo",data = bytes_petri_alpha, mime="image/jpeg")
                    alphaplus_allvar = st.download_button("Download Petri Net  - Alpha plus Algo",data = bytes_petri_alphaplus, mime="image/jpeg")

    if select_process == "Best Path":
        best_run = st.container()
        with best_run:
            col1__,col2__,  = st.columns([1,1])
            with col1__:
                st.markdown("""<h3 style='text-align: center'>Heurist Net</h3>""",unsafe_allow_html=True)
                i_img = Image.open(r'Images\Heuristic-net-best.png')
                i_img_ = i_img.resize((400,650))
                st.image(i_img_,use_column_width=False)
            with col2__:
                st.markdown("""<h3 style='text-align:center'> Starting Activities </h3>""",unsafe_allow_html=True)
                start_Activities_best= pd.read_csv(r'Dataframes\start_activities_best.csv')
                start_Activities_best = start_Activities_best.drop(start_Activities_best.columns.to_list()[0],axis =1)
                start_Activities_best
                st.markdown("""<h3 style='text-align:center'> End Activities </h3>""",unsafe_allow_html=True)
                end_activities_best = pd.read_csv(r'Dataframes\end_activities_best.csv')
                end_activities_best = end_activities_best.drop(end_activities_best.columns.to_list()[0],axis=1)
                end_activities_best_ = end_activities_best.T
                end_activities_best_.columns = end_activities_best_.loc['Activity',:]
                end_activities_best_ = end_activities_best_.drop('Activity',axis=0)
                end_activities_best_
                st.markdown("""<h3 style='text-align:center'> Download net </h3>""",unsafe_allow_html=True)
                buf_net_best = BytesIO()
                i_img.save(buf_net_best,format ='png')
                byte_im_best = buf_net_best.getvalue()
                st.download_button("Click to Download", data = byte_im_best, mime= "image/jpeg")
        st.markdown("<hr>",unsafe_allow_html=True)
        with st.expander("Additional Info"):
            st.markdown("""<h3 style='text-align:center'> % of Ending Activities  </h3>""",unsafe_allow_html=True)
            plot_ends_best = plx.bar(end_activities_best, x = 'Percentage',y='Activity')
            st.plotly_chart(plot_ends_best, use_container_width=True)
        st.markdown("<hr>",unsafe_allow_html=True)
        with st.container():
                st.markdown("""<h3 style='text-align:center'>  Business Process Model  </h3>""",unsafe_allow_html=True)
                st.image(r'Images\bpmn_best.png',use_column_width=True)
                co1__a,co2__a,co3__a,co4__a,co5__a = st.columns(5)
                with co3__a:
                    im_bp_best = Image.open(r'Images\bpmn_best.png')
                    buf__best = BytesIO()
                    im_bp_best.save(buf__best,format='png')
                    bytes_im_bp_best = buf__best.getvalue()
                    st.download_button("Click to Download", data=bytes_im_bp_best , mime="image/jpeg")
        st.markdown("<hr>",unsafe_allow_html=True)
        st.markdown("""<h3 style='text-align:center'>  Variants  </h3>""",unsafe_allow_html=True)
        variants_best_df__ = pd.read_csv(r'Dataframes\best_variants_details.csv')
        __  = []
        for trace in variants_best_df__['variant']:
            x = list(trace)
            __.append(len(x))
        variants_best_df__['len_variant'] = __
        #variants_best_df__.columns.rename({"%":"% ocu"})
        with st.container():
            ca_,cb_,cc_ = st.columns([2,1,1])
            with ca_:
                st.markdown("""<h4 style='text-align:center'> Select Variant </h4>""",unsafe_allow_html=True)
                selected_variant = st.selectbox('Select Variant',variants_best_df__['variant_name'],label_visibility='hidden')
            with cb_:
                st.markdown("""<h4 style='text-align:center'> Length of the Trace </h4>""",unsafe_allow_html=True)
                a = variants_best_df__[variants_best_df__['variant_name'] == selected_variant]['variant'].values[0]
                st.info("Number of elements of the trace : {}".format(len(list(a.split(",")))))
                a_ = variants_best_df__[variants_best_df__['variant_name'] == selected_variant]['%'].values[0]                   
                st.info("Percentage of occurance of this trace: {:.2f}".format(a_))
            with cc_:
                st.markdown("""<h4 style='text-align:center'> Elements of the Trace </h4>""",unsafe_allow_html=True)
                a = variants_best_df__[variants_best_df__['variant_name'] == selected_variant]['variant'].values[0]
                string_variant_=  ("".join(str(w) for w in a).replace("(","").replace(")","").replace("'","").replace(","," > "))
                string_variant = """<p style='text-align:center'>""" + string_variant_ + "</p>"
                st.markdown(string_variant_,unsafe_allow_html=True)
        with st.container():
                type_radio = st.radio('Choose', options= ['Length of the Variant', 'Ocurrance of the variant'],horizontal=True,label_visibility='hidden')
                if type_radio == 'Length of the Variant':
                    len_plot = plx.bar(variants_best_df__,x = 'variant_name', y = 'len_variant')
                    st.plotly_chart(len_plot,use_container_width=True)
                else:
                    len_m_plot = plx.bar(variants_best_df__, x ='variant_name',y="%")
                    st.plotly_chart(len_m_plot,use_container_width=True)
        st.markdown("<hr>",unsafe_allow_html=True)
        with st.container():
            st.markdown("""<h3 style='text-align:center'>  Aditional Process Maps  </h3>""",unsafe_allow_html=True)
            petri_inductive_img = Image.open(r'Images\petri_net_inductive_best.png')
            buf_pet_ind = BytesIO()
            petri_inductive_img.save(buf_pet_ind,format='png')
            bytes_petri_inductive = buf_pet_ind.getvalue()
            petri_alpha_img = Image.open(r'Images\Petri_net_alpha_best.png')
            buf_pet_alpha = BytesIO()
            petri_alpha_img.save(buf_pet_alpha,format='png')
            bytes_petri_alpha = buf_pet_alpha.getvalue()
            petri_alphaplus_img = Image.open(r'Images\Petri_net_alphaplus_best.png')
            buf_pet_alphaplus = BytesIO()
            petri_alphaplus_img.save(buf_pet_alphaplus,format='png')
            bytes_petri_alphaplus = buf_pet_alphaplus.getvalue()
            s1,s2 = st.columns([8,2])
            with s1:
                st.markdown("""<h4 style='text-align:center'>  Petri Net - Inductive  </h4>""",unsafe_allow_html=True)
                st.image(petri_inductive_img,use_column_width=True)
            with s2:
                Inductive_allvar = st.download_button("Download Petri Net  - Inductive",data = bytes_petri_inductive, mime="image/jpeg")
                alpha_allvat = st.download_button("Download Petri Net  - Alpha Algo",data = bytes_petri_alpha, mime="image/jpeg")
                alphaplus_allvar = st.download_button("Download Petri Net  - Alpha plus Algo",data = bytes_petri_alphaplus, mime="image/jpeg")

    if select_process == "Relevant Variants":
        relevant = st.container()
        with relevant:
            col1__,col2__,  = st.columns([1,1])
            with col1__:
                st.markdown("""<h3 style='text-align: center'>Heurist Net</h3>""",unsafe_allow_html=True)
                i_img = Image.open(r'Images\Heuristic-net-relevant.png')
                i_img_ = i_img.resize((400,650))
                st.image(i_img_,use_column_width=False)
            with col2__:
                st.markdown("""<h3 style='text-align:center'> Starting Activities </h3>""",unsafe_allow_html=True)
                start_Activities_best= pd.read_csv(r'Dataframes\start_activities_relevant.csv')
                start_Activities_best = start_Activities_best.drop(start_Activities_best.columns.to_list()[0],axis =1)
                start_Activities_best
                st.markdown("""<h3 style='text-align:center'> End Activities </h3>""",unsafe_allow_html=True)
                end_activities_best = pd.read_csv(r'Dataframes\end_activities_relevant.csv')
                end_activities_best = end_activities_best.drop(end_activities_best.columns.to_list()[0],axis=1)
                end_activities_best_ = end_activities_best.T
                end_activities_best_.columns = end_activities_best_.loc['Activity',:]
                end_activities_best_ = end_activities_best_.drop('Activity',axis=0)
                end_activities_best_
                st.markdown("""<h3 style='text-align:center'> Download net </h3>""",unsafe_allow_html=True)
                buf_net_best = BytesIO()
                i_img.save(buf_net_best,format ='png')
                byte_im_best = buf_net_best.getvalue()
                st.download_button("Click to Download", data = byte_im_best, mime= "image/jpeg")
        st.markdown("<hr>",unsafe_allow_html=True)
        with st.expander("Additional Info"):
            st.markdown("""<h3 style='text-align:center'> % of Ending Activities  </h3>""",unsafe_allow_html=True)
            plot_ends_best = plx.bar(end_activities_best, x = 'Percentage',y='Activity')
            st.plotly_chart(plot_ends_best, use_container_width=True)
        st.markdown("<hr>",unsafe_allow_html=True)
        with st.container():
                st.markdown("""<h3 style='text-align:center'>  Business Process Model  </h3>""",unsafe_allow_html=True)
                st.image(r'Images\bpmn_relevant.png',use_column_width=True)
                co1__a,co2__a,co3__a,co4__a,co5__a = st.columns(5)
                with co3__a:
                    im_bp_best = Image.open(r'Images\bpmn_relevant.png')
                    buf__best = BytesIO()
                    im_bp_best.save(buf__best,format='png')
                    bytes_im_bp_best = buf__best.getvalue()
                    st.download_button("Click to Download", data=bytes_im_bp_best , mime="image/jpeg")
        st.markdown("<hr>",unsafe_allow_html=True)
        st.markdown("""<h3 style='text-align:center'>  Variants  </h3>""",unsafe_allow_html=True)
        variants_best_df__ = pd.read_csv(r'Dataframes\relevant_variants_details.csv')
        __  = []
        for trace in variants_best_df__['variant']:
            x = list(trace)
            __.append(len(x))
        variants_best_df__['len_variant'] = __
        #variants_best_df__.columns.rename({"%":"% ocu"})
        with st.container():
            ca_,cb_,cc_ = st.columns([2,1,1])
            with ca_:
                st.markdown("""<h4 style='text-align:center'> Select Variant </h4>""",unsafe_allow_html=True)
                selected_variant = st.selectbox('Select Variant',variants_best_df__['variant_name'],label_visibility='hidden')
            with cb_:
                st.markdown("""<h4 style='text-align:center'> Length of the Trace </h4>""",unsafe_allow_html=True)
                a = variants_best_df__[variants_best_df__['variant_name'] == selected_variant]['variant'].values[0]
                st.info("Number of elements of the trace : {}".format(len(list(a.split(",")))))
                a_ = variants_best_df__[variants_best_df__['variant_name'] == selected_variant]['%'].values[0]                   
                st.info("Percentage of occurance of this trace: {:.2f}".format(a_))
            with cc_:
                st.markdown("""<h4 style='text-align:center'> Elements of the Trace </h4>""",unsafe_allow_html=True)
                a = variants_best_df__[variants_best_df__['variant_name'] == selected_variant]['variant'].values[0]
                string_variant_=  ("".join(str(w) for w in a).replace("(","").replace(")","").replace("'","").replace(","," > "))
                string_variant = """<p style='text-align:center'>""" + string_variant_ + "</p>"
                st.markdown(string_variant_,unsafe_allow_html=True)
        with st.container():
                type_radio = st.radio('Choose', options= ['Length of the Variant', 'Ocurrance of the variant'],horizontal=True,label_visibility='hidden')
                if type_radio == 'Length of the Variant':
                    len_plot = plx.bar(variants_best_df__,x = 'variant_name', y = 'len_variant')
                    st.plotly_chart(len_plot,use_container_width=True)
                else:
                    len_m_plot = plx.bar(variants_best_df__, x ='variant_name',y="%")
                    st.plotly_chart(len_m_plot,use_container_width=True)
        st.markdown("<hr>",unsafe_allow_html=True)
        with st.container():
            st.markdown("""<h3 style='text-align:center'>  Aditional Process Maps  </h3>""",unsafe_allow_html=True)
            petri_inductive_img = Image.open(r'Images\petri_net_inductive_relevant.png')
            buf_pet_ind = BytesIO()
            petri_inductive_img.save(buf_pet_ind,format='png')
            bytes_petri_inductive = buf_pet_ind.getvalue()
            petri_alpha_img = Image.open(r'Images\Petri_net_alpha_best.png')
            buf_pet_alpha = BytesIO()
            petri_alpha_img.save(buf_pet_alpha,format='png')
            bytes_petri_alpha = buf_pet_alpha.getvalue()
            petri_alphaplus_img = Image.open(r'Images\Petri_net_alphaplus_best.png')
            buf_pet_alphaplus = BytesIO()
            petri_alphaplus_img.save(buf_pet_alphaplus,format='png')
            bytes_petri_alphaplus = buf_pet_alphaplus.getvalue()
            s1,s2 = st.columns([8,2])
            with s1:
                st.markdown("""<h4 style='text-align:center'>  Petri Net - Inductive  </h4>""",unsafe_allow_html=True)
                st.image(petri_inductive_img,use_column_width=True)
            with s2:
                Inductive_allvar = st.download_button("Download Petri Net  - Inductive",data = bytes_petri_inductive, mime="image/jpeg")
                alpha_allvat = st.download_button("Download Petri Net  - Alpha Algo",data = bytes_petri_alpha, mime="image/jpeg")
                alphaplus_allvar = st.download_button("Download Petri Net  - Alpha plus Algo",data = bytes_petri_alphaplus, mime="image/jpeg")

if selectbox =='data':
    st.markdown("""
    <h1 style='text-align:center'>
        Data
    </h1>
    """,unsafe_allow_html= True)
    st.markdown(
        """
        Take a look at a sample of the Data, click the link below the table to download the file.
        This file is not human readable, please have a .XES.gz file reader if you want to explore for yourslef of use python  + pm4py
        """
        )
    #ss = eventlog_df.copy()
    #ss = ss.loc[(ss['case:concept:name'].isin(['A','B','ZZZ']))]
    #ss.to_csv('sample_eventolog.csv')
    sample_data = pd.read_csv(r'Dataframes\sample_eventolog.csv')
    sample_data = sample_data.drop(sample_data.columns.to_list()[0], axis =1)
    st.dataframe(sample_data,use_container_width=True)

    st.markdown("""<a href="https://data.4tu.nl/ndownloader/files/24058772" download>Download XES file</a>""",unsafe_allow_html=True)

    st.markdown("<hr>",unsafe_allow_html=True)

    with st.container():
        cl_a , cl_b = st.columns(2)
        with cl_a:
            st.subheader("Description")
            st.markdown("""
            <p align="justify">
                The 'Hospital Billing' event log was obtained from the financial modules of the
                ERP system of a regional hospital. The event log contains events that are 
                related to the billing of medical services that have been provided by the 
                hospital. Each trace of the event log records the activities executed to bill 
                a package of medical services that were bundled together. The event log does 
                not contain information about the actual medical services provided by the 
                hospital. The 100,000 traces in the event log are a random sample of process instances 
                that were recorded over three years. Several attributes such as the 'state' of
                the process, the 'caseType', the underlying 'diagnosis' etc. are included in 
                the event log. Events and attribute values have been anonymized. The time 
                stamps of events have been randomized for this purpose, but the time between
                events within a trace has not been altered. More information about the event log can be found in the related publications.Please cite as:
                Mannhardt, F (Felix) (2017) Hospital Billing - Event Log. 
                Eindhoven University of Technology. Dataset. 
                https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741
            </p>
            """,unsafe_allow_html=True)
        with cl_b:
            st.markdown("""
            <h3 style='text-align:center'>
                Atributes
            </h1>
            """,unsafe_allow_html=True)
            attributes = pd.read_csv(r"Dataframes\Dataset Atrbutes.csv")
            attributes
            




