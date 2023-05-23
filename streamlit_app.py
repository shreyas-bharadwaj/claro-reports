import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
import altair as alt

st.set_page_config(layout='wide')
today_date = pd.to_datetime('today').date()

cols_patient = ['Patient ID', 'Current Care Coordinator', 'Currently Active', 'Most Recent Randomization Date']
patient = pd.read_csv("./data/patient.csv", usecols=cols_patient)

cols_episode = ['Patient ID', 'Organization Name', 'Clinic Name']
episode = pd.read_csv("./data/episode.csv", usecols=cols_episode)

agencies = pd.read_csv('./data/agencies.csv')

# Filter the required Organization Names once to avoid duplication
org_names_filtered = episode[~episode['Organization Name'].isin(['Hidalgo Medical Services', 'University of New Mexico (Non-current)'])]

#selector
option = st.selectbox('Select an Organization', org_names_filtered['Organization Name'].unique())
site_name = option
clinics = org_names_filtered[org_names_filtered['Organization Name']==site_name]['Clinic Name'].unique()

st.header("CLARO Site Report | Version 2.0") 
st.subheader(f"data current as of {today_date}")

with st.container():
    col1, col2 = st.columns(2)

    #first chart patient count by CC
    with col1:
        chart1 = patient[patient['Currently Active']==1].copy()
        chart1['Current Care Coordinator'] = chart1['Current Care Coordinator'].str.split().str[0]
        merge = org_names_filtered[org_names_filtered['Organization Name']==site_name]
        merge = pd.merge(chart1, merge, on='Patient ID')
        chart1 = merge['Current Care Coordinator'].value_counts().reset_index()
        chart1.columns = ['Current Care Coordinator', 'Patient ID']
        chart1 = chart1.sort_values(by='Patient ID', ascending=False)
        ticks = np.arange(0, chart1['Patient ID'].max()+1, 3).tolist()
        fig, ax = plt.subplots()
        plt.title("Caseload Size by CC")
        plt.xlabel("# Patients")
        plt.ylabel("CC")
        plt.xticks(ticks)
        ax = sns.barplot(data=chart1, x='Patient ID', y='Current Care Coordinator', orient='h')
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)

        #third chart 
        chart3 = patient[patient['Currently Active']==1].copy()
        chart3['Most Recent Randomization Date'] = chart3['Most Recent Randomization Date'].astype('datetime64')
        chart3e = merge.copy()
        chart3e['today'] = today_date
        chart3e['today'] = chart3e['today'].astype('datetime64')
        chart3 = pd.merge(chart3, chart3e, on='Patient ID')
        chart3['# Days Enrolled'] = (chart3['today'] - chart3['Most Recent Randomization Date']).dt.days
        chart3 = chart3.groupby('Current Care Coordinator')['# Days Enrolled'].median().reset_index()
        chart3.columns = ['Current Care Coordinator', '# Days Enrolled (median)']
        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart3, x='# Days Enrolled (median)', y='Current Care Coordinator', orient='h')
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
