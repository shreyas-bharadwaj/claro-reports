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

        chart3 = patient[['Patient ID', 'Current Care Coordinator', 'Most Recent Randomization Date', 'Currently Active']]
        chart3 = chart3[chart3['Currently Active']==1]
        chart3['Most Recent Randomization Date'] = chart3['Most Recent Randomization Date'].astype('datetime64')
        chart3e = episode[['Patient ID', 'Organization Name']]
        chart3e = chart3e[chart3e['Organization Name']==site_name]
        chart3e['today'] = pd.to_datetime('today').date()
        chart3e['today'] = chart3e['today'].astype('datetime64')
        chart3 = pd.merge(chart3, chart3e, how='inner', on='Patient ID')
        chart3['# Days Enrolled'] = (chart3['today'] - chart3['Most Recent Randomization Date']).dt.days
        chart3 = chart3[['Current Care Coordinator', '# Days Enrolled']].groupby('Current Care Coordinator').median().reset_index()
        chart3['# Days Enrolled (median)'] = chart3['# Days Enrolled']
        fig, ax = plt.subplots() #solved by add this line 
        ax = sns.barplot(data=chart3, x='# Days Enrolled (median)', y='Current Care Coordinator', orient='h')
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)

    cols_patient = ['Patient ID', 'Current Care Coordinator', 'Currently Active', 'Current Primary Clinic']
    patient = pd.read_csv("./data/patient.csv", usecols=cols_patient)

    cols_contact_note = ['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']
    contact_note = pd.read_csv("./data/contact_note.csv", usecols=cols_contact_note)

    with col2:
        # second chart
        chart1 = patient[(patient['Currently Active']==1) & (patient['Current Primary Clinic'].isin(clinics))]
        chart1 = chart1['Current Primary Clinic'].value_counts().reset_index()
        chart1.columns = ['Current Primary Clinic', 'Patient ID']
        chart1 = chart1.sort_values(by='Patient ID', ascending=False)
        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart1, x='Patient ID', y='Current Primary Clinic', orient='h')
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)

        # fourth chart
        chart4 = contact_note[contact_note['Contact Type']=='I/A'].copy()
        chart4['Contact Date'] = chart4['Contact Date'].astype('datetime64')
        chart4e = org_names_filtered.copy()
        chart4e['Randomization Date'] = chart4e['Randomization Date'].astype('datetime64')
        chart4 = pd.merge(chart4, chart4e, on='Patient ID')
        chart4['# Days to Initial Visit (median)'] = (chart4['Contact Date'] - chart4['Randomization Date']).dt.days
        chart4 = chart4.groupby('Patient Clinic Names')['# Days to Initial Visit (median)'].median().reset_index()
        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart4, x='# Days to Initial Visit (median)', y='Patient Clinic Names', orient='h')
        plt.title("# of Days to Initial Visit (median) by Clinic")
        plt.xlabel("# Days (median)")
        plt.ylabel("Clinic")
        plt.axvline(7)
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)

        # fifth chart
        chart5 = contact_note[contact_note['Contact Type']=='I/A'].copy()
        chart5['Contact Date'] = chart5['Contact Date'].astype('datetime64')
        chart5['30 Days Ago'] = today_date - pd.Timedelta(days=31)
        chart5 = chart5[chart5['Provider Organization Names']==site_name]
        chart5 = chart5[chart5['Contact Date']>=chart5['30 Days Ago']]
        chart5 = chart5.groupby('Patient Clinic Names')['Patient ID'].nunique().reset_index()
        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart5, x='Patient ID', y='Patient Clinic Names', orient='h')
        ticks = np.arange(0, chart5['Patient ID'].max()+1, 1).tolist()
        plt.title("# Patients w/ Initial Visit in Last 30 Days by Clinic")
        plt.xlabel("# Patients")
        plt.ylabel("Clinic")
        plt.xticks(ticks)
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
