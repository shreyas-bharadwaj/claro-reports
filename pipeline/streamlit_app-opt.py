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

cols_episode = ['Patient ID', 'Care Coordinator Initial Encounter Date', '# Care Coordinator Encounter', 'Organization Name', 'Last Care Coordinator Encounter Date']
episode = pd.read_csv("./data/episode.csv", usecols=cols_episode)

# chart 7: Patient Engagement Status By Clinic
chart7 = episode[episode['Organization Name']==site_name].copy()
chart7['Today'] = today_date
chart7['Last Care Coordinator Encounter Date'] = chart7['Last Care Coordinator Encounter Date'].astype('datetime64')

chart7p = patient[['Patient ID', 'Most Recent Randomization Date', 'Currently Active', 'Current Primary Clinic']].copy()
chart7p['Most Recent Randomization Date'] = chart7p['Most Recent Randomization Date'].astype('datetime64')

chart7 = pd.merge(chart7, chart7p, on='Patient ID', how='inner')
chart7['Days Since Last CC Visit'] = (chart7['Today'] - chart7['Last Care Coordinator Encounter Date']).dt.days
chart7['Days Since Randomized'] = (chart7['Today'] - chart7['Most Recent Randomization Date']).dt.days

def genStatus(row):
    if (row['Days Since Randomized'] <= 30 and pd.isnull(row['Care Coordinator Initial Encounter Date'])):
        return 'Recently Enrolled'
    elif (row['Days Since Randomized'] >= 31 and pd.isnull(row['Care Coordinator Initial Encounter Date'])):
        return 'Difficult to Engage'
    elif (row['Days Since Randomized'] <= 61):
        return 'Early'
    else:
        return 'Late'

def genEngagement(row):
    if ((row['Status'] == 'Early' and row['Days Since Last CC Visit'] <= 14) or
        (row['Status'] == 'Late' and row['Days Since Last CC Visit'] <= 31)):
        return 'Engaged'
    elif row['Status'] in ['Difficult to Engage', 'Recently Enrolled']:
        return row['Status']
    else:
        return 'Not Engaged'

chart7['Status'] = chart7.apply(genStatus, axis=1)
chart7['Engagement Status'] = chart7.apply(genEngagement, axis=1)

with st.container():
    st.header("Patient Engagement Status By Clinic") 
    cole1, cole2 = st.columns(2)
    
    with cole1:
        chart7 = chart7[chart7['Currently Active']==1]
        chart7 = chart7.groupby(['Current Primary Clinic','Engagement Status'])['Patient ID'].nunique().unstack().fillna(0)
        chart7 = chart7.rename(columns={'Difficult to Engage':'Unable to Make Initial Contact', 'Engaged':'Engaged','Not Engaged':'Not Engaged', 'Recently Enrolled':'Recently Enrolled'})
        st.bar_chart(chart7, height=400)
        
    with cole2:
        # chart 8: days since last CC Visit
        chart8 = contact_note[['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']]
        chart8['Contact Date'] = pd.to_datetime(chart8['Contact Date'])
        chart8 = chart8[chart8['Contact Type'].isin(['I/A','F/U']) & chart8['Patient Clinic Names'].isin(clinics)]
        
        chart8p = patient.loc[patient['Currently Active']==1, ['Patient ID']]
        chart8 = pd.merge(chart8, chart8p, how='inner', on='Patient ID')
        chart8 = chart8.sort_values(by=['Contact Date'], ascending=False).drop_duplicates(subset=['Patient ID','Contact Date'], keep='first')
        
        chart8r = episode[['Patient ID', 'Randomization Date']].copy()
        chart8r['Randomization Date'] = pd.to_datetime(chart8r['Randomization Date'])
        
        chart8 = pd.merge(chart8, chart8r, on='Patient ID', how='inner')
        chart8['Today'] = today_date
        chart8['delta'] = (chart8['Today']-chart8['Contact Date']).dt.days
        merger = chart8.groupby('Patient ID')['Contact Date'].max().reset_index()
        x = pd.merge(chart8,merger,on='Patient ID',how='inner').drop_duplicates(subset=['Patient ID','Contact Date_y'], keep='first')
        x['delta'] = (x['Today']-x['Contact Date_y']).dt.days
        x['Tx Stage'] = np.where((x['Today'] - x['Randomization Date']).dt.days <= 61, 'Months 1 & 2', 'Months 3+')
        chart8 = x.groupby(['Patient Clinic Names','Tx Stage'])['delta'].median().reset_index()
        
        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart8, x='delta', y='Patient Clinic Names', orient='h', hue='Tx Stage')
        plt.title("# Days (median) Since Last CC Visit")
        plt.xlabel("# Days (median)")
        plt.ylabel("Clinic")
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)

with st.container():
    st.header("MOUD Status By Clinic") 
    chart9c = contact_note[['Patient ID','Medication for OUD - Name','Medication for OUD - Days Prescribed','Medication for OUD - Number of Refills','Medication for OUD - Frequency','Contact Date','Contact Type']]
    chart9c = chart9c[chart9c['Contact Type'].isin(['X/X','P/E'])]
    chart9c['Days of Med'] = (chart9c['Medication for OUD - Days Prescribed'] * (chart9c['Medication for OUD - Number of Refills']+1))
    chart9c['Today'] = today_date
    chart9c['Contact Date'] = pd.to_datetime(chart9c['Contact Date'])
    chart9c['Run out Date'] = chart9c['Contact Date'] + pd.to_timedelta(chart9c['Days of Med'], unit='D')
    
    conditions = [
        (chart9c['Medication for OUD - Name']=='Methadone'),
        (chart9c['Run out Date'] < chart9c['Today']),
        (chart9c['Run out Date'] >= chart9c['Today'])
    ]

    values = ['Methadone', 'Flagged for CC Review', 'Active']

    chart9c['Med Status'] = np.select(conditions, values, default='Unknown')
    
    chart9c = chart9c.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')
    p = patient.loc[(patient['Currently Active']==1) & (patient['Current Primary Clinic'].isin(clinics)), ['Patient ID','Current Primary Clinic']]
    merger1 = pd.merge(p, chart9c, on='Patient ID', how='inner')
    
    y = episode.loc[(episode['Last MOUD Prescriber Encounter Date'].isna()) & (episode['Organization Name']==site_name), ['Patient ID','Last MOUD Prescriber Encounter Date', 'Organization Name']]
    y['Med Status'] = 'Not Started'
    merger = pd.merge(p, y, on='Patient ID', how='inner')
    concat = pd.concat([merger1, merger])
    final = concat.groupby(['Current Primary Clinic','Med Status'])['Patient ID'].count().reset_index()
    
    bars = alt.Chart(final).mark_bar().encode(
        x=alt.X('Patient ID'),
        y=alt.Y('Current Primary Clinic'),
        color=alt.Color('Med Status')
        )
    st.altair_chart(alt.layer(bars, data=final).resolve_scale(color='independent').properties(height=525), use_container_width=True)

with st.container():
    st.header("Psychotropic Medication Status") 
    col1, col2 = st.columns(2)
    chart11c = contact_note[['Patient ID','Contact Date','Contact Type', 'Current Medication 1', 'Current Medication 2']]
    chart11c = chart11c[chart11c['Contact Type'].isin(['I/A','F/U'])]
    chart11c['Contact Date'] = chart11c['Contact Date'].astype('datetime64')
    x = chart11c
    x = x.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')

    p = patient[['Patient ID','Currently Active','Current Primary Clinic']]
    p = p[(p['Currently Active']==1) & (p['Current Primary Clinic'].isin(clinics))]

    merger1=pd.merge(p,x,on='Patient ID',how='inner')

    y = episode[['Patient ID','Historic Diagnosis - Depression','Historic Diagnosis - PTSD', 'Organization Name']]
    y = y[y['Organization Name']==site_name]

    merger = pd.merge(merger1,y,on='Patient ID', how='inner')

    def genMaint(df):
        df = df.fillna(0)
        df['Med Status'] = df['Current Medication 1'].apply(lambda x: 'Maintained' if x != 0 else 'Not Maintained')
        df['MDD'] = df['Historic Diagnosis - Depression'].apply(lambda x: 'MDD' if x==1 else np.nan)
        df['PTSD'] = df['Historic Diagnosis - PTSD'].apply(lambda x: 'PTSD' if x==1 else np.nan)
        return df


    working_df = genMaint(merger)

    final_ptsd = working_df.groupby(['Current Primary Clinic','Med Status','PTSD'])['Patient ID'].count().reset_index()

    final_mdd = working_df.groupby(['Current Primary Clinic','Med Status','MDD'])['Patient ID'].count().reset_index()

    bars_mdd = alt.Chart(final_mdd).mark_bar().encode(
    x=alt.X('Patient ID', stack="normalize", axis=alt.Axis(format='%')),
    y='Current Primary Clinic',
    color='Med Status'
    )

    bars_ptsd = alt.Chart(final_ptsd).mark_bar().encode(
    x=alt.X('Patient ID', stack="normalize", axis=alt.Axis(format='%')),
    y='Current Primary Clinic',
    color='Med Status'
    )
    
    with col1:
        st.subheader("Proportion of MDD Patients by MH Rx Status")
        st.altair_chart((bars_mdd).properties(height=300),use_container_width=True)
    with col2:
        st.subheader("Proportion of PTSD Patients by MH Rx Status")
        st.altair_chart((bars_ptsd).properties(height=300),use_container_width=True)

with st.container():
    st.header("Psychotherapy Status By Clinic") 
    col3, col4 = st.columns(2)
    with col3:
        chart10c = contact_note[['Patient ID','Contact Date','Contact Type','Provider Name']]
        chart10c = chart10c[chart10c['Contact Type'].isin(['B/P'])]
        
        chart10c['Today'] = pd.to_datetime('today').date()
        chart10c['Today'] = pd.to_datetime(chart10c['Today'])
        chart10c['Contact Date'] = pd.to_datetime(chart10c['Contact Date'])
        chart10c['Days Since BHP'] = (chart10c['Today'] - chart10c['Contact Date']).dt.days
        chart10c['BHP Status'] = np.where(chart10c['Provider Name'].str.contains('healthy', na=False, case=False), 'Healthy Families', 'Any')
        chart10c['BHP Status'] = np.where(chart10c['Days Since BHP'] <= 31, 'Last 30 Days', chart10c['BHP Status'])
        x = chart10c.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')

        p = patient[['Patient ID','Currently Active','Current Primary Clinic']]
        p = p[(p['Currently Active']==1) & (p['Current Primary Clinic'].isin(clinics))].reset_index()

        merger1 = pd.merge(p,x,on='Patient ID', how='inner')
        y = episode[['Patient ID','Last BH Provider Encounter Date', 'Organization Name']]
        y['Last BH Provider Encounter Date'] = y['Last BH Provider Encounter Date'].fillna(value=0)
        y = y[(y['Last BH Provider Encounter Date']==0) & (y['Organization Name']==site_name)]
        y['BHP Status'] = 'None'
        merger = pd.merge(p,y,on='Patient ID', how='inner')

        a = agencies[['Patient ID','Purpose','Agency Name and Contact Info']]
        merge2 = pd.merge(p, a,on='Patient ID', how='left')
        merge2['Other MH Tx'] = merge2['Purpose'].apply(lambda x: "Other MH Tx" if x == 1 or x ==2 else 'None')
        merge2 = merge2[merge2['Other MH Tx']=='Other MH Tx'].drop_duplicates(subset='Patient ID')

        concat = pd.concat((merger1,merger))
        test = pd.merge(concat, merge2, on='Patient ID', how='left').reset_index()
        test['BHP Status'] = np.where((test['BHP Status']=='None') & (test['Other MH Tx']=='Other MH Tx'), 'Other MH Tx', test['BHP Status'])
        concat = test[['Patient ID','Current Primary Clinic_x','BHP Status']].reset_index()
        final = concat.drop_duplicates(subset='Patient ID').groupby(['Current Primary Clinic_x','BHP Status'])['Patient ID'].count().reset_index().rename(columns={'Current Primary Clinic_x':'Current Primary Clinic'})
        bars = alt.Chart(final).mark_bar().encode(
            x=alt.Y('Patient ID'),
            y=alt.X('Current Primary Clinic'),
            color=alt.Color('BHP Status')
        )

        st.altair_chart((bars).properties(height=525), use_container_width=True)

    with col4:
        chart6 = contact_note.loc[contact_note['Contact Type']=='B/P', ['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']].copy()
        chart6['Contact Date'] = pd.to_datetime(chart6['Contact Date'])
        chart6 = chart6[chart6['Patient Clinic Names'].isin(clinics)]
        chart6 = pd.merge(chart6, patient.loc[patient['Currently Active']==1, ['Patient ID']], how='inner', on='Patient ID')
        chart6 = pd.merge(chart6, episode[['Patient ID', 'Randomization Date']], on='Patient ID', how='inner')
        chart6 = chart6.sort_values(by=['Patient ID', 'Contact Date'], ascending=True).drop_duplicates(subset='Patient ID', keep='first')
        chart6['Randomization Date'] = pd.to_datetime(chart6['Randomization Date'])
        chart6['delta'] = (chart6['Contact Date'] - chart6['Randomization Date']).dt.days
        chart6 = pd.DataFrame(chart6.groupby('Patient Clinic Names')['delta'].median()).reset_index()
        fig, ax = plt.subplots() #solved by add this line 
        ax = sns.barplot(data=chart6, x='delta', y='Patient Clinic Names', orient='h')
        plt.title("Median Days to First BHP Visit by Clinic")
        plt.xlabel("# Days (median)")
        plt.ylabel("Clinic")
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
