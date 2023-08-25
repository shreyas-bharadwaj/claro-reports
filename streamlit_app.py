import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
import altair as alt

st.set_page_config(layout='wide')
patient = pd.read_csv("./data/patient.csv")
episode = pd.read_csv("./data/episode.csv")
contact_note = pd.read_csv("./data/contact_note.csv")
agencies = pd.read_csv('./data/agencies.csv')

# Define columns; you can adjust the ratio to change the width
col1, col2, col3 = st.columns(3)

# Place the selector in the first column
with col1:
    st.image("./claro-logo.png", width=200)
    option = st.selectbox(
        'Select an Organization',
        episode[~episode['Organization Name'].isin(['Hidalgo Medical Services', 'University of New Mexico (Non-current)'])]['Organization Name'].unique()
    )
with col2:
    st.image
    

site_name = option
clinics = episode[episode['Organization Name'] == site_name]['Clinic Name'].unique()
st.header("CLARO Site Report | Version 2.1") 
st.subheader("data current as of {}".format(pd.to_datetime('today').date()))

with st.container():
    col1, col2 = st.columns(2)
#first chart patient count by CC
    try:
        with col1:
            chart1 = patient[['Patient ID', 'Current Care Coordinator', 'Currently Active']]
            chart1['Current Care Coordinator'] = chart1['Current Care Coordinator'].apply(lambda x: x.split()[0])
            chart1 = chart1[chart1['Currently Active']==1]
            merge = episode[['Patient ID', 'Organization Name']]
            merge = merge[merge['Organization Name']==site_name]
            merge = pd.merge(chart1, merge, how='inner', on='Patient ID')
            chart1 = pd.DataFrame(merge.groupby('Current Care Coordinator')['Patient ID'].count()).reset_index()
            chart1 = chart1.sort_values(by='Patient ID', ascending=False)
            # ticks = np.arange(0, chart1['Patient ID'].max()+1, 3).tolist()
            max_value = chart1['Patient ID'].max()
            if pd.notna(max_value):
                ticks = np.arange(0, max_value + 1, 3).tolist()
            else:
                ticks = []  # or some default value that makes sense in your context
            if chart1.empty:
                print("The DataFrame chart1 is empty.")  # Or handle in a way that fits your application
            else:
                if 'Patient ID' not in chart1 or 'Current Care Coordinator' not in chart1:
                    print("Missing required columns.")  # Or handle as needed
                else:
                        # Optionally, drop rows with NaN in the specified columns
                    chart1 = chart1.dropna(subset=['Patient ID', 'Current Care Coordinator'])
                    if chart1.empty:
                        print("After dropping NaN values, chart1 is empty.")  # Handle as needed
                    else:
                        ax = sns.barplot(data=chart1, x='Patient ID', y='Current Care Coordinator', orient='h')
            fig, ax = plt.subplots() #solved by add this line 
            plt.title("Caseload Size by CC")
            plt.xlabel("# Patients")
            plt.ylabel("CC")
            plt.xticks(ticks)
            ax = sns.barplot(data=chart1, x='Patient ID', y='Current Care Coordinator', orient='h')
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
    except Exception as e:  # Catching all exceptions and aliasing it as 'e'
        print(f"An error occurred at chart 1 [Patient Count by CC]: {e}")  # Printing the error
        st.write("No data for 'Patient Count by CC' and 'Days Enrolled (median) by CC', charts unable to render")

        #third chart 
    try:
        chart3 = patient[['Patient ID', 'Current Care Coordinator', 'Most Recent Randomization Date', 'Currently Active']]
        chart3 = chart3[chart3['Currently Active'] == 1]
        chart3['Most Recent Randomization Date'] = pd.to_datetime(chart3['Most Recent Randomization Date'])

        chart3e = episode[['Patient ID', 'Organization Name']]
        chart3e = chart3e[chart3e['Organization Name'] == site_name]
        chart3e['today'] = pd.to_datetime('today')

        chart3 = pd.merge(chart3, chart3e, how='inner', on='Patient ID')
        chart3['# Days Enrolled'] = (chart3['today'] - chart3['Most Recent Randomization Date']).dt.days
        chart3 = chart3[['Current Care Coordinator', '# Days Enrolled']].groupby('Current Care Coordinator').median().reset_index()
        chart3['# Days Enrolled (median)'] = chart3['# Days Enrolled']
        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart3, x='# Days Enrolled (median)', y='Current Care Coordinator', orient='h')
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
    except Exception as e:
        print(f"An error occurred at chart3: {e}")
        st.write("No data for 'Patient Count by CC' and 'Days Enrolled (median) by CC', charts unable to render")


    #second chart patient count by Clinic Name
    with col2:
        try:
            chart1 = patient[['Patient ID', 'Current Care Coordinator', 'Currently Active', 'Current Primary Clinic']]
            chart1 = chart1[(chart1['Currently Active']==1) & (chart1['Current Primary Clinic'].isin(clinics))]
            merge = chart1
            chart1 = pd.DataFrame(merge.groupby('Current Primary Clinic')['Patient ID'].count()).reset_index()
            chart1 = chart1.sort_values(by='Patient ID', ascending=False)
            fig, ax = plt.subplots() #solved by add this line 
            ax = sns.barplot(data=chart1, x='Patient ID', y='Current Primary Clinic', orient='h')
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at chart2 [Patient Count by Clinic Name]: {e}")  # Printing the error
            st.write("No data for 'Patient Count by Clinic' and 'Days Enrolled (median) by Clinic', charts unable to render")



        #fourth chart #days to first visit (median) by Clinic 
    try:
        chart4 = contact_note[['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names']]
        chart4['Contact Date'] = pd.to_datetime(chart4['Contact Date'])
        chart4 = chart4[chart4['Contact Type'] == 'I/A']
        chart4e = episode[['Patient ID', 'Organization Name', 'Randomization Date']]
        chart4e = chart4e[chart4e['Organization Name'] == site_name]
        chart4e['Randomization Date'] = pd.to_datetime(chart4e['Randomization Date'])
        chart4p = patient[['Patient ID', 'Current Primary Clinic', 'Currently Active']]
        chart4p = chart4p[chart4p['Currently Active'] == 1]
        chart4 = pd.merge(chart4, chart4e, how='inner', on='Patient ID')
        chart4 = pd.merge(chart4, chart4p, how='inner', on='Patient ID')
        chart4['# Days to Initial Visit (median)'] = (chart4['Contact Date'] - chart4['Randomization Date']).dt.days
        chart4 = chart4[['Patient Clinic Names', '# Days to Initial Visit (median)']].groupby('Patient Clinic Names').median().reset_index()
        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart4, x='# Days to Initial Visit (median)', y='Patient Clinic Names', orient='h')
        plt.title("# of Days to Initial Visit (median) by Clinic")
        plt.xlabel("# Days (median)")
        plt.ylabel("Clinic")
        plt.axvline(7)
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
    except Exception as e:
        print(f"An error occurred at chart4 [#Days to First Visit (Median) by Clinic]: {e}")
        st.write("No data for 'Patient Count by Clinic' and 'Days Enrolled (median) by Clinic', charts unable to render")



        #chart 5: number of patients who had an intiation visit within the last 30 days
        try:
            chart5 = contact_note[['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']]
            chart5['Contact Date'] = chart5['Contact Date'].astype('datetime64')
            chart5=chart5[chart5['Contact Type']=='I/A']
            chart5['30 Days Ago'] = pd.to_datetime('today').date()-pd.Timedelta(days=31)
            chart5 = chart5[chart5['Provider Organization Names']==site_name]
            chart5= chart5[chart5['Contact Date']>=chart5['30 Days Ago']]
            chart5 = chart5[['Patient Clinic Names', 'Patient ID']].groupby('Patient Clinic Names').nunique().reset_index()
            fig, ax = plt.subplots() #solved by add this line 
            ax = sns.barplot(data=chart5, x='Patient ID', y='Patient Clinic Names', orient='h')
            ticks = np.arange(0, chart5['Patient ID'].max()+1, 1).tolist()
            plt.title("# Patients w/ Initial Visit in Last 30 Days by Clinic")
            plt.xlabel("# Patients")
            plt.ylabel("Clinic")
            plt.xticks(ticks)
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at chart5 ['Number of Patients who had an initiation visit within the last 30 days]: {e}")  # Printing the error
            st.write("No data for 'Number of Patients Who Had an Initiation Visit Within the Last 30 Days', chart unable to render")



#chart 7: Patient Engagement Status By Clinic
try:
    chart7 = episode[['Patient ID', 'Care Coordinator Initial Encounter Date', '# Care Coordinator Encounter', 'Organization Name', 'Last Care Coordinator Encounter Date']]
    chart7 = chart7[chart7['Organization Name'] == site_name]
    chart7['Today'] = pd.to_datetime('today')
    chart7['Last Care Coordinator Encounter Date'] = pd.to_datetime(chart7['Last Care Coordinator Encounter Date'])
    chart7p = patient[['Patient ID', 'Most Recent Randomization Date', 'Currently Active', 'Current Primary Clinic']]
    chart7p['Most Recent Randomization Date'] = pd.to_datetime(chart7p['Most Recent Randomization Date'])

    chart7 = pd.merge(chart7, chart7p, on='Patient ID', how='inner')
    chart7['Days Since Last CC Visit'] = (chart7['Today'] - chart7['Last Care Coordinator Encounter Date']).dt.days
    chart7['Days Since Randomized'] = (chart7['Today'] - chart7['Most Recent Randomization Date']).dt.days
except Exception as e:
    print(f"An error occurred at chart7 [Patient Engagement Status By Clinic]: {e}")
    st.write("No data for 'Patient Engagement Status By Clinic', chart unable to render")

def genStatus(df):
    df['Care Coordinator Initial Encounter Date'] = df['Care Coordinator Initial Encounter Date'].fillna(value=0)
    conditions = [
        (df['Days Since Randomized'] <= 30) & (df['Care Coordinator Initial Encounter Date'] == 0),
        (df['Days Since Randomized'] >= 31) & (df['Care Coordinator Initial Encounter Date'] == 0),
        (df['Days Since Randomized'] <= 61)
    ]
    choices = ['Recently Enrolled', 'Difficult to Engage', 'Early']
    df['Status'] = np.select(conditions, choices, default='Late')
    return df


def genEngagement(df):
    conditions = [
        ((df['Status'] == 'Early') & (df['Days Since Last CC Visit'] <= 14)),
        ((df['Status'] == 'Late') & (df['Days Since Last CC Visit'] <= 31)),
        (df['Status'] == 'Difficult to Engage'),
        (df['Status'] == 'Recently Enrolled')
    ]
    choices = ['Engaged', 'Engaged', 'Difficult to Engage', 'Recently Enrolled']
    df['Engagement Status'] = np.select(conditions, choices, default='Not Engaged')
    return df



with st.container():
    st.header("Patient Engagement Status By Clinic") 
    cole1, cole2 = st.columns(2)
    with cole1:
        try:
            chart7 = genEngagement(genStatus(chart7))
            chart7 = chart7[chart7['Currently Active']==1]
            chart7 = chart7[['Patient ID','Current Primary Clinic','Engagement Status']]
            chart7 = chart7.groupby(['Current Primary Clinic','Engagement Status']).nunique().reset_index()
            chart7 = chart7.pivot(index='Current Primary Clinic', columns='Engagement Status').fillna(value=0)
            chart7plot = chart7.copy()
            chart7plot.columns = chart7plot.columns.to_flat_index().str.join('_')
            chart7plot = chart7plot.rename(columns={'Patient ID_Difficult to Engage':'Unable to Make Initial Contact', 'Patient ID_Engaged':'Engaged','Patient ID_Not Engaged':'Not Engaged', 'Patient ID_Recently Enrolled':'Recently Enrolled'})
            st.bar_chart(chart7plot, height=400)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at chart7 [Patient Engagement Status by Clinic]: {e}")  # Printing the error
            st.write("No data for 'Patient Engagement Status By Clinic', chart unable to render")
    with cole2:
        try:
        #chart 8: days since last CC Visit
            chart8 = contact_note[['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']]
            chart8['Contact Date'] = pd.to_datetime(chart8['Contact Date'])
            chart8 = chart8[chart8['Contact Type'].isin(['I/A', 'F/U'])]
            chart8 = chart8[chart8['Patient Clinic Names'].isin(clinics)]
            chart8p = patient[['Patient ID', 'Currently Active']]
            chart8p = chart8p[chart8p['Currently Active'] == 1]
            chart8 = pd.merge(chart8, chart8p, how='inner', on='Patient ID')
            chart8 = chart8.sort_values(by=['Contact Date'], ascending=False)
            chart8test = chart8.drop_duplicates(subset=['Patient ID', 'Contact Date'], keep='first')
            chart8 = chart8test.copy()
            chart8r = episode[['Patient ID', 'Randomization Date']]
            chart8r['Randomization Date'] = pd.to_datetime(chart8r['Randomization Date'])
            chart8 = pd.merge(chart8, chart8r, on='Patient ID', how='inner')
            chart8['Today'] = pd.to_datetime('today')
            chart8['delta'] = (chart8['Today'] - chart8['Contact Date']).dt.days
            merger = pd.DataFrame(chart8.groupby('Patient ID')['Contact Date'].max())
            x = pd.merge(chart8,merger,on='Patient ID',how='inner')
            x = x.drop_duplicates(subset=['Patient ID','Contact Date_y'], keep='first')
            x['delta'] = (x['Today']-x['Contact Date_y']).dt.days
            def genStage(df):
                for index,row in df.iterrows():
                    if (df.loc[index,'Today']- df.loc[index,'Randomization Date']).days<=61:
                        df.loc[index,'Tx Stage']='Months 1 & 2'
                    else:
                        df.loc[index,'Tx Stage']='Months 3+'
                return df
            x = genStage(x)
            chart8 = pd.DataFrame(x.groupby(['Patient Clinic Names','Tx Stage'])['delta'].median()).reset_index()
            fig, ax = plt.subplots() #solved by add this line 
            ax = sns.barplot(data=chart8, x='delta', y='Patient Clinic Names', orient='h', hue='Tx Stage')
            plt.title("# Days (median) Since Last CC Visit")
            plt.xlabel("# Days (median)")
            plt.ylabel("Clinic")
            ax.bar_label(ax.containers[0], label_type='center')
            # ax.bar_label(ax.containers[1], label_type='center')
            st.pyplot(fig)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at chart8 [Days Since Last CC Visit]: {e}")  # Printing the error
            st.write("No data for 'Days Since Last CC Visit', chart unable to render")

with st.container():
    try:
        st.header("MOUD Status By Clinic") 
        chart9c = contact_note[['Patient ID', 'Medication for OUD - Name', 'Medication for OUD - Days Prescribed', 'Medication for OUD - Number of Refills', 'Medication for OUD - Frequency', 'Contact Date', 'Contact Type']]
        chart9c = chart9c[chart9c['Contact Type'].isin(['X/X', 'P/E'])]
        chart9c['Days of Med'] = (chart9c['Medication for OUD - Days Prescribed'] * chart9c['Medication for OUD - Number of Refills']) + chart9c['Medication for OUD - Days Prescribed']

        def genMedStatus(df):
            df['Today'] = pd.to_datetime('today')  # Using pd.to_datetime() directly
            df['Contact Date'] = pd.to_datetime(df['Contact Date'])  # Converting 'Contact Date' to datetime
            df['Days of Med'] = df['Days of Med'].fillna(value=0)
            for index, row in df.iterrows():
                df.loc[index,'Run out Date'] = df.loc[index,'Contact Date'] + timedelta(days=df.loc[index,'Days of Med'])
                if df.loc[index,'Medication for OUD - Name']=='Methadone':
                    df.loc[index,'Med Status'] = 'Methadone'
                elif df.loc[index,'Run out Date'] < df.loc[index,'Today']:
                    df.loc[index, 'Med Status'] = 'Flagged for CC Review'
                elif df.loc[index, 'Run out Date'] >= df.loc[index,'Today']:
                    df.loc[index,'Med Status'] = 'Active'
                else:
                    df.loc[index,'Med Status'] = 'Unknown'
            return df
        x = genMedStatus(chart9c)
        x = x.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')
        p = patient[['Patient ID','Currently Active','Current Primary Clinic']]
        p = p[(p['Currently Active']==1) & (p['Current Primary Clinic'].isin(clinics))]
        merger1=pd.merge(p,x,on='Patient ID',how='inner')
        y = episode[['Patient ID','Last MOUD Prescriber Encounter Date', 'Organization Name']]
        y['Last MOUD Prescriber Encounter Date'] = y['Last MOUD Prescriber Encounter Date'].fillna(value=0)
        y = y[y['Last MOUD Prescriber Encounter Date']==0]
        y = y[y['Organization Name']==site_name]
        y['Med Status'] = 'Not Started'
        merger = pd.merge(p,y,on='Patient ID', how='inner')
        concat = pd.concat((merger1,merger))
        concat = concat[['Patient ID','Current Primary Clinic','Med Status']].reset_index()
        concat = concat.drop(columns='index')
        final = concat.groupby(['Current Primary Clinic','Med Status'])['Patient ID'].count().reset_index()
        # final2 = final.pivot_table(index='Current Primary Clinic', columns='Med Status').fillna(value=0)
        # final2.columns = final2.columns.to_flat_index().str.join('_')
        # final2 = final2.rename(columns={'Patient ID_Active':'Active', 'Patient ID_Not Started':'Not Started','Patient ID_Flagged for CC Review':'Flagged for CC Review','Patient ID_Methadone':'Methadone'})
        # final2 = final2[['Active','Flagged for CC Review','Methadone','Not Started']]
        #st.bar_chart(final2, height=450)
        bars = alt.Chart(final).mark_bar().encode(
            x=alt.X('Patient ID'),
            y=alt.Y('Current Primary Clinic'),
            color=alt.Color('Med Status')
            )
        
        st.altair_chart(alt.layer(bars, data=final).resolve_scale(color='independent').properties(height=525), use_container_width=True)
    except Exception as e:  # Catching all exceptions and aliasing it as 'e'
        print(f"An error occurred at chart9c [MOUD Status by Clinic]: {e}")  # Printing the error
        st.write("No data for 'MOUD Status By Clinic', chart unable to render")

with st.container():
    try:
        st.header("Psychotropic Medication Status") 
        col1, col2 = st.columns(2)
        chart11c = contact_note[['Patient ID','Contact Date','Contact Type', 'Current Medication 1', 'Current Medication 2']]
        chart11c = chart11c[chart11c['Contact Type'].isin(['I/A','F/U'])]
        chart11c['Contact Date'] = pd.to_datetime(chart11c['Contact Date'])  # Converting 'Contact Date' to datetime
        x = chart11c
        x = x.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')

        p = patient[['Patient ID','Currently Active','Current Primary Clinic']]
        p = p[(p['Currently Active']==1) & (p['Current Primary Clinic'].isin(clinics))]

        merger1=pd.merge(p,x,on='Patient ID',how='inner')

        y = episode[['Patient ID','Historic Diagnosis - Depression','Historic Diagnosis - PTSD', 'Organization Name']]
        y = y[y['Organization Name']==site_name]

        merger = pd.merge(merger1,y,on='Patient ID', how='inner')

        def genMaint(df):
            df = df.fillna(value=0)
            for index, row in df.iterrows():
                if df.loc[index,'Current Medication 1'] != 0:
                    df.loc[index,'Med Status']='Maintained'
                else:
                    df.loc[index, 'Med Status']='Not Maintained'
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
    except Exception as e:  # Catching all exceptions and aliasing it as 'e'
        print(f"An error occurred: {e}")  # Printing the error
        st.write("No data for 'Psychotropic Medication Status', chart unable to render")
    
        
    with col1:
        try:
            st.subheader("Proportion of MDD Patients by MH Rx Status")
            st.altair_chart((bars_mdd).properties(height=300),use_container_width=True)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred: {e}")  # Printing the error
            st.write("No data for 'Psychotropic Medication Status', chart unable to render")
    with col2:
        try:
            st.subheader("Proportion of PTSD Patients by MH Rx Status")
            st.altair_chart((bars_ptsd).properties(height=300),use_container_width=True)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at chart 11c [Psychotropic Medication Status]: {e}")  # Printing the error
            st.write("No data for 'Psychotropic Medication Status', chart unable to render")


# final = merger.groupby(['Current Primary Clinic','Historic Diagnosis - Depression'])['Patient ID'].count().reset_index()
# final
with st.container():
    st.header("Psychotherapy Status By Clinic") 
    col3, col4 = st.columns(2)
    with col3:
        try:
            chart10c = contact_note[['Patient ID','Contact Date','Contact Type','Provider Name']]
            chart10c = chart10c[chart10c['Contact Type'].isin(['B/P'])]
            def genBHP(df):
                df['Today'] = pd.to_datetime('today')
                df['Contact Date'] = pd.to_datetime(df['Contact Date'])  # Converting 'Contact Date' to datetime
                df['BHP Status'] = df['Provider Name'].apply(lambda x: 'Healthy Families' if 'healthy' in str(x).lower() else x)
                for index, row in df.iterrows():
                    df.loc[index,'Days Since BHP'] = (df.loc[index,'Today'] - df.loc[index,'Contact Date']).days
                    if df.loc[index,'Days Since BHP']<=31:
                        df.loc[index,'BHP Status'] = 'Last 30 Days'
                    else:
                        df.loc[index,'BHP Status'] = 'Any'
                return df
            x = genBHP(chart10c)
            x = x.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')

            p = patient[['Patient ID','Currently Active','Current Primary Clinic']]
            p = p[(p['Currently Active']==1) & (p['Current Primary Clinic'].isin(clinics))].reset_index()

            merger1=pd.merge(p,x,on='Patient ID', how='inner')


            y = episode[['Patient ID','Last BH Provider Encounter Date', 'Organization Name']]
            y['Last BH Provider Encounter Date'] = y['Last BH Provider Encounter Date'].fillna(value=0)
            y = y[y['Last BH Provider Encounter Date']==0]
            y = y[y['Organization Name']==site_name]

            y['BHP Status'] = 'None'

            merger = pd.merge(p,y,on='Patient ID', how='inner')

            a = agencies[['Patient ID','Purpose','Agency Name and Contact Info']]
            merge2 = pd.merge(p, a,on='Patient ID', how='left')
            merge2['Other MH Tx'] = merge2['Purpose'].apply(lambda x: "Other MH Tx" if x == 1 or x ==2 else 'None')
            merge2 = merge2[merge2['Other MH Tx']=='Other MH Tx'].drop_duplicates(subset='Patient ID')


            concat = pd.concat((merger1,merger))

            test = pd.merge(concat, merge2, on='Patient ID', how='left').reset_index()

            def otherMhTx(df):
                for index, row in df.iterrows():
                    if df.loc[index, 'BHP Status']=='None' and df.loc[index, 'Other MH Tx']=='Other MH Tx':
                        df.loc[index,'BHP Status'] = 'Other MH Tx'
                return df

            concat = otherMhTx(test)

            # x = x.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')
            # z = patient[['Patient ID','Currently Active','Current Primary Clinic']]
            # z = z[(z['Currently Active']==1) & (z['Current Primary Clinic'].isin(clinics))]
            # active = pd.merge(x,z,on='Patient ID', how='inner')

            # final = pd.merge(active,y,on=['Patient ID','Med Status'], how='outer')
            concat = concat[['Patient ID','Current Primary Clinic_x','BHP Status']].reset_index()
            concat = concat.drop(columns='index')
            concat = concat.drop_duplicates(subset='Patient ID')
            final = concat.groupby(['Current Primary Clinic_x','BHP Status'])['Patient ID'].count().reset_index()
            final = final.rename(columns={'Current Primary Clinic_x':'Current Primary Clinic'})
            # final2 = final.pivot_table(index='Current Primary Clinic', columns='BHP Status').fillna(value=0)
            # final2.columns = final2.columns.to_flat_index().str.join('_')
            # final2 = final2.rename(columns={'Patient ID_Any':'Any', 'Patient ID_Last 30 Days':'Last 30 Days','Patient ID_None':'None'})
            #st.bar_chart(final2, height=425)
            bars = alt.Chart(final).mark_bar().encode(
            x=alt.Y('Patient ID'),
            y=alt.X('Current Primary Clinic'),
            color=alt.Color('BHP Status')
            )

            st.altair_chart((bars).properties(
        height=525
    ), use_container_width=True)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at chart 10c [Psychotherapy Status By Clinic]: [{e}")  # Printing the error
            st.write("No data for 'Psychotherapy Status By Clinic', chart unable to render")



    with col4:
        try:
            #chart 6 median days to first bhp visit by clinic
            chart6 = contact_note[['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']]
            chart6['Contact Date'] = pd.to_datetime(chart6['Contact Date'])  # Converting 'Contact Date' to datetime
            chart6=chart6[chart6['Contact Type']=='B/P']
            chart6 = chart6[chart6['Patient Clinic Names'].isin(clinics)]
            chart6p = patient[['Patient ID', 'Currently Active']]
            chart6p = chart6p[chart6p['Currently Active']==1]
            chart6 = pd.merge(chart6, chart6p, how='inner', on='Patient ID')
            chart6r = episode[['Patient ID', 'Randomization Date']]
            chart6 = pd.merge(chart6, chart6r, on='Patient ID', how='inner')
            chart6 = chart6.sort_values(by=['Patient ID', 'Contact Date'], ascending=True).drop_duplicates(subset='Patient ID', keep='first')
            chart6['Randomization Date'] = pd.to_datetime(chart6['Randomization Date'])  # Converting 'Contact Date' to datetime
            chart6['delta'] = (chart6['Contact Date'] - chart6['Randomization Date']).dt.days
            chart6 = pd.DataFrame(chart6.groupby('Patient Clinic Names')['delta'].median()).reset_index()
            fig, ax = plt.subplots() #solved by add this line 
            ax = sns.barplot(data=chart6, x='delta', y='Patient Clinic Names', orient='h')
            plt.title("Median Days to First BHP Visit by Clinic")
            plt.xlabel("# Days (median)")
            plt.ylabel("Clinic")
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at chart 6 [median days to first bhp visit by clinic: {e}")  # Printing the error
            st.write("No data for 'Median Days to First BHP Visit by Clinic', chart unable to render")