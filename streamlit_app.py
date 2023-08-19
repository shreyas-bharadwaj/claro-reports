import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
import altair as alt
import os
import warnings

warnings.filterwarnings('ignore')

os.environ['NUMEXPR_MAX_THREADS'] = '8'


st.set_page_config(layout='wide')
patient = pd.read_csv("./data/patient.csv")
episode = pd.read_csv("./data/episode.csv")
contact_note = pd.read_csv("./data/contact_note.csv")
agencies = pd.read_csv('./data/agencies.csv')

#selector
option = st.selectbox('Select an Organization', episode[~episode['Organization Name'].isin(['Hidalgo Medical Services', 'University of New Mexico (Non-current)'])]['Organization Name'].unique())

site_name = option
clinics = episode[episode['Organization Name']==site_name]['Clinic Name'].unique()
st.header("CLARO Site Report | Version 2.0") 
st.subheader("data current as of {}".format(pd.to_datetime('today').date()))

with st.container():
    col1, col2 = st.columns(2)
#first chart patient count by CC
    try:
        with col1:
            chart1 = patient[['Patient ID', 'Current Care Coordinator', 'Currently Active']]
            chart1['Current Care Coordinator'] = chart1['Current Care Coordinator'].apply(lambda x: x.split()[0])
            chart1 = chart1[chart1['Currently Active'] == 1]
            merge = episode[['Patient ID', 'Organization Name']]
            merge = merge[merge['Organization Name'] == site_name]
            chart1 = pd.merge(chart1, merge, how='inner', on='Patient ID')
            chart1 = chart1.groupby('Current Care Coordinator')['Patient ID'].count().reset_index().sort_values(by='Patient ID', ascending=False)
            
            max_value = chart1['Patient ID'].max()
            ticks = np.arange(0, max_value + 1, 3).tolist() if pd.notna(max_value) else []

            if not chart1.empty and 'Patient ID' in chart1 and 'Current Care Coordinator' in chart1:
                fig, ax = plt.subplots()
                plt.title("Caseload Size by CC")
                plt.xlabel("# Patients")
                plt.ylabel("CC")
                plt.xticks(ticks)
                ax = sns.barplot(data=chart1, x='Patient ID', y='Current Care Coordinator', orient='h')
                ax.bar_label(ax.containers[0], label_type='center')
                st.pyplot(fig)
            else:
                print("The DataFrame chart1 is empty or missing required columns.")  # Handle as needed
    except Exception as e:
        print(f"An error occurred: {e}")  # Printing the error
        st.write("No data for 'Patient Count by CC' and 'Days Enrolled (median) by CC', charts unable to render")


        #third chart 
    try:
        chart3 = patient[['Patient ID', 'Current Care Coordinator', 'Most Recent Randomization Date', 'Currently Active']]
        chart3 = chart3[chart3['Currently Active'] == 1]
        chart3['Most Recent Randomization Date'] = pd.to_datetime(chart3['Most Recent Randomization Date']).dt.date

        chart3e = episode.loc[episode['Organization Name'] == site_name, ['Patient ID']]
        chart3e['today'] = pd.to_datetime('today').date()

        chart3 = pd.merge(chart3, chart3e, how='inner', on='Patient ID')
        chart3['# Days Enrolled'] = (chart3['today'] - chart3['Most Recent Randomization Date']).dt.days

        chart3 = chart3.groupby('Current Care Coordinator')['# Days Enrolled'].median().reset_index()
        chart3.rename(columns={'# Days Enrolled': '# Days Enrolled (median)'}, inplace=True)

        fig, ax = plt.subplots()
        ax = sns.barplot(data=chart3, x='# Days Enrolled (median)', y='Current Care Coordinator', orient='h')
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
    except Exception as e:
        print(f"An error occurred: {e}")  # Printing the error
        st.write("No data for 'Patient Count by CC' and 'Days Enrolled (median) by CC', charts unable to render")

    #second chart patient count by Clinic Name
    with col2:
        try:
            chart1 = patient[['Patient ID', 'Current Care Coordinator', 'Currently Active', 'Current Primary Clinic']]
            chart1 = chart1[(chart1['Currently Active'] == 1) & (chart1['Current Primary Clinic'].isin(clinics))]
            chart1 = chart1.groupby('Current Primary Clinic')['Patient ID'].count().reset_index()
            chart1 = chart1.sort_values(by='Patient ID', ascending=False)

            fig, ax = plt.subplots()
            ax = sns.barplot(data=chart1, x='Patient ID', y='Current Primary Clinic', orient='h')
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:
            print(f"An error occurred: {e}")  # Printing the error
            st.write("No data for 'Patient Count by Clinic' and 'Days Enrolled (median) by Clinic', charts unable to render")




        #fourth chart #days to first visit (median) by Clinic 
        try:
            # Filter and convert dates
            chart4 = contact_note[contact_note['Contact Type'] == 'I/A']
            chart4['Contact Date'] = pd.to_datetime(chart4['Contact Date']).dt.date

            # Filter episodes by organization and convert dates
            chart4e = episode[episode['Organization Name'] == site_name]
            chart4e['Randomization Date'] = pd.to_datetime(chart4e['Randomization Date']).dt.date

            # Filter patients by activity status
            chart4p = patient[patient['Currently Active'] == 1]

            # Merge datasets
            chart4 = pd.merge(chart4, chart4e, how='inner', on='Patient ID')
            chart4 = pd.merge(chart4, chart4p, how='inner', on='Patient ID')

            # Compute the number of days to the initial visit
            chart4['# Days to Initial Visit (median)'] = (chart4['Contact Date'] - chart4['Randomization Date']).dt.days

            # Group by clinic and compute median
            chart4 = chart4.groupby('Patient Clinic Names')['# Days to Initial Visit (median)'].median().reset_index()

            # Plot the data
            fig, ax = plt.subplots()
            ax = sns.barplot(data=chart4, x='# Days to Initial Visit (median)', y='Patient Clinic Names', orient='h')
            plt.title("# of Days to Initial Visit (median) by Clinic")
            plt.xlabel("# Days (median)")
            plt.ylabel("Clinic")
            plt.axvline(7)
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:
            print(f"An error occurred at chart 4 {e}")  # Printing the error
            st.write("No data for 'Patient Count by Clinic' and 'Days Enrolled (median) by Clinic', charts unable to render")


        #chart 5: number of patients who had an intiation visit within the last 30 days
        try:
            # Define the date 30 days ago
            thirty_days_ago = pd.to_datetime('today').date() - pd.Timedelta(days=31)

            # Filter the necessary data
            chart5 = contact_note[contact_note['Contact Type'] == 'I/A']
            chart5['Contact Date'] = pd.to_datetime(chart5['Contact Date']).dt.date
            chart5 = chart5[(chart5['Provider Organization Names'] == site_name) & (chart5['Contact Date'] >= thirty_days_ago)]

            # Group by clinic and count unique patient IDs
            chart5 = chart5.groupby('Patient Clinic Names')['Patient ID'].nunique().reset_index()

            # Plot the data
            fig, ax = plt.subplots()
            ax = sns.barplot(data=chart5, x='Patient ID', y='Patient Clinic Names', orient='h')
            ticks = np.arange(0, chart5['Patient ID'].max() + 1, 1).tolist()
            plt.title("# Patients w/ Initial Visit in Last 30 Days by Clinic")
            plt.xlabel("# Patients")
            plt.ylabel("Clinic")
            plt.xticks(ticks)
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:
            print(f"An error occurred at chart 5 {e}")  # Printing the error
            st.write("No data for 'Number of Patients Who Had an Initiation Visit Within the Last 30 Days', chart unable to render")




try:
    chart7 = episode[['Patient ID', 'Care Coordinator Initial Encounter Date', '# Care Coordinator Encounter', 'Organization Name', 'Last Care Coordinator Encounter Date']]
    chart7 = chart7[chart7['Organization Name']==site_name]
    chart7['Today'] =  pd.to_datetime('today').date()
    chart7['Today'] = chart7['Today'].astype('datetime64[D]')
    chart7['Last Care Coordinator Encounter Date'] = chart7['Last Care Coordinator Encounter Date'].astype('datetime64[D]')
    chart7p = patient[['Patient ID','Most Recent Randomization Date','Currently Active', 'Current Primary Clinic']]
    chart7p['Most Recent Randomization Date'] = chart7p['Most Recent Randomization Date'].astype('datetime64[D]')

    chart7 = pd.merge(chart7, chart7p, on='Patient ID', how='inner')
    chart7['Days Since Last CC Visit'] = (chart7['Today'] - chart7['Last Care Coordinator Encounter Date']).dt.days
    chart7['Days Since Randomized'] = (chart7['Today'] - chart7['Most Recent Randomization Date']).dt.days
except Exception as e:  # Catching all exceptions and aliasing it as 'e'
    print(f"An error occurred: {e}")  # Printing the error
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

def genMedStatus(df):
    # Convert dates and fill NA values
    df['Today'] = pd.to_datetime(pd.to_datetime('today').date())
    df['Days of Med'] = df['Days of Med'].fillna(0)
    df['Contact Date'] = pd.to_datetime(df['Contact Date']).dt.date.astype('datetime64[D]')
    
    # Calculate the run-out date for medication
    df['Run out Date'] = df['Contact Date'] + pd.to_timedelta(df['Days of Med'], unit='D')
    
    # Determine medication status based on conditions
    df['Med Status'] = 'Unknown' # Default value
    df.loc[df['Medication for OUD - Name'] == 'Methadone', 'Med Status'] = 'Methadone'
    df.loc[(df['Run out Date'] < df['Today']) & (df['Medication for OUD - Name'] != 'Methadone'), 'Med Status'] = 'Flagged for CC Review'
    df.loc[(df['Run out Date'] >= df['Today']) & (df['Medication for OUD - Name'] != 'Methadone'), 'Med Status'] = 'Active'
    
    return df


def genMaint(df):
    # Fill NaN values with 0
    df = df.fillna(value=0)

    # Determine medication status using a vectorized approach
    df['Med Status'] = np.where(df['Current Medication 1'] != 0, 'Maintained', 'Not Maintained')

    # Create 'MDD' and 'PTSD' columns using vectorized operations
    df['MDD'] = np.where(df['Historic Diagnosis - Depression'] == 1, 'MDD', np.nan)
    df['PTSD'] = np.where(df['Historic Diagnosis - PTSD'] == 1, 'PTSD', np.nan)

    return df

def otherMhTx(df):
    mask = (df['BHP Status'] == 'None') & (df['Other MH Tx'] == 'Other MH Tx')
    df.loc[mask, 'BHP Status'] = 'Other MH Tx'
    return df

def genBHP(df):
    df['Today'] = pd.to_datetime('today').date()
    df['Today'] = pd.to_datetime(df['Today'])  # Convert to pandas datetime
    df['Contact Date'] = pd.to_datetime(df['Contact Date'])  # Convert to pandas datetime
    df['BHP Status'] = df['Provider Name'].apply(lambda x: 'Healthy Families' if 'healthy' in str(x).lower() else x)
    for index, row in df.iterrows():
        df.loc[index,'Days Since BHP'] = (df.loc[index,'Today'] - df.loc[index,'Contact Date']).days
        if df.loc[index,'Days Since BHP'] <= 31:
            df.loc[index,'BHP Status'] = 'Last 30 Days'
        else:
            df.loc[index,'BHP Status'] = 'Any'
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
            print(f"An error occurred: {e}")  # Printing the error
            st.write("No data for 'Patient Engagement Status By Clinic', chart unable to render")

    with cole2:
        try:
        #chart 8: days since last CC Visit
            chart8 = contact_note[['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']]
            chart8['Contact Date'] = chart8['Contact Date'].astype('datetime64[D]')
            chart8=chart8[chart8['Contact Type'].isin(['I/A','F/U'])]
            chart8 = chart8[chart8['Patient Clinic Names'].isin(clinics)]
            chart8p = patient[['Patient ID', 'Currently Active']]
            chart8p = chart8p[chart8p['Currently Active']==1]
            chart8 = pd.merge(chart8, chart8p, how='inner', on='Patient ID')
            chart8 = chart8.sort_values(by=['Contact Date'], ascending=False)
            chart8test=chart8.drop_duplicates(subset=['Patient ID','Contact Date'], keep='first')
            chart8=chart8test.copy()
            chart8r = episode[['Patient ID', 'Randomization Date']]
            chart8 = pd.merge(chart8, chart8r, on='Patient ID', how='inner')
            chart8['Today'] = pd.to_datetime('today').date()
            chart8['Randomization Date'] = chart8['Randomization Date'].astype('datetime64[D]')
            chart8['Today'] = chart8['Today'].astype('datetime64[D]')
            chart8['delta'] = (chart8['Today']-chart8['Contact Date']).dt.days
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
            print(f"An error occurred: {e}")  # Printing the error
            st.write("No data for 'Days Since Last CC Visit', chart unable to render")

with st.container():
    try:
        st.header("MOUD Status By Clinic") 
        chart9c = contact_note[['Patient ID','Medication for OUD - Name','Medication for OUD - Days Prescribed','Medication for OUD - Number of Refills','Medication for OUD - Frequency','Contact Date','Contact Type']]
        chart9c = chart9c[chart9c['Contact Type'].isin(['X/X','P/E'])]
        chart9c['Days of Med']  = (chart9c['Medication for OUD - Days Prescribed'] * chart9c['Medication for OUD - Number of Refills'])+chart9c['Medication for OUD - Days Prescribed']
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
        print(f"An error occurred at 9c {e}")  # Printing the error
        st.write("No data for 'MOUD Status By Clinic', chart unable to render")

with st.container():
    try:
        st.header("Psychotropic Medication Status") 
        col1, col2 = st.columns(2)

        chart11c = contact_note[['Patient ID', 'Contact Date', 'Contact Type', 'Current Medication 1', 'Current Medication 2']]
        chart11c = chart11c[chart11c['Contact Type'].isin(['I/A', 'F/U'])]
        chart11c['Contact Date'] = chart11c['Contact Date'].astype('datetime64[D]')
        x = chart11c.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')

        p = patient.loc[(patient['Currently Active'] == 1) & patient['Current Primary Clinic'].isin(clinics), ['Patient ID', 'Currently Active', 'Current Primary Clinic']]

        merger1 = pd.merge(p, x, on='Patient ID', how='inner')
        y = episode.loc[episode['Organization Name'] == site_name, ['Patient ID', 'Historic Diagnosis - Depression', 'Historic Diagnosis - PTSD', 'Organization Name']]
        merger = pd.merge(merger1, y, on='Patient ID', how='inner')

        working_df = genMaint(merger)  # Using the optimized genMaint function

        # Grouping and aggregation can be done in a single step
        final_ptsd = working_df.groupby(['Current Primary Clinic', 'Med Status', 'PTSD'])['Patient ID'].count().reset_index()
        final_mdd = working_df.groupby(['Current Primary Clinic', 'Med Status', 'MDD'])['Patient ID'].count().reset_index()

        # Bar chart creation remains the same
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
        print(f"An error occurred at chart 11c {e}")  # Printing the error
        st.write("No data for 'Psychotropic Medication Status', chart unable to render")

    
        
    with col1:
        try:
            st.subheader("Proportion of MDD Patients by MH Rx Status")
            st.altair_chart((bars_mdd).properties(height=300),use_container_width=True)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at mdd by rx status {e}")  # Printing the error
            st.write("No data for 'Psychotropic Medication Status', chart unable to render")
    with col2:
        try:
            st.subheader("Proportion of PTSD Patients by MH Rx Status")
            st.altair_chart((bars_ptsd).properties(height=300),use_container_width=True)
        except Exception as e:  # Catching all exceptions and aliasing it as 'e'
            print(f"An error occurred at ptsd by mx rx {e}")  # Printing the error
            st.write("No data for 'Psychotropic Medication Status', chart unable to render")


# final = merger.groupby(['Current Primary Clinic','Historic Diagnosis - Depression'])['Patient ID'].count().reset_index()
# final
with st.container():
    st.header("Psychotherapy Status By Clinic")
    col3, col4 = st.columns(2)
    with col3:
        try:
            active_patients = patient[(patient['Currently Active'] == 1) & (patient['Current Primary Clinic'].isin(clinics))].reset_index()

            chart10c = contact_note[contact_note['Contact Type'].isin(['B/P'])]
            bhp_status_df = genBHP(chart10c).drop_duplicates(subset=['Patient ID'], keep='first')

            merger1 = pd.merge(active_patients, bhp_status_df, on='Patient ID', how='inner')

            y = episode[(episode['Last BH Provider Encounter Date'].fillna(0) == 0) & (episode['Organization Name'] == site_name)]
            y['BHP Status'] = 'None'
            merger = pd.merge(active_patients, y, on='Patient ID', how='inner')

            merge2 = pd.merge(active_patients, agencies[['Patient ID', 'Purpose', 'Agency Name and Contact Info']], on='Patient ID', how='left')
            merge2['Other MH Tx'] = merge2['Purpose'].apply(lambda x: "Other MH Tx" if x == 1 or x == 2 else 'None')
            merge2 = merge2[merge2['Other MH Tx'] == 'Other MH Tx'].drop_duplicates(subset='Patient ID')

            concat = pd.concat((merger1, merger))
            concat_with_other_mh_tx = pd.merge(concat, merge2, on='Patient ID', how='left').reset_index()
            concat_with_other_mh_tx = otherMhTx(concat_with_other_mh_tx)

            final_df = concat_with_other_mh_tx[['Patient ID', 'Current Primary Clinic_x', 'BHP Status']].drop_duplicates(subset='Patient ID')
            final = final_df.groupby(['Current Primary Clinic_x', 'BHP Status'])['Patient ID'].count().reset_index()
            final = final.rename(columns={'Current Primary Clinic_x': 'Current Primary Clinic'})

            bars = alt.Chart(final).mark_bar().encode(
                x=alt.Y('Patient ID'),
                y=alt.X('Current Primary Clinic'),
                color=alt.Color('BHP Status')
            )

            st.altair_chart((bars).properties(height=525), use_container_width=True)
        except Exception as e:
            print(f"An error occurred at chart 10c {e}")
            st.write("No data for 'Psychotherapy Status By Clinic', chart unable to render")




    with col4:
        with col4:
            try:
                active_patients = patient[patient['Currently Active'] == 1]
                chart6 = contact_note[(contact_note['Contact Type'] == 'B/P') & (contact_note['Patient Clinic Names'].isin(clinics))]
                
                # Convert 'Contact Date' and 'Randomization Date' to datetime type
                chart6['Contact Date'] = pd.to_datetime(chart6['Contact Date'])
                chart6 = chart6.merge(active_patients[['Patient ID']], on='Patient ID', how='inner')
                chart6 = chart6.merge(episode[['Patient ID', 'Randomization Date']], on='Patient ID', how='inner')
                chart6['Randomization Date'] = pd.to_datetime(chart6['Randomization Date'])  # Ensure datetime type

                chart6 = chart6.sort_values(by=['Patient ID', 'Contact Date']).drop_duplicates(subset='Patient ID')
                chart6['delta'] = (chart6['Contact Date'] - chart6['Randomization Date']).dt.days
                median_days_to_visit = chart6.groupby('Patient Clinic Names')['delta'].median().reset_index()

                fig, ax = plt.subplots()
                ax = sns.barplot(data=median_days_to_visit, x='delta', y='Patient Clinic Names', orient='h')
                plt.title("Median Days to First BHP Visit by Clinic")
                plt.xlabel("# Days (median)")
                plt.ylabel("Clinic")
                ax.bar_label(ax.containers[0], label_type='center')
                st.pyplot(fig)
            except Exception as e:
                print(f"An error occurred at chart 6 {e}")
                st.write("No data for 'Median Days to First BHP Visit by Clinic', chart unable to render")


