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

#helper functions
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

def genStage(df):
    # Calculate the difference in days between 'Today' and 'Randomization Date'
    days_difference = (df['Today'] - df['Randomization Date']).dt.days
    
    # Create a condition based on the days' difference
    condition = days_difference <= 61
    
    # Assign values to the 'Tx Stage' column based on the condition
    df['Tx Stage'] = np.where(condition, 'Months 1 & 2', 'Months 3+')
    
    return df

def genMedStatus(df):
    today = pd.to_datetime('today')
    df['Today'] = today
    df['Contact Date'] = pd.to_datetime(df['Contact Date'])
    df['Days of Med'] = df['Days of Med'].fillna(value=0)
    df['Run out Date'] = df['Contact Date'] + pd.to_timedelta(df['Days of Med'], unit='D')

    for index in df.index:
        if df.at[index, 'Medication for OUD - Name'] == 'Methadone':
            df.at[index, 'Med Status'] = 'Methadone'
        elif df.at[index, 'Run out Date'] < today:
            df.at[index, 'Med Status'] = 'Flagged for CC Review'
        elif df.at[index, 'Run out Date'] >= today:
            df.at[index, 'Med Status'] = 'Active'
        else:
            df.at[index, 'Med Status'] = 'Unknown'

    return df

def genBHP(df):
    today = pd.to_datetime('today')  # Pre-calculate 'Today'
    df['Today'] = today
    df['Contact Date'] = pd.to_datetime(df['Contact Date'])
    df['Days Since BHP'] = (today - df['Contact Date']).dt.days
    df['BHP Status'] = df['Provider Name'].str.lower().str.contains('healthy').replace({True: 'Healthy Families', False: 'Any'})
    df.loc[df['Days Since BHP'] <= 31, 'BHP Status'] = 'Last 30 Days'
    return df

def otherMhTx(df):
    mask = (df['BHP Status'] == 'None') & (df['Other MH Tx'] == 'Other MH Tx')
    df.loc[mask, 'BHP Status'] = 'Other MH Tx'
    return df

def genMaint(df):
    df = df.fillna(value=0)
    
    # Using a vectorized approach to assign 'Med Status'
    df['Med Status'] = np.where(df['Current Medication 1'] != 0, 'Maintained', 'Not Maintained')
    
    # Using a vectorized approach for 'MDD' and 'PTSD' columns
    df['MDD'] = np.where(df['Historic Diagnosis - Depression'] == 1, 'MDD', np.nan)
    df['PTSD'] = np.where(df['Historic Diagnosis - PTSD'] == 1, 'PTSD', np.nan)
    
    return df


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
        today_date = pd.to_datetime('today')

        chart3 = patient.loc[patient['Currently Active'] == 1, ['Patient ID', 'Current Care Coordinator', 'Most Recent Randomization Date']]
        chart3['Most Recent Randomization Date'] = pd.to_datetime(chart3['Most Recent Randomization Date'])
        
        chart3e = episode.loc[episode['Organization Name'] == site_name, ['Patient ID']]
        chart3e['today'] = today_date
        
        chart3 = pd.merge(chart3, chart3e, how='inner', on='Patient ID')
        chart3['# Days Enrolled'] = (chart3['today'] - chart3['Most Recent Randomization Date']).dt.days

        chart3_median = chart3.groupby('Current Care Coordinator')['# Days Enrolled'].median().reset_index()
        chart3_median.rename(columns={'# Days Enrolled': '# Days Enrolled (median)'}, inplace=True)

        fig, ax = plt.subplots()
        sns.barplot(data=chart3_median, x='# Days Enrolled (median)', y='Current Care Coordinator', orient='h', ax=ax)
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
    except Exception as e:
        print(f"An error occurred at chart3: {e}")
        st.write("No data for 'Patient Count by CC' and 'Days Enrolled (median) by CC', charts unable to render")



    with col2:
        try:
            chart1 = patient.loc[(patient['Currently Active'] == 1) & (patient['Current Primary Clinic'].isin(clinics)),
                                ['Patient ID', 'Current Primary Clinic']]
            
            chart1_count = chart1.groupby('Current Primary Clinic')['Patient ID'].count().reset_index()
            chart1_count = chart1_count.sort_values(by='Patient ID', ascending=False)
            chart1_count.rename(columns={'Patient ID': 'Patient Count'}, inplace=True)

            fig, ax = plt.subplots()
            sns.barplot(data=chart1_count, x='Patient Count', y='Current Primary Clinic', orient='h', ax=ax)
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:
            print(f"An error occurred at chart2 [Patient Count by Clinic Name]: {e}")
            st.write("No data for 'Patient Count by Clinic' and 'Days Enrolled (median) by Clinic', charts unable to render")




    try:
        chart4 = contact_note.loc[contact_note['Contact Type'] == 'I/A', ['Patient ID', 'Contact Date', 'Patient Clinic Names']]
        chart4['Contact Date'] = pd.to_datetime(chart4['Contact Date'])
        
        chart4e = episode.loc[episode['Organization Name'] == site_name, ['Patient ID', 'Randomization Date']]
        chart4e['Randomization Date'] = pd.to_datetime(chart4e['Randomization Date'])

        chart4p = patient.loc[patient['Currently Active'] == 1, ['Patient ID', 'Current Primary Clinic']]

        chart4 = chart4.merge(chart4e, on='Patient ID').merge(chart4p, on='Patient ID')
        chart4['# Days to Initial Visit (median)'] = (chart4['Contact Date'] - chart4['Randomization Date']).dt.days
        chart4_median = chart4.groupby('Patient Clinic Names')['# Days to Initial Visit (median)'].median().reset_index()

        fig, ax = plt.subplots()
        sns.barplot(data=chart4_median, x='# Days to Initial Visit (median)', y='Patient Clinic Names', orient='h', ax=ax)
        plt.title("# of Days to Initial Visit (median) by Clinic")
        plt.xlabel("# Days (median)")
        plt.ylabel("Clinic")
        plt.axvline(7)
        ax.bar_label(ax.containers[0], label_type='center')
        st.pyplot(fig)
    except Exception as e:
        print(f"An error occurred at chart4 [#Days to First Visit (Median) by Clinic]: {e}")
        st.write("No data for 'Patient Count by Clinic' and 'Days Enrolled (median) by Clinic', charts unable to render")



        try:
            thirty_days_ago = pd.to_datetime('today').date() - pd.Timedelta(days=31)
            chart5 = contact_note.loc[
                (contact_note['Contact Type'] == 'I/A') &
                (contact_note['Provider Organization Names'] == site_name) &
                (pd.to_datetime(contact_note['Contact Date']) >= thirty_days_ago),
                ['Patient ID', 'Patient Clinic Names']
            ]

            chart5_count = chart5.groupby('Patient Clinic Names')['Patient ID'].nunique().reset_index()

            fig, ax = plt.subplots()
            sns.barplot(data=chart5_count, x='Patient ID', y='Patient Clinic Names', orient='h', ax=ax)
            ticks = np.arange(0, chart5_count['Patient ID'].max() + 1, 1).tolist()
            plt.title("# Patients w/ Initial Visit in Last 30 Days by Clinic")
            plt.xlabel("# Patients")
            plt.ylabel("Clinic")
            plt.xticks(ticks)
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:
            print(f"An error occurred at chart5 ['Number of Patients who had an initiation visit within the last 30 days]: {e}")
            st.write("No data for 'Number of Patients Who Had an Initiation Visit Within the Last 30 Days', chart unable to render")




try:
    today_date = pd.to_datetime('today')
    
    # Filter relevant columns and rows
    chart7 = episode.loc[
        episode['Organization Name'] == site_name,
        ['Patient ID', 'Care Coordinator Initial Encounter Date', 'Last Care Coordinator Encounter Date']
    ]
    
    # Convert to datetime
    chart7['Care Coordinator Initial Encounter Date'] = pd.to_datetime(chart7['Care Coordinator Initial Encounter Date'])
    chart7['Last Care Coordinator Encounter Date'] = pd.to_datetime(chart7['Last Care Coordinator Encounter Date'])
    
    # Get necessary patient details
    chart7p = patient[['Patient ID', 'Most Recent Randomization Date', 'Currently Active', 'Current Primary Clinic']]
    chart7p['Most Recent Randomization Date'] = pd.to_datetime(chart7p['Most Recent Randomization Date'])

    # Merge datasets
    chart7 = pd.merge(chart7, chart7p, on='Patient ID', how='inner')
    
    # Compute the days
    chart7['Days Since Last CC Visit'] = (today_date - chart7['Last Care Coordinator Encounter Date']).dt.days
    chart7['Days Since Randomized'] = (today_date - chart7['Most Recent Randomization Date']).dt.days

except Exception as e:
    print(f"An error occurred at chart7 [Patient Engagement Status By Clinic]: {e}")
    st.write("No data for 'Patient Engagement Status By Clinic', chart unable to render")


with st.container():
    st.header("Patient Engagement Status By Clinic")
    cole1, cole2 = st.columns(2)
    with cole1:
        try:
            # Apply the necessary transformations
            chart7 = genEngagement(genStatus(chart7))
            chart7 = chart7[chart7['Currently Active'] == 1]
            chart7_count = chart7.groupby(['Current Primary Clinic', 'Engagement Status']).size().reset_index(name='Count')
            
            # Pivot the DataFrame for plotting
            chart7_pivot = chart7_count.pivot(index='Current Primary Clinic', columns='Engagement Status', values='Count').fillna(0)
            
            # Rename the columns
            chart7_pivot.columns = ['Unable to Make Initial Contact', 'Engaged', 'Not Engaged', 'Recently Enrolled']
            
            # Plot the bar chart
            st.bar_chart(chart7_pivot, height=400)
        except Exception as e:
            print(f"An error occurred at chart7 [Patient Engagement Status by Clinic]: {e}")
            st.write("No data for 'Patient Engagement Status By Clinic', chart unable to render")

    with cole2:
        try:
            # Filter necessary columns and conditions
            chart8 = contact_note.loc[contact_note['Contact Type'].isin(['I/A', 'F/U']) & contact_note['Patient Clinic Names'].isin(clinics),
                                    ['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names', 'Provider Organization Names']]
            chart8['Contact Date'] = pd.to_datetime(chart8['Contact Date'])
            
            # Merge with patient DataFrame
            chart8 = chart8.merge(patient[patient['Currently Active'] == 1][['Patient ID', 'Currently Active']], on='Patient ID')
            
            # Drop duplicates
            chart8 = chart8.sort_values(by=['Contact Date'], ascending=False).drop_duplicates(subset=['Patient ID', 'Contact Date'])
            
            # Merge with episode DataFrame
            chart8r = episode[['Patient ID', 'Randomization Date']]
            chart8r['Randomization Date'] = pd.to_datetime(chart8r['Randomization Date'])
            chart8 = chart8.merge(chart8r, on='Patient ID')
            
            # Calculate delta
            chart8['Today'] = pd.to_datetime('today')
            chart8['delta'] = (chart8['Today'] - chart8['Contact Date']).dt.days
            
            # Merge with the max Contact Date for each Patient ID
            merger = chart8.groupby('Patient ID')['Contact Date'].max().reset_index()
            x = chart8.merge(merger, on=['Patient ID', 'Contact Date'], how='inner')
            x['delta'] = (x['Today'] - x['Contact Date']).dt.days
            
            # Apply genStage function (assuming it's already optimized)
            x = genStage(x)
            
            # Group by and create the plot
            chart8 = x.groupby(['Patient Clinic Names', 'Tx Stage'])['delta'].median().reset_index()
            fig, ax = plt.subplots()
            ax = sns.barplot(data=chart8, x='delta', y='Patient Clinic Names', orient='h', hue='Tx Stage')
            plt.title("# Days (median) Since Last CC Visit")
            plt.xlabel("# Days (median)")
            plt.ylabel("Clinic")
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:
            print(f"An error occurred at chart8 [Days Since Last CC Visit]: {e}")
            st.write("No data for 'Days Since Last CC Visit', chart unable to render")


with st.container():
    try:
        st.header("MOUD Status By Clinic")
        chart9c = contact_note[['Patient ID', 'Medication for OUD - Name', 'Medication for OUD - Days Prescribed', 'Medication for OUD - Number of Refills', 'Medication for OUD - Frequency', 'Contact Date', 'Contact Type']]
        chart9c = chart9c[chart9c['Contact Type'].isin(['X/X', 'P/E'])]
        chart9c['Days of Med'] = (chart9c['Medication for OUD - Days Prescribed'] * chart9c['Medication for OUD - Number of Refills']) + chart9c['Medication for OUD - Days Prescribed']

        x = genMedStatus(chart9c)
        x = x.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')

        p = patient.loc[(patient['Currently Active'] == 1) & patient['Current Primary Clinic'].isin(clinics), ['Patient ID', 'Currently Active', 'Current Primary Clinic']]

        merger1 = pd.merge(p, x, on='Patient ID', how='inner')

        y = episode.loc[episode['Organization Name'] == site_name, ['Patient ID', 'Last MOUD Prescriber Encounter Date', 'Organization Name']]
        y['Last MOUD Prescriber Encounter Date'] = y['Last MOUD Prescriber Encounter Date'].fillna(value=0)
        y = y[y['Last MOUD Prescriber Encounter Date'] == 0]
        y['Med Status'] = 'Not Started'

        merger = pd.merge(p, y, on='Patient ID', how='inner')
        concat = pd.concat((merger1, merger))
        concat = concat[['Patient ID', 'Current Primary Clinic', 'Med Status']].reset_index(drop=True)

        final = concat.groupby(['Current Primary Clinic', 'Med Status'])['Patient ID'].count().reset_index()

        bars = alt.Chart(final).mark_bar().encode(
            x=alt.X('Patient ID'),
            y=alt.Y('Current Primary Clinic'),
            color=alt.Color('Med Status')
        )

        st.altair_chart(alt.layer(bars, data=final).resolve_scale(color='independent').properties(height=525), use_container_width=True)
    except Exception as e:
        print(f"An error occurred at chart9c [MOUD Status by Clinic]: {e}")
        st.write("No data for 'MOUD Status By Clinic', chart unable to render")


with st.container():
    try:
        st.header("Psychotropic Medication Status")
        col1, col2 = st.columns(2)
        chart11c = contact_note.loc[contact_note['Contact Type'].isin(['I/A', 'F/U']), ['Patient ID', 'Contact Date', 'Contact Type', 'Current Medication 1', 'Current Medication 2']]
        chart11c['Contact Date'] = pd.to_datetime(chart11c['Contact Date'])  # Converting 'Contact Date' to datetime

        # Selecting most recent contacts
        x = chart11c.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')

        p = patient.loc[(patient['Currently Active'] == 1) & patient['Current Primary Clinic'].isin(clinics), ['Patient ID', 'Currently Active', 'Current Primary Clinic']]
        merger1 = pd.merge(p, x, on='Patient ID', how='inner')

        y = episode.loc[episode['Organization Name'] == site_name, ['Patient ID', 'Historic Diagnosis - Depression', 'Historic Diagnosis - PTSD', 'Organization Name']]
        merger = pd.merge(merger1, y, on='Patient ID', how='inner')

        working_df = genMaint(merger)
        final_ptsd = working_df.groupby(['Current Primary Clinic', 'Med Status', 'PTSD'])['Patient ID'].count().reset_index()
        final_mdd = working_df.groupby(['Current Primary Clinic', 'Med Status', 'MDD'])['Patient ID'].count().reset_index()

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
    except Exception as e:
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
            chart10c = contact_note.loc[contact_note['Contact Type'].isin(['B/P']), ['Patient ID', 'Contact Date', 'Contact Type', 'Provider Name']]
            x = genBHP(chart10c)
            x = x.drop_duplicates(subset=['Patient ID'], keep='last')

            p = patient.loc[(patient['Currently Active'] == 1) & (patient['Current Primary Clinic'].isin(clinics)), ['Patient ID', 'Currently Active', 'Current Primary Clinic']].reset_index()

            merger1 = pd.merge(p, x, on='Patient ID', how='inner')

            y = episode.loc[(episode['Last BH Provider Encounter Date'] == 0) & (episode['Organization Name'] == site_name), ['Patient ID', 'Organization Name']]
            y['BHP Status'] = 'None'
            merger = pd.merge(p, y, on='Patient ID', how='inner')

            merge2 = pd.merge(p, agencies.loc[agencies['Purpose'].isin([1, 2]), ['Patient ID', 'Purpose']], on='Patient ID', how='left')
            merge2['Other MH Tx'] = 'Other MH Tx'

            concat = pd.concat((merger1, merger))
            test = pd.merge(concat, merge2.drop_duplicates(subset='Patient ID'), on='Patient ID', how='left').reset_index()

            concat = otherMhTx(test)
            concat = concat.drop_duplicates(subset='Patient ID')
            final = concat.groupby(['Current Primary Clinic_x', 'BHP Status'])['Patient ID'].count().reset_index()
            final = final.rename(columns={'Current Primary Clinic_x': 'Current Primary Clinic'})

            bars = alt.Chart(final).mark_bar().encode(
                x=alt.Y('Patient ID'),
                y=alt.X('Current Primary Clinic'),
                color=alt.Color('BHP Status')
            )

            st.altair_chart((bars).properties(
                height=525
            ), use_container_width=True)
        except Exception as e:
            print(f"An error occurred at chart 10c [Psychotherapy Status By Clinic]: [{e}")
            st.write("No data for 'Psychotherapy Status By Clinic', chart unable to render")



    with col4:
        try:
            chart6 = contact_note[['Patient ID', 'Contact Type', 'Contact Date', 'Patient Clinic Names']]
            chart6 = chart6[chart6['Contact Type'] == 'B/P']
            chart6['Contact Date'] = pd.to_datetime(chart6['Contact Date'])
            chart6 = chart6[chart6['Patient Clinic Names'].isin(clinics)]

            chart6p = patient[patient['Currently Active'] == 1][['Patient ID']]
            chart6r = episode[['Patient ID', 'Randomization Date']]

            # Merging required information
            chart6 = chart6.merge(chart6p, on='Patient ID').merge(chart6r, on='Patient ID')
            chart6['Randomization Date'] = pd.to_datetime(chart6['Randomization Date'])
            
            # Calculating the delta and getting the first occurrence for each patient
            chart6['delta'] = (chart6['Contact Date'] - chart6['Randomization Date']).dt.days
            chart6 = chart6.groupby(['Patient ID', 'Patient Clinic Names'])['delta'].first().reset_index().groupby('Patient Clinic Names')['delta'].median().reset_index()

            fig, ax = plt.subplots()
            ax = sns.barplot(data=chart6, x='delta', y='Patient Clinic Names', orient='h')
            plt.title("Median Days to First BHP Visit by Clinic")
            plt.xlabel("# Days (median)")
            plt.ylabel("Clinic")
            ax.bar_label(ax.containers[0], label_type='center')
            st.pyplot(fig)
        except Exception as e:
            print(f"An error occurred at chart 6 [median days to first bhp visit by clinic]: {e}")
            st.write("No data for 'Median Days to First BHP Visit by Clinic', chart unable to render")
