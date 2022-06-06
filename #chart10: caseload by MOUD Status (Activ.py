#chart10: caseload by MOUD Status (Active, Methadone, Flagged for CC Review, Not Started)
# SELECT "Contact Note Contact Note"."Patient_ID" AS "Patient ID",
#        "Contact Note Contact Note"."Medication_for_OUD_Name" AS "Medication For OUD Name",
#        "Contact Note Contact Note"."Medication_for_OUD_Days_Prescribed" AS "Medication For OUD Days Prescribed",
#        "Contact Note Contact Note"."Medication_for_OUD_Number_of_Refills" AS "Medication For OUD Number Of Refills",
#        "Contact Note Contact Note"."Medication_for_OUD_Frequency" AS "Medication For OUD Frequency",
#        MAX("Contact Note Contact Note"."Contact_Date") AS "Maximum Contact Date"
# FROM "GoogleSheets"."Contact_Note_Contact_Note" AS "Contact Note Contact Note"
# WHERE (("Contact Note Contact Note"."Contact_Type" = 'X/X'))
#   OR ("Contact Note Contact Note"."Contact_Type" = 'P/E')

# "(\"Medication For OUD Days Prescribed\"*\"Medication For OUD Number Of Refills\")+\"Medication For OUD Days Prescribed\"

#dateadd(\"Maximum Contact Date\", \"Days of Medication Prescribed\", 'day')
#new_name": "Run Out Date"
#datediff(\"Today\",\"Run Out Date\",'day')"
#"case when \"Medication For OUD Name\" is \"Methadone\" then \"Methadone\" when \"Custom Formula\"<0 then \"Overdue\" else \"Active\" end",

chart10c = contact_note[['Patient ID','Medication for OUD - Name','Medication for OUD - Days Prescribed','Medication for OUD - Number of Refills','Medication for OUD - Frequency','Contact Date','Contact Type']]
chart10c = chart10c[chart10c['Contact Type'].isin(['X/X','P/E'])]
chart10c['Days of Med']  = (chart10c['Medication for OUD - Days Prescribed'] * chart10c['Medication for OUD - Number of Refills'])+chart10c['Medication for OUD - Days Prescribed']
def genMedStatus(df):
    df['Today'] = pd.to_datetime('today').date()
    df['Today'] = df['Today'].astype('datetime64')
    df['Days of Med'] = df['Days of Med'].fillna(value=0)
    df['Contact Date'] = df['Contact Date'].astype('datetime64')
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
x = genMedStatus(chart10c)
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

# x = x.sort_values(by='Contact Date', ascending=False).drop_duplicates(subset=['Patient ID'], keep='first')
# z = patient[['Patient ID','Currently Active','Current Primary Clinic']]
# z = z[(z['Currently Active']==1) & (z['Current Primary Clinic'].isin(clinics))]
# active = pd.merge(x,z,on='Patient ID', how='inner')

# final = pd.merge(active,y,on=['Patient ID','Med Status'], how='outer')
concat = concat[['Patient ID','Current Primary Clinic','Med Status']].reset_index()
concat = concat.drop(columns='index')
final = concat.groupby(['Current Primary Clinic','Med Status'])['Patient ID'].count().reset_index()