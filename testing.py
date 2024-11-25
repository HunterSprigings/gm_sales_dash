#Import packages
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_authenticator as stauth
import bcrypt
sns.set_style("whitegrid")
sns.color_palette("Set2")
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
from datetime import datetime,timedelta
from psw import hashed_password


password = st.text_input('Password:', type='password')

if not bcrypt.checkpw(password.encode(), hashed_password.encode()):
    st.success('Access Denied.')
    st.write('Please enter the correct password.')
else:
    st.error('ACCESS GRANTED!')

    dist_path = r'bc_customer_distribution.csv'
    inv_path = r'bc_age_of_inventory.csv'
    ven_path = r'bc_vendor_purchase_order.csv'

    # Data Loading and Cleaning
    @st.cache_data
    def load_data():
        df_dist = pd.read_csv(dist_path)
        df_inv = pd.read_csv(inv_path)
        df_ven = pd.read_csv(ven_path)
        
        # Clean and process distribution data
        df_dist['DATE'] = pd.to_datetime(df_dist['DATE'])
        df_dist['CITY'] = df_dist['CITY'].str.title()
        df_dist.sort_values(by='DATE', ascending=True, inplace=True)
        
        # Clean and process inventory data
        df_inv['DATE'] = pd.to_datetime(df_inv['DATE'])
        df_inv.sort_values(by='DATE', ascending=True, inplace=True)
        

        # Clean and process vendor data
        df_ven['EXPECTED_RECEIPT_DATE'] = pd.to_datetime(df_ven['EXPECTED_RECEIPT_DATE'], errors='coerce')
        df_ven['ORDER_DATE'] = pd.to_datetime(df_ven['ORDER_DATE'], errors='coerce')
        df_ven['DATE'] = pd.to_datetime(df_ven['DATE'], errors='coerce')

        # Drop rows with invalid DATE values
        df_ven = df_ven.dropna(subset=['DATE'])

        df_ven = df_ven.dropna(subset=['DATE'])
        df_ven['MONDAY_DATE'] = df_ven['DATE'].apply(lambda x: x - timedelta(days=x.weekday()))
        df_ven.sort_values(by='DATE', ascending=True, inplace=True)
        
        return df_dist, df_inv, df_ven

    df_dist, df_inv, df_ven = load_data()

    #Temporary DF's for manipulation within inventory loop
    df_inv_temp = df_inv
    df_dist_temp = df_dist
    df_ven_temp = df_ven

    #Select the latest date & filter by that week
    latest_date = df_dist_temp['DATE'].iloc[-1]
    df2 = df_dist_temp[df_dist_temp['DATE']== latest_date]
    grouped_sales = df2.groupby(by='PRODUCT_NAME').agg({'CASE_SALES':'sum'}).sort_values(by='CASE_SALES',ascending=False).reset_index()

    #Pivot the Data
    df_pivot = df_inv_temp.pivot_table(index='DATE', columns='PRODUCT_NAME', values='CURRENT_QUANTITY', aggfunc='sum', fill_value=0)
    df_pivot = df_pivot.reindex(pd.date_range(start=df_pivot.index.min(), end=df_pivot.index.max(), freq='W-MON'))

    #Select the latest date & filter by that week
    latest_date = df_dist_temp['DATE'].iloc[-1]
    df2 = df_dist_temp[df_dist_temp['DATE']== latest_date]
    grouped_sales = df2.groupby(by='PRODUCT_NAME').agg({'CASE_SALES':'sum'}).sort_values(by='CASE_SALES',ascending=False).reset_index()


    #Create joined and pivoted df's
    grouped_sales = df_dist_temp.groupby(by=['DATE', 'PRODUCT_NAME']).agg({'CASE_SALES': 'sum'}).reset_index().sort_values(by='DATE', ascending=True)
    inv_grouped = df_inv.groupby(by=['DATE','PRODUCT_NAME']).sum().reset_index()

    #12 Units in a Case
    inv_grouped['CURRENT_QUANTITY'] = inv_grouped['CURRENT_QUANTITY'] / 12

    joined = grouped_sales.merge(inv_grouped[['DATE', 'PRODUCT_NAME', 'CURRENT_QUANTITY']], 
                                on=['DATE', 'PRODUCT_NAME'], 
                                how='right')

    joined_dropped = joined.drop_duplicates(subset=['DATE','PRODUCT_NAME'])

    joined_agg = joined_dropped.groupby(['DATE', 'PRODUCT_NAME'])[['CURRENT_QUANTITY', 'CASE_SALES']].sum().reset_index()
    joined_agg2 = joined_dropped.groupby(['DATE', 'PRODUCT_NAME'])[['CURRENT_QUANTITY', 'CASE_SALES']].sum().reset_index()
    df_pivot = joined_agg.pivot(index='DATE', columns='PRODUCT_NAME', values='CURRENT_QUANTITY')
    df_pivot2 = joined_agg2.pivot(index='DATE', columns='PRODUCT_NAME', values='CASE_SALES')
    df_pivot_mg = df_pivot['GREEN MONKE-MANGO GUAVA-350ML']
    df_pivot_tc = df_pivot['GREEN MONKE-TROPICAL CITRUS-350ML']
    df_pivot_op = df_pivot['GREEN MONKE-ORANGE PASSIONFRUIT-350ML']


    st.title('British Columbia - Green Monké Performace Report')

    # Plot time series of sales by SKU
    fig = plt.figure(figsize=(10, 45))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[0.7, 0.2, 0.2, 0.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])

    palette = {
        'GREEN MONKE-TROPICAL CITRUS-350ML': '#AEE263',
        'GREEN MONKE-MANGO GUAVA-350ML': '#E55982',
        'GREEN MONKE-ORANGE PASSIONFRUIT-350ML': '#F98615'
    }

    # Plot Sales Data (Line Plot with Data Labels)
    for column in df_pivot2.columns:
        ax1.plot(df_pivot2.index, df_pivot2[column], color=palette[column], label=column, linewidth=4)
        
        # Add data labels
        for i, value in enumerate(df_pivot2[column]):
            ax1.text(df_pivot2.index[i], value, f'{value:,.0f}', ha='center', va='bottom', fontsize=10)

    # Plot Bar charts with Data Labels
    bars = [
        (ax2, df_pivot_tc, '#AEE263'),
        (ax3, df_pivot_mg, '#E55982'),
        (ax4, df_pivot_op, '#F98615')
    ]

    for ax, data, color in bars:
        ax.bar(data.index, data.values, width=(data.index[0] - data.index[1]) / 4, color=color)
        
        # Add data labels
        for i, value in enumerate(data.values):
            ax.text(data.index[i], value, f'{value:,.0f}', ha='center', va='bottom', fontsize=10, rotation=90)

    # Set x-axis limits and ticks for all subplots
    shared_xticks = df_pivot.index
    ax1.set_xticks(shared_xticks)
    ax2.set_xticks(shared_xticks)
    ax3.set_xticks(shared_xticks)
    ax4.set_xticks(shared_xticks)

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels(shared_xticks.strftime('%Y-%m-%d'), rotation=90)

    # Set y-axis limits for consistency
    max_value = max(df_pivot_tc.max(), df_pivot_mg.max(), df_pivot_op.max())
    ax2.set_ylim(0, max_value + 100)
    ax3.set_ylim(0, max_value + 100)
    ax4.set_ylim(0, max_value + 100)

    # Grid and x-tick labels adjustments
    ax1.grid(axis='y', visible=False)
    ax2.grid(axis='y', visible=False)
    ax3.grid(axis='y', visible=False)
    ax4.grid(axis='y', visible=False)

    # Set titles and labels
    ax1.set_title(
        f'Trended Sales to Date PE {latest_date.strftime("%Y-%m-%d")}',
        fontdict={'fontsize': 18, 'fontweight': 'bold'},
        pad=70
    )
    ax1.set_ylabel('Case Sales',
                fontdict={'fontsize': 14, 'fontweight': 'bold'},
                labelpad=10)
    ax3.set_ylabel('Inventory Levels',
                fontdict={'fontsize': 14, 'fontweight': 'bold'},
                labelpad=10)

    ax1.legend(['MG','OP','TC'],loc='center', bbox_to_anchor=(0.5, 1.1), ncols=3, title='Product Name')
    ax1.set_facecolor((0.5, 0.8, 0.8, 0.05))
    ax2.set_facecolor((0.5, 0.8, 0.8, 0.05))
    ax3.set_facecolor((0.5, 0.8, 0.8, 0.05))
    ax4.set_facecolor((0.5, 0.8, 0.8, 0.05))


    plt.subplots_adjust(bottom=0.7)
    st.pyplot(fig)
