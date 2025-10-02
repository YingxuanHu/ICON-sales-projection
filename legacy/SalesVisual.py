import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from pandas.tseries.offsets import DateOffset

sales = pd.read_csv('/Users/alvinhu/Desktop/ICON/Data/Invoice.csv')
# Change the path of the file
# Invoice.csv contains the two year sales on ICON product

sales = sales[sales['ItemType'] == 'Inventory'] 
# Only show the ones that is "Inventory" type

# Don't show zero values in Quantity column

sales['InvoiceTxnDate'] = sales['InvoiceTxnDate'].str[:10] 
# Show the date in format "YYYY-MM-DD" 10 chars

sales = sales[['Item','ItemSku', 'Quantity', 'InvoiceTxnDate']] 
# Only show these four columns

sales['InvoiceTxnDate'] = pd.to_datetime(sales['InvoiceTxnDate']) 
# Convert the InvoiceTxnDate to datetime type

sales = sales.sort_values('InvoiceTxnDate') 
# Sort the SKUs by dates in ascending order

##****************************************************************************************************************
## Uncomment the following if you want to save the filtered data to local 
#sales.to_csv('sales.csv', index = False)
##****************************************************************************************************************

grouped = sales.groupby('ItemSku')
# Grouped by the Unique ItemSku



##****************************************************************************************************************
##Uncomment the following if you want to save the generated PDF into a specific directory
#output_dir = "/Users/alvinhu/Desktop/ICON/ROI/SalesGraph"
## This is the desired output directory

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# pdf_path = os.path.join(output_dir, "Sales_history_graphs.pdf")
# # Make a new directory to store the PDF containing the graphs
##****************************************************************************************************************

trends_bool = []

with PdfPages("SalesPro") as pdf:
    for name, group in grouped:
        monthly_data = group.resample('MS', on='InvoiceTxnDate')['Quantity'].sum().reset_index()
        # Count the unique month that has data for each SKU
        if monthly_data["InvoiceTxnDate"].nunique() >= 6:
            plt.figure(figsize=(10, 6))
            plt.plot(monthly_data['InvoiceTxnDate'], monthly_data['Quantity'], marker='o', linestyle='-', color='skyblue', label='Monthly Sales')

            ax = plt.gca()  # Get the current Axes instance
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            # Convert InvoiceTxnDate to a number (e.g., months since the first date in the dataset)
            x_dates = monthly_data['InvoiceTxnDate'].map(pd.Timestamp.toordinal)
            # Fit the data to a 1st degree polynomial (linear) and create a polynomial function
            z = np.polyfit(x_dates, monthly_data['Quantity'], 1)
            p = np.poly1d(z)
            # Plot the trendline over the range of dates
            plt.plot(monthly_data['InvoiceTxnDate'], p(x_dates), linestyle='--', color='red', label='Trendline')

            # Adding labels, title and legend
            plt.title(f"Monthly Sales with Trendline for ItemSku: {name}")
            plt.xlabel('Date')
            plt.ylabel('Quantity Sold')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            pdf.savefig() # Save the current figure to the PDF.
            plt.close()  # Close the figure to free memory.

            trend_direction = "Positive" if z[0] > 0 else "Negative"
            trends_bool.append({'ItemSku': name, 'Trend': trend_direction})
            # Get the trend 
        else:
            # If the available sales data is within 6 months then give the message below
            error_message = f"The ItemSku: {name} has less than 6 months of data information."
            # Generate the desired error message
            
            plt.figure(figsize=(8, 1))
            plt.text(0.5, 0.5, error_message, ha='center', va='center', wrap=True)
            plt.axis('off')  # Turn off the axis.

            pdf.savefig()
            plt.close()  # Close the figure to free memory.
           

trendbool_df = pd.DataFrame(trends_bool)

#print(trend_df)

