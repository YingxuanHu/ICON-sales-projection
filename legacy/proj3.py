import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

df = pd.read_csv('/Users/alvinhu/Desktop/ICON/Data/Invoice.csv')
df = df[df['ItemType'] == 'Inventory']
df['InvoiceTxnDate'] = pd.to_datetime(df['InvoiceTxnDate'].str[:10])
df = df[['ItemSku', 'Quantity', 'InvoiceTxnDate']]

df['Month'] = df['InvoiceTxnDate'].dt.to_period('M')
grouped_sku = df.groupby(['ItemSku', 'Month']).agg({'Quantity':'sum'}).reset_index()


intro_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prophet Forecasts</title>
    <style>
        /* Define CSS styles for the description box */
        .description-box {
            border: 2px solid #333; /* Border style */
            padding: 10px; /* Padding inside the box */
            background-color: #f0f0f0; /* Background color */
        }
    </style>
</head>
<body>
    <h1>ItemSku Sales Forecasts</h1>
    <div class="description-box">
        <p style="font-size: 20px; margin-bottom: 30px;">This page contains sales forecasts for different ItemSkus. Each graph represents the sales forecast for a specific ItemSku over time.</p>
        <p style="margin-left: 20px; font-size: 20px; color: blue;"> NOTE: </p>
        <p style="margin-left: 20px;">The blue shaded region is a visual representation of the forecast's uncertainty.</p>
        <p style="margin-left: 20px;"> A wider shaded region indicates higher uncertainty, while a narrower shaded region indicates lower uncertainty.<p>
        <p style="margin-left: 20px;"> Each black dot represents a historical data<p>
        <p style="margin-left: 20px;"> Blue line represents the projection<p>
    </div>
"""
combined_html = intro_html

for sku, group in grouped_sku.groupby('ItemSku'):
    if len(group) >= 6:
        prophet_df = group.rename(columns={'Month': 'ds', 'Quantity': 'y'})
        prophet_df['ds'] = prophet_df['ds'].dt.to_timestamp()

        # Set the floor to 0 and determine a cap value for each ItemSku
        prophet_df['floor'] = 0
        cap_value = 2 * group['Quantity'].max()
        prophet_df['cap'] = cap_value

        model = Prophet(growth='logistic')
        model.fit(prophet_df)

        future_months = model.make_future_dataframe(periods=24, freq='M')
        future_months['floor'] = 0
        future_months['cap'] = cap_value

        forecast = model.predict(future_months)

        # Replace negative values with 0 in the forecasted data
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))

        # Round up every prediction value to the nearest whole number
        forecast['yhat'] = forecast['yhat'].apply(lambda x: round(x))

        # Generate Plotly figure
        fig = plot_plotly(model, forecast)

        # Add title to the plot
        title = f"Forecast for <b>{sku}</b>"
        fig.update_layout(title=title)

        # Update y-axis and x-axis titles
        fig.update_yaxes(title_text="Quantity")
        fig.update_xaxes(title_text="Date")

        # Convert to HTML string and append
        combined_html += fig.to_html(full_html=False, include_plotlyjs='cdn')

    else:
        error_message = f"The ItemSku: <b>{sku}</b> has doesn't have enough data infomraton to generate a projection"
        
        # Add error message to the HTML
        combined_html += f"<p>{error_message}</p>"

combined_html += """
</body>
</html>
"""

# Save the combined HTML to a single file
with open('/Users/alvinhu/Desktop/ICON/Data/projection.html', 'w') as file:
    file.write(combined_html)

#'/Users/alvinhu/Desktop/ICON/Data/combined.html'