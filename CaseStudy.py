import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data
#data = [99.461, 97.29519, 102.6193, 99.31623, 100.4548, 104.1828, 100.3394, 98.47524, 92.99175, 98.56411]
# Data for the boxplots
data1 = [109, 109.5, 109, 109, 111, 110.5, 110.5, 111.5, 110.5, 111, 108, 112.5, 111, 112.5, 113, 112.5, 114, 113, 111.5, 112.5]
data2 = [105, 110, 105, 106.7, 113.3, 113.3, 113.3, 110, 113.3, 111.7, 115.0, 113.3]
data3 = [104.3, 106.7, 106.7]

# Creating the boxplot
fig = go.Figure()

# Add traces for each dataset
fig.add_trace(go.Box(y=data1, name='N'))
fig.add_trace(go.Box(y=data2, name='C'))
fig.add_trace(go.Box(y=data3, name='Kend&J'))

# Update layout with white background
fig.update_layout(
    title='Boxplots of Data1, Data2, and Data3',
    xaxis_title='Data Sets',
    yaxis_title='Values',
    plot_bgcolor='white',  # Set plot background to white
    paper_bgcolor='white'  # Set outer background to white
)

# Show plot
fig.show()