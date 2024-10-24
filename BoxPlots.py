import plotly.graph_objects as go

# Define the data for each heuristic
data = {
    "SA": [108.5, 110, 106, 108.5, 108.5, 109.5, 111.5, 110, 110.5, 112.5, 111.5, 111.5, 112.5, 110.5, 114, 114.5, 114, 115, 115, 113, 102.5, 108, 112, 115, 115],
    "BFDH": [150, 136, 152.5, 145.5, 148.5, 139, 162.5, 138, 134, 130.5, 149.5, 156.5, 130.5, 134, 133, 128, 130.5, 128, 118.5, 132.5, 115, 130, 136, 157.5, 119],
    "BFDH*": [157, 138.5, 153.5, 141, 147, 138.5, 153.5, 137.5, 129, 130.5, 152, 139.5, 131, 130.5, 147, 127, 130, 128.5, 126, 129, 150, 126, 136,	148.75,	142],
    "BFDW": [189, 168.5, 165.5, 222, 159.5, 183, 180.5, 178, 184.5, 192.5, 180.5, 197.5, 171, 192.5, 166.5, 210.5, 225, 176.5, 210.5, 218.5, 150, 156, 220, 283.75, 146],
    "BFDW*": [126.5, 130.5, 149.5, 138, 157.5, 132.5, 137.5, 135.5, 130, 144, 140.5, 136.5, 152, 139.5, 144, 136.5, 147, 145, 160, 146, 127.5, 130, 138.75, 135],
    "FFDH": [150, 136, 152.5, 145.5, 148.5, 139, 162.5, 138, 134, 130.5, 149.5, 156.5, 130.5, 134, 133, 128, 130.5, 128, 118.5, 132.5, 115, 130, 136, 157.5, 119],
    "FFDH*": [157, 138.5, 153.5, 141, 147, 138.5, 153.5, 137.5, 129, 130.5, 152, 139.5, 131, 130.5, 147, 127, 130, 128.5, 126, 129, 150, 126, 136, 148.75, 142],
    "FFDW": [189, 168.5, 165.5, 222, 159.5, 183, 180.5, 178, 184.5, 192.5, 180.5, 197.5, 171, 192.5, 166.5, 210.5, 225, 176.5, 210.5, 218.5, 150, 156, 220, 283.75, 146],
    "FFDW*": [126.5, 130.5, 149.5, 138, 157.5, 132.5, 137.5, 135.5, 130, 144, 140.5, 136.5, 152, 139.5, 144, 136.5, 147, 145, 160, 146, 127.5, 130, 137.5, 135],
    "NFDH": [155.5, 143.5, 161.5, 160, 169.5, 143.5, 178.5, 145.5, 135.5, 138.5, 149.5, 172, 145, 140, 144, 129.5, 137, 132.5, 118.5, 139.5, 120, 146, 136, 162.5, 133],
    "NFDH*": [157, 140, 171, 141, 147, 138.5, 164.5, 137.5, 129, 133.5, 157, 141, 131.5, 130.5, 148, 129.5, 132, 133.5, 127, 130.5, 150, 136, 136, 150, 142],
    "NFDW": [221, 191.5, 172.5, 222, 190, 201, 222.5, 212.5, 193, 192.5, 206, 213.5, 177.5, 192.5, 190, 246, 259, 213.5, 233.5, 223, 150, 168, 220, 237.5, 163],
    "NFDW*": [131, 149, 171.5, 149, 163, 157.5, 156.5, 154.5, 150, 158.5, 149, 155, 169.5, 154, 170, 165.5, 157, 170.5, 170, 163, 130, 138, 148.75, 157],
}

# Create box plot
fig = go.Figure()

for algorithm, values in data.items():
    fig.add_trace(go.Box(
        y=values,
        name=algorithm,
        width=0.5  # Adjust this value to control box width
    ))
# Update layout
fig.update_layout(
    title={
        'text': r"$\text{Performance of Algorithms on Benchmarks}$",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 30}
    },
    xaxis_title=dict(
        text= r"$\text{Algorithms}$",
        font=dict(size=24)
    ),
    yaxis_title=dict(
        text= r"$\text{Percentage of Optimal (%)}$",
        font=dict(size=24)
    ),
    boxmode='group',
    xaxis=dict(
        tickangle=-45,
        tickfont=dict(size=20),
        showgrid=True,
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=1.2,
        showline=True,
        linewidth=1.2,
        linecolor='black'
    ),
    yaxis=dict(
        tickfont=dict(size=16),
        showgrid=True,  # Ensures grid lines are shown
        gridcolor='lightgray',  # Set grid line color
        gridwidth=0.2,  # Set grid line thickness
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=1.2,
        showline=True,
        linewidth=1.2,
        linecolor='black'
    ),
    boxgroupgap=0.1,
    boxgap=0.1,
    plot_bgcolor='white'
)

# Show the figure
fig.show()