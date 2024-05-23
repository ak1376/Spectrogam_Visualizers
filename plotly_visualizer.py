import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import numpy as np
from scipy.io import wavfile
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample data for the spectrogram and scatter plot
x = np.linspace(0, 10, 100)
y_spectrogram = np.random.rand(1000, 100)  # Sample spectrogram data
y_scatter = np.cos(x)  # Sample scatter plot data

# Sample RGB array corresponding to each scatter plot point
rgb_array = np.random.randint(0, 256, size=(len(x), 3))  # Generate random RGB values for demonstration

# Global variable to store the mask
mask = np.zeros_like(y_spectrogram, dtype=bool)

# Mapping of scatter plot points to spectrogram segments (time bins)
scatter_to_spectrogram_mapping = [int((i / len(y_scatter)) * y_spectrogram.shape[1]) for i in range(len(y_scatter))]

app.layout = html.Div([
    dcc.Graph(
        id='waveform-plot',
        style={'height': '300px'},
        figure={
            'data': [],
            'layout': {
                'title': 'Waveform',
                'plot_bgcolor': 'black',
                'paper_bgcolor': 'black',
                'font': {'color': 'white'},
                'xaxis': {'title': 'Time (s)', 'color': 'white'},
                'yaxis': {'title': 'Amplitude', 'color': 'white'}
            }
        }
    ),
    dcc.Graph(
        id='spectrogram',
        figure={
            'data': [go.Heatmap(z=y_spectrogram)],
            'layout': {
                'title': 'Spectrogram',
                'plot_bgcolor': 'black',
                'paper_bgcolor': 'black',
                'font': {'color': 'white'},
                'xaxis': {'title': 'X-Axis', 'color': 'white', 'rangeslider': {'visible': True}},
                'yaxis': {'title': 'Y-Axis', 'color': 'white'}
            }
        }
    ),
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=x,
                    y=y_scatter,
                    mode='markers',
                    marker=dict(
                        color=['rgb({},{},{})'.format(rgb[0], rgb[1], rgb[2]) for rgb in rgb_array],  # RGB array for scatter plot colors
                        size=10,
                        line=dict(
                            width=2,
                            color='DarkSlateGrey'
                        )
                    )
                )
            ],
            'layout': go.Layout(
                title='Scatter Plot',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font={'color': 'white'},
                xaxis={'title': 'X-Axis', 'color': 'white'},
                yaxis={'title': 'Cosine Output', 'color': 'white'}
            )
        },
        config={
            'modeBarButtonsToAdd': ['lasso2d']
        }
    ),
    html.Div([
        html.Button('Clear Selection', id='clear-button', n_clicks=0, style={'margin': '10px'}),
        html.Button('Show Highlighted Regions', id='show-highlighted-button', n_clicks=0, style={'margin': '10px'}),
        html.Button('Return Indices', id='return-indices-button', n_clicks=0, style={'margin': '10px'}),
        dcc.Upload(
            id='uploader',
            children=html.Button('Upload WAV File', style={
                'margin': '10px',
                'height': '40px',
                'lineHeight': '40px',
                'borderWidth': '1px',
                'borderStyle': 'solid',
                'borderRadius': '5px',
                'backgroundColor': '#f0f0f0'
            }),
            style={
                'display': 'inline-block'
            },
            multiple=False
        )
    ], style={'display': 'flex', 'alignItems': 'center'}),
    html.Div(id='indices-output'),
    html.Div(id='audio-container', children=html.Audio(id='audio', controls=True, src=''))
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('spectrogram', 'relayoutData')],
    [State('scatter-plot', 'figure')]
)
def update_scatter_plot_color(relayoutData, scatter_plot_figure):
    if relayoutData and 'xaxis.range' in relayoutData:
        x_start, x_end = relayoutData['xaxis.range']
        new_colors = ['white' if x_start <= x_val <= x_end else 'rgb({},{},{})'.format(rgb[0], rgb[1], rgb[2]) for x_val, rgb in zip(x, rgb_array)]
        scatter_plot_figure['data'][0]['marker']['color'] = new_colors
        return scatter_plot_figure
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output('spectrogram', 'figure'),
    [Input('scatter-plot', 'selectedData'), 
     Input('clear-button', 'n_clicks'), 
     Input('show-highlighted-button', 'n_clicks')],
    [State('spectrogram', 'figure')]
)
def update_spectrogram(selectedData, clear_btn_n_clicks, show_highlighted_btn_n_clicks, current_figure):
    global mask
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # If no buttons were pressed and no selection was made, do not update.
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'clear-button':
        mask = np.zeros_like(y_spectrogram, dtype=bool)

    elif triggered_id == 'scatter-plot' and selectedData:
        # Update mask based on lasso selection
        for point in selectedData['points']:
            index = scatter_to_spectrogram_mapping[int(point['x'])]
            mask[:, index] = True

    if triggered_id == 'show-highlighted-button':
        # Apply the mask to highlight selected regions and turn the rest black
        current_figure['data'][0]['z'] = np.where(mask, y_spectrogram, np.zeros_like(y_spectrogram))
    else:
        # Reset to original spectrogram with the mask applied for highlighting
        current_figure['data'][0]['z'] = np.where(mask, y_spectrogram, y_spectrogram * 0.1)

    return current_figure

@app.callback(
    Output('indices-output', 'children'),
    [Input('return-indices-button', 'n_clicks')]
)
def return_indices(n_clicks):
    if n_clicks:
        global mask
        indices = np.where(mask)
        np.savetxt('highlighted_indices.txt', np.unique(indices[1]), fmt='%d')  # Save indices to a text file
    return ''

@app.callback(
    Output('waveform-plot', 'figure'),
    [Input('uploader', 'contents')],
    [State('uploader', 'filename')]
)
def process_uploaded_file(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    wav_io = io.BytesIO(decoded)
    sample_rate, samples = wavfile.read(wav_io)
    
    # Generate time values for waveform
    time_values = np.arange(len(samples)) / sample_rate

    # Create the waveform plot
    figure = {
        'data': [
            go.Scatter(x=time_values, y=samples, mode='lines', line=dict(color='royalblue'))
        ],
        'layout': go.Layout(
            title='Waveform',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font={'color': 'white'},
            xaxis={'title': 'Time (s)', 'color': 'white'},
            yaxis={'title': 'Amplitude', 'color': 'white'}
        )
    }
    
    return figure

@app.callback(
    Output('audio', 'src'),
    [Input('uploader', 'contents')]
)
def update_audio_src(contents):
    if contents:
        return contents
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
