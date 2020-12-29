# -*- coding: utf-8 -*-

import json
from itertools import product

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
from dash.dependencies import Input, Output

import crit2vec


def load_data():
    data = {}
    num_show_words = [100, 500, 1000, 5000]
    n_neighbours = [4, 8, 16, 32, 64]
    dims = [2, 3]
    words = {}
    for model in ['full-model']:#, 'mighty-nein', 'vox-machina']:
        words[model] = {}
        data[model] = {}
        with open(f"data/crit2vec_{model}.words.json") as f:
            words[model]['all'] = json.load(f)['words_by_freq']

        for num_words in num_show_words:
            with open(f"data/{model}/umap/words-{num_words}.json") as f:
                words[model][num_words] = json.load(f)

        for neighbours, dim, num_words in product(n_neighbours, dims, num_show_words):
            data[model][dim, neighbours, num_words] = np.load(f"data/{model}/umap/neighbours-{neighbours}_dim-{dim}_words-{num_words}.npy")
    
    return n_neighbours, dims, data, words



NEIGHBOUR_OPTIONS, DIM_OPTIONS, DATA, WORDS = load_data()
MODEL = "full-model"
NLPs = {model: spacy.load(f"data/models/crit2vec_{model}") for model in ['full-model']}#, 'vox-machina', 'mighty-nein']}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA, "https://use.typekit.net/ezg3tjx.css"])
server = app.server
app.title = "Crit2Vec"
PLOT_COLOR1 = "#3e4450"
PLOT_COLOR2 = "#f9c28c"
PLOT_COLOR3 = "#ddd4c6"
DEFAULT_WORD = "pike"
TEMPLATE = {
    'layout': go.Layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=go.layout.Margin(l=20, r=20, t=20, b=20),
        hovermode='closest',
        font = {
            'family': 'futura-pt',
            'color': 'rgb(64, 32, 17)'
        }
    )
}


info = dbc.Card(
    [html.H1([html.Span("Crit"), html.Span("Y", className="two"), html.Span("Vec")]),
    dcc.Markdown("""\
        Explore [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) embeddings of
        the [Critical Role transcripts](https://crtranscript.tumblr.com/transcripts).
        """)], body=True, className="title-card"
)


controls = dbc.Card(
    dbc.CardBody(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Neighbours used in KNN graph:"),
                    dcc.Dropdown(
                        id="n-neighbours",
                        options=[
                            {"label": n, "value": n} for n in NEIGHBOUR_OPTIONS 
                        ],
                        value=16,
                    ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Embedding dimension:"),
                    dcc.Dropdown(
                        id="dim",
                        options=[
                            {"label": d, "value": d} for d in DIM_OPTIONS 
                        ],
                        value=2,
                    ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Number of words:"),
                    dcc.Dropdown(
                        id="num-words",
                        options=[
                            {"label": d, "value": d} for d in [num_words for num_words in WORDS[MODEL] if num_words != 'all']
                        ],
                        value=500,
                    ),
                ]
            ),
        ],
    )
)


word_calculator = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Positive words:"),
                dcc.Dropdown(
                    id="positive-words",
                    options=[
                        {"label": word, "value": word} for word in WORDS[MODEL]['all'] 
                    ],
                    value=['vax', 'sister'],
                    multi=True,
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Negative words:"),
                dcc.Dropdown(
                    id="negative-words",
                    options=[
                        {"label": word, "value": word} for word in WORDS[MODEL]['all'] 
                    ],
                    value=['brother'],
                    multi=True,
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("No. result words to show"),
                dcc.Slider(
                    id="num-show-words",
                    dots=True,
                    min=1,
                    max=10,
                    value=5
                )
            ]
        )
    ],
    body=True
)


app.layout = dbc.Container(
    [
        dbc.CardDeck(
            [
                dbc.Col([info, controls], md=3),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Words grouped by similarity"),
                            html.H5("UMAP of word vectors"),
                            dcc.Graph(id="umap-graph")
                        ]),
                    ), md=6
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4(id="selected-word"),
                            dcc.Graph(id="similarity-graph")
                            ])
                        ), md=3),
            ],
            className = "umap-deck"
        ),
        
        dbc.CardDeck(
            [
                dbc.Col(word_calculator, md=3),
                dbc.Col(
                    dbc.Card([
                        html.H4(id="arithmetic-expression"),
                        dcc.Graph(id="arithmetic-graph")
                    ], body=True), md=6),
                dbc.Col(
                    dbc.Card([
                        html.H4("What is this?"),
                        dcc.Markdown(
                            """This is a dashboard to visualise *word embeddings* of the Critical Role transcript.
                            Word embeddings is a technique to represent words as a list of numbers (in this case 100), so we perform mathematical operations with them. To the left, you add and subtract words.
                            In the scatterplot above, we use a technique called [UMAP](https://umap-learn.readthedocs.io/en/latest/) to visualise how words are distributed in relation to one another.
                            Words that are close together are similar, but we cannot say anything about words that are far apart, they may still be similar.
                            """
                        )
                    ], body=True), md=3)
            ],
            className = "arithmetic-deck"
        )
    ],
    fluid=True,
)



@app.callback(
    Output("umap-graph", "figure"),
    [
        Input("n-neighbours", "value"),
        Input("dim", "value"),
        Input("num-words", "value"),
        Input("umap-graph", 'clickData')
    ],
)
def make_graph(n_neighbours, dim, num_words, clickData):
    df = pd.DataFrame(DATA[MODEL][dim, int(n_neighbours), num_words])
    print(DATA[MODEL][dim, int(n_neighbours), num_words].shape, dim)
    df["Words"] = [word.capitalize() for word in WORDS[MODEL][num_words]]
    
    #px.defaults.plot_bgcolor = "rgba(0, 0, 0, 0)"
    #px.defaults.paper_bgcolor = "rgba(0, 0, 0, 0)"
    if dim == 2:
        figure = px.scatter(df, x=0, y=1, hover_name="Words", custom_data=["Words"], template=TEMPLATE)
    else:
        figure = px.scatter_3d(df, x=0, y=1, z=2, hover_name="Words", custom_data=["Words"], template=TEMPLATE)


    if clickData is not None:
        word = clickData["points"][0]["customdata"][0].lower()
    else:
        word = DEFAULT_WORD.lower()
    
    try:
        index = WORDS[MODEL][num_words].index(word)
    except ValueError:
        pass
    else:
        color = [PLOT_COLOR1]*len(df)
        color[index] = PLOT_COLOR2
        size = [7.5]*len(df)
        size[index] = 15
        with figure.batch_update():
            figure.data[0].marker.color = color
            figure.data[0].marker.size = size
    #figure.update_layout()
    
    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor=PLOT_COLOR3, title="", showticklabels=True,)
    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor=PLOT_COLOR3, title="", showticklabels=True,)
    return figure


@app.callback(
    Output('similarity-graph', 'figure'),
    Input('umap-graph', 'clickData'),
)
def find_similar_words(clickData):
    if clickData is None:
        word = DEFAULT_WORD.lower()
    else:
        word = clickData["points"][0]["customdata"][0].lower()

    similar_words = crit2vec.find_nearest(NLPs[MODEL], positive=[word], n_vectors=6)
    data = pd.DataFrame({
        'Word': [f"{word.capitalize()}  " for word in similar_words][::-1],
        'Similarity': [similarity for similarity in similar_words.values()][::-1]
    })

    colors = [PLOT_COLOR1]*len(similar_words)
    figure = px.bar(data, text="Similarity", x="Similarity", y="Word", orientation="h", template=TEMPLATE, color_discrete_sequence=colors)
    
    figure.update_traces(texttemplate='%{text:.2f}',)
    figure.update_xaxes(zeroline=False, showline=False, linewidth=2, linecolor='black', showticklabels=False)
    figure.update_yaxes(zeroline=False, showline=False, linewidth=2, linecolor='black', title="")
    figure.layout.margin = go.layout.Margin(l=100, t=20, b=20, r=0)
    return figure

@app.callback(
    Output('selected-word', 'children'),
    Input('umap-graph', 'clickData'),
)
def find_arithmetic_expression_title(clickData):
    if clickData is None:
        word = DEFAULT_WORD
    else:
        word = clickData["points"][0]["customdata"][0]
    return f'Closest words to "{word.capitalize()}"'

@app.callback(
    Output('arithmetic-graph', 'figure'),
    Input('positive-words', 'value'),
    Input('negative-words', 'value'),
    Input('num-show-words', 'value')
)
def make_arithmetic_expression_bar_chart(positive, negative, num_show_words):
    similar_words = crit2vec.find_nearest(NLPs[MODEL], positive=positive, negative=negative, n_vectors=num_show_words)
    all_results = [word for word in similar_words]
    data = pd.DataFrame({
        'Word': [f"{word.capitalize()}  " for word in all_results[::-1]],
        'Similarity': [similarity for similarity in similar_words.values()][::-1]
    })
    
    colors = [PLOT_COLOR1]*len(all_results)
    figure= px.bar(data,text="Similarity",  x="Similarity", y="Word", orientation="h", template=TEMPLATE, height=300, color_discrete_sequence=colors)

    figure.update_traces(texttemplate='%{text:.2f}')
    figure.update_xaxes(zeroline=False, showline=False, linewidth=2, linecolor='black', showticklabels=False)
    figure.update_yaxes(zeroline=False, showline=False, linewidth=2, linecolor='black', title="")
    figure.layout.margin = go.layout.Margin(l=100, t=20, b=20, r=20)
    return figure

@app.callback(
    Output('arithmetic-expression', 'children'),
    Input('positive-words', 'value'),
    Input('negative-words', 'value'),
)
def find_arithmetic_expression_title(positive, negative):
    similar_words = crit2vec.find_nearest(NLPs[MODEL], positive=positive, negative=negative, n_vectors=1)
    all_results = [word for word in similar_words]
    top_result = all_results[0]
    
    title_string = " + ".join(positive) 
    if len(negative) > 0:
        title_string += " - " + " - ".join(negative)
    title_string += " = " + top_result
    return title_string


if __name__ == "__main__":
    app.run_server()
