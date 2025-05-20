import argparse

import dash
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

# --- Load Data ---
parser = argparse.ArgumentParser(description="Interactive UMAP viewer")
parser.add_argument("-u", "--umap", required=True, help="Path to UMAP results CSV")
parser.add_argument("-a", "--annotations", required=True, help="Path to annotations CSV")
args = parser.parse_args()

umap_df = pd.read_csv(args.umap, index_col=0)
anno_df = pd.read_csv(args.annotations, index_col=0, dtype=str)
data = umap_df.join(anno_df)

# --- Dash App ---
app = dash.Dash(__name__)
app.title = "Interactive UMAP Viewer"

app.layout = html.Div(
    [
        html.H2("UMAP Interactive Viewer"),
        html.Label("Select annotation column:"),
        dcc.Dropdown(
            id="annotation-column",
            options=[{"label": col, "value": col} for col in anno_df.columns],
            placeholder="Choose annotation column",
        ),
        html.Br(),
        html.Label("Highlight selected labels (optional):"),
        dcc.Dropdown(id="selected-labels", multi=True, placeholder="Select labels to highlight"),
        html.Br(),
        dcc.Graph(id="umap-plot", style={"height": "800px"}),
        html.Div(id="hover-info", style={"whiteSpace": "pre-wrap", "marginTop": "20px"}),
        html.Hr(),
        html.H4("Export UMAP Plot to PNG"),
        html.Div(
            [
                html.Label("Output filename (e.g. `my_plot.png`):"),
                dcc.Input(
                    id="png-filename", type="text", value="umap_plot.png", style={"width": "300px"}
                ),
                html.Label("Save directory (absolute path):", style={"marginLeft": "20px"}),
                dcc.Input(
                    id="png-path", type="text", value=".", style={"width": "400px"}
                ),  # Default: current dir
            ],
            style={"marginBottom": "10px"},
        ),
        html.Div(
            [
                html.Label("Width (inches):"),
                dcc.Input(id="png-width", type="number", value=8, step=1),
                html.Label("Height (inches):", style={"marginLeft": "20px"}),
                dcc.Input(id="png-height", type="number", value=6, step=1),
                html.Label("Resolution (dpi):", style={"marginLeft": "20px"}),
                dcc.Input(id="png-res", type="number", value=300, step=50),
                html.Button(
                    "Save as PNG", id="save-png-btn", n_clicks=0, style={"marginLeft": "20px"}
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "10px"},
        ),
        html.Div(id="save-confirmation", style={"marginTop": "10px", "color": "green"}),
    ]
)


# --- Update label options when annotation column changes ---
@app.callback(Output("selected-labels", "options"), Input("annotation-column", "value"))
def update_label_options(col):
    if col is None:
        return []
    return [{"label": val, "value": val} for val in sorted(data[col].dropna().unique())]


# --- Update scatter plot ---
@app.callback(
    Output("umap-plot", "figure"),
    [Input("annotation-column", "value"), Input("selected-labels", "value")],
)
def update_figure(col, selected_vals):
    plot_data = data.copy()
    fig = go.Figure()

    # Case 1: No annotation column selected
    if col is None:
        fig.add_trace(go.Scattergl(
            x=plot_data["UMAP1"],
            y=plot_data["UMAP2"],
            mode="markers",
            marker=dict(size=2, opacity=0.4, color="gray"),
            text=plot_data.index,
            hovertemplate="%{text}<extra></extra>",
            name="Data"
        ))
        fig.update_layout(yaxis_scaleanchor="x", margin=dict(l=10, r=10, t=30, b=10))
        return fig

    # Setup color mapping
    col_values = plot_data[col].astype(str)
    unique_vals = sorted(col_values.unique())
    palette = pc.qualitative.Plotly
    label2color = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}

    # Case 2: No labels selected → plot one trace per label with dimmed style
    if not selected_vals:
        for val in unique_vals:
            sub = plot_data[col_values == val]
            fig.add_trace(go.Scattergl(
                x=sub["UMAP1"],
                y=sub["UMAP2"],
                mode="markers",
                marker=dict(size=2, opacity=0.4, color=label2color[val]),
                text=sub.index,
                customdata=sub[anno_df.columns],
                hovertemplate="<b>%{text}</b><br>" + "<br>".join(
                    [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(anno_df.columns)]
                ) + "<extra></extra>",
                name=val
            ))
        fig.update_layout(yaxis_scaleanchor="x", showlegend=True, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    # Case 3: Labels selected → highlight selected, show others as gray
    highlight_mask = plot_data[col].isin(selected_vals)
    highlights = plot_data[highlight_mask]
    others = plot_data[~highlight_mask]

    # Plot other (gray)
    fig.add_trace(go.Scattergl(
        x=others["UMAP1"],
        y=others["UMAP2"],
        mode="markers",
        marker=dict(size=2, opacity=0.2, color="lightgray"),
        text=others.index,
        customdata=others[anno_df.columns],
        hovertemplate="<b>%{text}</b><br>" + "<br>".join(
            [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(anno_df.columns)]
        ) + "<extra></extra>",
        name="Other"
    ))

    # Plot each highlighted label separately
    for val in selected_vals:
        sub = highlights[highlights[col].astype(str) == val]
        fig.add_trace(go.Scattergl(
            x=sub["UMAP1"],
            y=sub["UMAP2"],
            mode="markers",
            marker=dict(size=4, opacity=0.4, color=label2color[val]),
            text=sub.index,
            customdata=sub[anno_df.columns],
            hovertemplate="<b>%{text}</b><br>" + "<br>".join(
                [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(anno_df.columns)]
            ) + "<extra></extra>",
            name=val
        ))

    fig.update_layout(yaxis_scaleanchor="x", showlegend=True, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# --- Show info on hover ---
@app.callback(Output("hover-info", "children"), Input("umap-plot", "hoverData"))
def show_hover_info(hoverData):
    if not hoverData:
        return "Hover over a dot to see annotation info."
    try:
        idx = hoverData["points"][0]["pointIndex"]
        label = data.index[idx]
        row = anno_df.loc[label]
        return f"Label: {label}\n" + "\n".join(f"{k}: {v}" for k, v in row.items())
    except Exception as e:
        return f"Error: {e}"


# --- Save PNG on Button Click ---
@app.callback(
    Output("save-confirmation", "children"),
    Input("save-png-btn", "n_clicks"),
    [
        Input("annotation-column", "value"),
        Input("selected-labels", "value"),
        Input("png-width", "value"),
        Input("png-height", "value"),
        Input("png-res", "value"),
        Input("png-filename", "value"),
        Input("png-path", "value"),
    ],
)
def save_png(n_clicks, col, selected_vals, width, height, res, filename, path):
    if n_clicks == 0:
        return ""

    import os

    plot_data = data.copy()

    if selected_vals:
        plot_data["__color__"] = plot_data[col].apply(
            lambda x: str(x) if x in selected_vals else "Other"
        )
    else:
        plot_data["__color__"] = plot_data[col].astype(str)

    def numeric_sort_key(x):
        try:
            return float(x)
        except ValueError:
            return float("inf")

    color_order = sorted(plot_data["__color__"].unique(), key=numeric_sort_key)

    fig = px.scatter(
        plot_data,
        x="UMAP1",
        y="UMAP2",
        color="__color__",
        hover_name=plot_data.index,
        hover_data=anno_df.columns.tolist(),
        category_orders={"__color__": color_order},
    )
    fig.update_traces(marker=dict(size=4, opacity=0.5))  # Add this

    # Build full output path
    if not filename.lower().endswith(".png"):
        filename += ".png"
    full_path = os.path.join(path, filename)

    try:
        fig.write_image(full_path, width=int(width * res), height=int(height * res), scale=1)
        return f"Plot saved to `{full_path}`"
    except Exception as e:
        return f"Error saving image: {e}"


# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
