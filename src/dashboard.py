# src/dashboard.py

import pandas as pd
import numpy as np
import os
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ----------------------------------------------------------------
# 1. DATA LOADERS
# ----------------------------------------------------------------

def load_backtest_results(ticker: str) -> pd.DataFrame:
    path = os.path.join("data", f"{ticker}_backtest_results.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_featured_data(ticker: str) -> pd.DataFrame:
    from features import add_features
    path = os.path.join("data", f"{ticker}_processed.csv")
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume",
               "log_return", "realized_vol", "vix"],
        index_col="Date",
        parse_dates=True
    )
    df = add_features(df)
    return df


def load_feature_importance(ticker: str) -> pd.Series:
    from xgboost import XGBRegressor
    feature_cols = [
        "vol_lag_1", "vol_lag_5", "vol_lag_10", "vol_lag_21",
        "vol_of_vol_5", "vol_of_vol_21",
        "mean_return_5", "mean_return_21",
        "vol_5d", "vol_63d", "vol_ratio", "abs_return",
        "vix_normalized", "vol_risk_premium", "vix_lag_1", "vix_rolling_5"
    ]
    model = XGBRegressor()
    model.load_model(os.path.join("models", f"{ticker}_xgb_model.json"))
    return pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=True)


# ----------------------------------------------------------------
# 2. METRICS
# ----------------------------------------------------------------

def compute_metrics(results: pd.DataFrame) -> dict:
    actual    = results["actual"].values
    predicted = results["predicted"].values
    mae  = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    corr = np.corrcoef(actual, predicted)[0, 1]
    actual_dir    = np.sign(actual[1:]    - actual[:-1])
    predicted_dir = np.sign(predicted[1:] - predicted[:-1])
    dir_acc = (actual_dir == predicted_dir).mean()
    return {
        "MAE":             f"{mae:.4f}",
        "RMSE":            f"{rmse:.4f}",
        "Correlation":     f"{corr:.4f}",
        "Directional Acc": f"{dir_acc:.2%}"
    }


# ----------------------------------------------------------------
# 3. DESIGN TOKENS
# ----------------------------------------------------------------

COLORS = {
    "bg":            "#070B14",
    "surface":       "#0D1421",
    "border":        "#1C2A3E",
    "accent":        "#F0A500",
    "actual":        "#00C9A7",
    "predicted":     "#F0A500",
    "residual_pos":  "#FF5E5E",
    "residual_neg":  "#00C9A7",
    "regime_high":   "rgba(255, 94, 94, 0.08)",
    "regime_low":    "rgba(0, 201, 167, 0.06)",
    "text_primary":  "#E8EDF5",
    "text_secondary":"#6B7E99",
}

FONT_MONO    = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace"
FONT_DISPLAY = "'IBM Plex Sans', 'Helvetica Neue', sans-serif"

TICKERS = ["SPY", "QQQ", "GLD"]

TICKER_LABELS = {
    "SPY": "SPY — S&P 500 ETF",
    "QQQ": "QQQ — Nasdaq 100 ETF",
    "GLD": "GLD — Gold ETF"
}


# ----------------------------------------------------------------
# 4. CHART BUILDERS
# ----------------------------------------------------------------

def build_vol_chart(results: pd.DataFrame,
                    start_date=None, end_date=None) -> go.Figure:
    df = results.copy()
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    fig = go.Figure()

    HIGH_VOL_THRESHOLD = 0.25
    LOW_VOL_THRESHOLD  = 0.12

    in_high = False
    in_low  = False
    high_start = low_start = None

    for i, (date, row) in enumerate(df.iterrows()):
        vol = row["actual"]

        if vol > HIGH_VOL_THRESHOLD and not in_high:
            high_start = date
            in_high = True
        elif vol <= HIGH_VOL_THRESHOLD and in_high:
            fig.add_vrect(
                x0=high_start, x1=date,
                fillcolor=COLORS["regime_high"],
                layer="below", line_width=0
            )
            in_high = False

        if vol < LOW_VOL_THRESHOLD and not in_low:
            low_start = date
            in_low = True
        elif vol >= LOW_VOL_THRESHOLD and in_low:
            fig.add_vrect(
                x0=low_start, x1=date,
                fillcolor=COLORS["regime_low"],
                layer="below", line_width=0
            )
            in_low = False

    if in_high:
        fig.add_vrect(x0=high_start, x1=df.index[-1],
                      fillcolor=COLORS["regime_high"],
                      layer="below", line_width=0)
    if in_low:
        fig.add_vrect(x0=low_start, x1=df.index[-1],
                      fillcolor=COLORS["regime_low"],
                      layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["actual"],
        name="Actual Vol",
        line=dict(color=COLORS["actual"], width=2),
        hovertemplate="<b>Actual</b>: %{y:.2%}<br>%{x}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["predicted"],
        name="Predicted Vol",
        line=dict(color=COLORS["predicted"], width=2, dash="dot"),
        hovertemplate="<b>Predicted</b>: %{y:.2%}<br>%{x}<extra></extra>"
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
        title=dict(
            text="Realized Volatility — Actual vs Predicted · "
                 "<span style='color:#FF5E5E'>■</span> High Vol Regime (>25%)  "
                 "<span style='color:#00C9A7'>■</span> Low Vol Regime (<12%)",
            font=dict(size=12, family=FONT_DISPLAY,
                      color=COLORS["text_primary"]),
            x=0.02
        ),
        legend=dict(bgcolor=COLORS["bg"], bordercolor=COLORS["border"],
                    borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor=COLORS["border"], showgrid=True,
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=COLORS["border"], showgrid=True,
                   zeroline=False, tickformat=".1%",
                   tickfont=dict(size=10),
                   title="Annualized Volatility"),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=40)
    )
    return fig


def build_residual_chart(results: pd.DataFrame,
                         start_date=None, end_date=None) -> go.Figure:
    df = results.copy()
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    residuals = df["predicted"] - df["actual"]
    colors = [COLORS["residual_pos"] if r > 0
              else COLORS["residual_neg"] for r in residuals]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.index,
        y=residuals,
        marker_color=colors,
        name="Residual",
        hovertemplate="<b>Residual</b>: %{y:.4f}<br>%{x}<extra></extra>"
    ))

    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
        title=dict(
            text="Prediction Residuals (Predicted − Actual)  "
                 "<span style='color:#FF5E5E'>■</span> Overpredicted  "
                 "<span style='color:#00C9A7'>■</span> Underpredicted",
            font=dict(size=12, family=FONT_DISPLAY,
                      color=COLORS["text_primary"]),
            x=0.02
        ),
        xaxis=dict(gridcolor=COLORS["border"], showgrid=True,
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=COLORS["border"], showgrid=True,
                   zeroline=False, tickformat=".3f",
                   tickfont=dict(size=10),
                   title="Residual"),
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40)
    )
    return fig


def build_feature_importance_chart(importance: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=importance.values, y=importance.index,
        orientation="h",
        marker=dict(
            color=importance.values,
            colorscale=[[0, COLORS["border"]], [1, COLORS["accent"]]],
            showscale=False
        ),
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
        title=dict(
            text="Feature Importance (XGBoost)",
            font=dict(size=12, family=FONT_DISPLAY,
                      color=COLORS["text_primary"]),
            x=0.02
        ),
        xaxis=dict(gridcolor=COLORS["border"], showgrid=True,
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=COLORS["border"], showgrid=False,
                   tickfont=dict(size=10)),
        margin=dict(l=130, r=20, t=50, b=40)
    )
    return fig


def build_scatter_chart(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["actual"], y=results["predicted"],
        mode="markers",
        marker=dict(color=COLORS["accent"], size=4, opacity=0.5),
        hovertemplate=(
            "<b>Actual</b>: %{x:.2%}<br>"
            "<b>Predicted</b>: %{y:.2%}<extra></extra>"
        )
    ))
    min_val = min(results["actual"].min(), results["predicted"].min())
    max_val = max(results["actual"].max(), results["predicted"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLORS["actual"], width=1, dash="dash"),
        name="Perfect Prediction", hoverinfo="skip"
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
        title=dict(
            text="Predicted vs Actual Volatility",
            font=dict(size=12, family=FONT_DISPLAY,
                      color=COLORS["text_primary"]),
            x=0.02
        ),
        xaxis=dict(title="Actual Vol", gridcolor=COLORS["border"],
                   tickformat=".1%", tickfont=dict(size=10)),
        yaxis=dict(title="Predicted Vol", gridcolor=COLORS["border"],
                   tickformat=".1%", tickfont=dict(size=10)),
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=50)
    )
    return fig


# ----------------------------------------------------------------
# 5. LAYOUT HELPERS
# ----------------------------------------------------------------

def metric_card(label: str, value: str) -> html.Div:
    return html.Div(
        style={
            "background": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
            "borderTop": f"2px solid {COLORS['accent']}",
            "padding": "16px 24px",
            "flex": "1",
            "minWidth": "140px"
        },
        children=[
            html.Div(label, style={
                "fontSize": "10px",
                "letterSpacing": "0.12em",
                "textTransform": "uppercase",
                "color": COLORS["text_secondary"],
                "fontFamily": FONT_MONO,
                "marginBottom": "8px"
            }),
            html.Div(value, style={
                "fontSize": "24px",
                "fontFamily": FONT_MONO,
                "color": COLORS["accent"],
                "fontWeight": "600"
            })
        ]
    )


# ----------------------------------------------------------------
# 6. APP + CSS INJECTION
# ----------------------------------------------------------------

app = Dash(__name__, title="Realized Vol Predictor")

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown */
            .Select-control {
                background-color: #0D1421 !important;
                border-color: #1C2A3E !important;
                color: #E8EDF5 !important;
            }
            .Select-menu-outer {
                background-color: #0D1421 !important;
                border-color: #1C2A3E !important;
            }
            .Select-option {
                background-color: #0D1421 !important;
                color: #E8EDF5 !important;
            }
            .Select-option:hover, .Select-option.is-focused {
                background-color: #1C2A3E !important;
            }
            .Select-value-label {
                color: #E8EDF5 !important;
            }
            .Select-placeholder {
                color: #6B7E99 !important;
            }
            .Select-arrow {
                border-top-color: #6B7E99 !important;
            }

            /* Date picker */
            .DateInput, .DateInput_input,
            .DateRangePickerInput,
            .DateRangePickerInput__withBorder {
                background-color: #0D1421 !important;
                border-color: #1C2A3E !important;
                color: #E8EDF5 !important;
            }
            .DateInput_input {
                color: #E8EDF5 !important;
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 12px !important;
            }
            .DateRangePickerInput_arrow {
                color: #6B7E99 !important;
            }
            .DateRangePickerInput_arrow_svg {
                fill: #6B7E99 !important;
            }
            .DayPicker, .CalendarMonth, .CalendarMonthGrid {
                background-color: #0D1421 !important;
            }
            .CalendarDay__default {
                background: #0D1421 !important;
                color: #E8EDF5 !important;
                border-color: #1C2A3E !important;
            }
            .CalendarDay__selected {
                background: #F0A500 !important;
                color: #070B14 !important;
            }
            .CalendarDay__hovered_span,
            .CalendarDay__selected_span {
                background: #A67300 !important;
                color: #E8EDF5 !important;
            }
            .DayPickerNavigation_button {
                border-color: #1C2A3E !important;
                background-color: #0D1421 !important;
            }
            .CalendarMonth_caption {
                color: #E8EDF5 !important;
            }
            .DayOfWeek, .CalendarMonth_caption strong {
                color: #6B7E99 !important;
            }

            /* Global */
            body {
                margin: 0;
                background-color: #070B14;
            }
            * { box-sizing: border-box; }
            /* Fix white text in dropdown and date inputs */
            .VirtualizedSelectOption {
                background-color: #0D1421 !important;
                color: #E8EDF5 !important;
            }
            .Select-input input {
                color: #E8EDF5 !important;
                background-color: #0D1421 !important;
            }
            .Select--single .Select-value {
                color: #E8EDF5 !important;
            }
            input[type="text"],
            input[type="text"]::placeholder,
            input, input:focus, input:active {
                color: #E8EDF5 !important;
                background-color: #0D1421 !important;
                caret-color: #F0A500 !important;
                -webkit-text-fill-color: #E8EDF5 !important;
            }
            input::placeholder {
                color: #6B7E99 !important;
                -webkit-text-fill-color: #6B7E99 !important;
            }
            .DateInput_input__focused {
                background-color: #0D1421 !important;
                color: #E8EDF5 !important;
                border-bottom-color: #F0A500 !important;
            }
            .DateRangePickerInput_arrow_svg path {
                fill: #6B7E99 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    style={
        "backgroundColor": COLORS["bg"],
        "minHeight": "100vh",
        "fontFamily": FONT_DISPLAY,
        "color": COLORS["text_primary"]
    },
    children=[

        # HEADER
        html.Div(
            style={
                "borderBottom": f"1px solid {COLORS['border']}",
                "padding": "20px 32px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between"
            },
            children=[
                html.Div([
                    html.Span("REALIZED VOLATILITY PREDICTOR", style={
                        "fontFamily": FONT_MONO,
                        "fontSize": "12px",
                        "letterSpacing": "0.15em",
                        "color": COLORS["text_primary"]
                    }),
                ]),
                html.Div(
                    "XGBoost · VIX Features · Walk-Forward Backtest · 2021–2024",
                    style={
                        "fontFamily": FONT_MONO,
                        "fontSize": "10px",
                        "color": COLORS["text_secondary"],
                        "letterSpacing": "0.08em"
                    }
                )
            ]
        ),

        # CONTROLS ROW
        html.Div(
            style={
                "padding": "16px 32px",
                "borderBottom": f"1px solid {COLORS['border']}",
                "display": "flex",
                "alignItems": "center",
                "gap": "32px"
            },
            children=[
                html.Div([
                    html.Label("TICKER", style={
                        "fontFamily": FONT_MONO,
                        "fontSize": "10px",
                        "letterSpacing": "0.12em",
                        "color": COLORS["text_secondary"],
                        "display": "block",
                        "marginBottom": "6px"
                    }),
                    dcc.Dropdown(
                        id="ticker-dropdown",
                        options=[
                            {"label": TICKER_LABELS[t], "value": t}
                            for t in TICKERS
                        ],
                        value="SPY",
                        clearable=False,
                        style={
                            "width": "260px",
                            "fontFamily": FONT_MONO,
                            "fontSize": "12px",
                            "backgroundColor": COLORS["surface"],
                            "color": COLORS["text_primary"],
                            "border": f"1px solid {COLORS['border']}"
                        }
                    )
                ]),
                html.Div([
                    html.Label("DATE RANGE", style={
                        "fontFamily": FONT_MONO,
                        "fontSize": "10px",
                        "letterSpacing": "0.12em",
                        "color": COLORS["text_secondary"],
                        "display": "block",
                        "marginBottom": "6px"
                    }),
                    dcc.DatePickerRange(
                        id="date-picker",
                        display_format="YYYY-MM-DD",
                        style={"fontFamily": FONT_MONO}
                    )
                ])
            ]
        ),

        # METRICS BAR
        html.Div(id="metrics-bar", style={
            "display": "flex",
            "gap": "1px",
            "backgroundColor": COLORS["border"],
            "borderBottom": f"1px solid {COLORS['border']}"
        }),

        # CHARTS
        html.Div(
            style={"padding": "24px 32px"},
            children=[

                # Main vol chart
                html.Div(
                    style={
                        "background": COLORS["surface"],
                        "border": f"1px solid {COLORS['border']}",
                        "marginBottom": "16px"
                    },
                    children=[dcc.Graph(id="vol-chart",
                                        config={"displayModeBar": False})]
                ),

                # Residual chart
                html.Div(
                    style={
                        "background": COLORS["surface"],
                        "border": f"1px solid {COLORS['border']}",
                        "marginBottom": "16px"
                    },
                    children=[dcc.Graph(id="residual-chart",
                                        config={"displayModeBar": False})]
                ),

                # Bottom row
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px"
                    },
                    children=[
                        html.Div(
                            style={
                                "background": COLORS["surface"],
                                "border": f"1px solid {COLORS['border']}"
                            },
                            children=[dcc.Graph(id="importance-chart",
                                                config={"displayModeBar": False})]
                        ),
                        html.Div(
                            style={
                                "background": COLORS["surface"],
                                "border": f"1px solid {COLORS['border']}"
                            },
                            children=[dcc.Graph(id="scatter-chart",
                                                config={"displayModeBar": False})]
                        )
                    ]
                ),

                # Footer
                html.Div(
                    style={
                        "marginTop": "24px",
                        "paddingTop": "16px",
                        "borderTop": f"1px solid {COLORS['border']}",
                        "fontFamily": FONT_MONO,
                        "fontSize": "10px",
                        "color": COLORS["text_secondary"],
                        "display": "flex",
                        "justifyContent": "space-between"
                    },
                    children=[
                        html.Span(
                            "Data: Yahoo Finance via yfinance · "
                            "Model: XGBoost · "
                            "Features: Realized Vol + VIX · "
                            "Backtest: Walk-Forward (quarterly retraining)"
                        ),
                        html.Span("Elliot Becker · github.com/Elliot-Becker")
                    ]
                )
            ]
        )
    ]
)


# ----------------------------------------------------------------
# 7. CALLBACKS
# ----------------------------------------------------------------

@app.callback(
    Output("date-picker", "min_date_allowed"),
    Output("date-picker", "max_date_allowed"),
    Output("date-picker", "start_date"),
    Output("date-picker", "end_date"),
    Input("ticker-dropdown", "value")
)
def update_date_range(ticker):
    results = load_backtest_results(ticker)
    mn = results.index.min().date()
    mx = results.index.max().date()
    return mn, mx, mn, mx


@app.callback(
    Output("metrics-bar",      "children"),
    Output("vol-chart",        "figure"),
    Output("residual-chart",   "figure"),
    Output("importance-chart", "figure"),
    Output("scatter-chart",    "figure"),
    Input("ticker-dropdown",   "value"),
    Input("date-picker",       "start_date"),
    Input("date-picker",       "end_date")
)
def update_all(ticker, start_date, end_date):
    results    = load_backtest_results(ticker)
    importance = load_feature_importance(ticker)
    metrics    = compute_metrics(results)

    metrics_bar = [
        metric_card(label, value)
        for label, value in metrics.items()
    ]

    return (
        metrics_bar,
        build_vol_chart(results, start_date, end_date),
        build_residual_chart(results, start_date, end_date),
        build_feature_importance_chart(importance),
        build_scatter_chart(results)
    )


# ----------------------------------------------------------------
# 8. RUN
# ----------------------------------------------------------------

if __name__ == "__main__":
    print("\nStarting dashboard...")
    print("Open your browser to: http://127.0.0.1:8050\n")
    app.run(debug=True)