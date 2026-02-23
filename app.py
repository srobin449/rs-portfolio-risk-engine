import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="RS Portfolio Risk Engine", layout="wide")
st.title("RS Portfolio Risk Engine")

# =====================================
# SESSION STATE
# =====================================
if "run" not in st.session_state:
    st.session_state.run = False

# =====================================
# SIDEBAR INPUTS
# =====================================
with st.sidebar:
    st.header("Inputs")

    tickers_text = st.text_input(
        "Tickers (comma separated)",
        value="AAPL, MSFT, NVDA, AMZN, SPY"
    )

    start_date = st.date_input(
        "Start Date",
        value=dt.date(2020, 1, 1)
    )

    end_date = st.date_input(
        "End Date",
        value=dt.date.today()
    )

    risk_free = st.number_input(
        "Risk-Free Rate (Annual Decimal)",
        value=0.04
    )

    simulations = st.slider(
        "Monte Carlo Simulations",
        2000, 20000, 10000, 2000
    )

    custom_shock = st.slider(
        "Custom Market Shock (%)",
        -50, 0, -20
    )

    if st.button("Run Risk Analysis"):
        st.session_state.run = True

# =====================================
# MAIN ENGINE
# =====================================
if st.session_state.run:

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]

    if data.empty:
        st.error("No data returned.")
        st.stop()

    returns = data.pct_change().dropna()

    # =====================================
    # INDIVIDUAL STOCK METRICS
    # =====================================
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    individual_sharpe = (annual_returns - risk_free) / annual_volatility

    stock_metrics = pd.DataFrame({
        "Annual Return": annual_returns,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": individual_sharpe
    })

    st.subheader("Individual Stock Metrics")

    st.dataframe(
        stock_metrics.style.format({
            "Annual Return": "{:.2%}",
            "Annual Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}"
        })
    )

    # Risk / Return Scatter
    fig_ind, ax_ind = plt.subplots()
    ax_ind.scatter(annual_volatility, annual_returns)

    for i, txt in enumerate(stock_metrics.index):
        ax_ind.annotate(txt,
                        (annual_volatility[i], annual_returns[i]))

    ax_ind.set_xlabel("Annual Volatility")
    ax_ind.set_ylabel("Annual Return")
    ax_ind.set_title("Individual Stock Risk vs Return")
    st.pyplot(fig_ind)

    # =====================================
    # MONTE CARLO SIMULATION
    # =====================================
    results = np.zeros((3, simulations))
    weight_record = []

    for i in range(simulations):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weight_record.append(weights)

        port_return = np.sum(returns.mean() * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = (port_return - risk_free) / port_vol

        results[0, i] = port_return
        results[1, i] = port_vol
        results[2, i] = sharpe

    best_idx = np.argmax(results[2])
    best_weights = weight_record[best_idx]
    best_ret = results[0, best_idx]
    best_vol = results[1, best_idx]
    best_sharpe = results[2, best_idx]

    # Portfolio Metrics
    st.subheader("Optimal Portfolio Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{best_ret:.2%}")
    col2.metric("Expected Annual Volatility", f"{best_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{best_sharpe:.2f}")

    # Efficient Frontier
    st.subheader("Monte Carlo Efficient Frontier")

    fig, ax = plt.subplots()
    sc = ax.scatter(results[1, :], results[0, :],
                    c=results[2, :], cmap="viridis", alpha=0.4)
    ax.scatter(best_vol, best_ret, color="red", s=120, label="Optimal Portfolio")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.legend()
    fig.colorbar(sc, label="Sharpe Ratio")
    st.pyplot(fig)

    # Portfolio Weights
    st.subheader("Optimal Portfolio Weights")

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": best_weights
    })

    st.dataframe(weights_df)

    # Correlation Matrix
    st.subheader("Correlation Matrix")

    corr = returns.corr()
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(corr, cmap="coolwarm")
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45)
    ax2.set_yticklabels(corr.columns)
    fig2.colorbar(im)
    st.pyplot(fig2)

    # Drawdown
    st.subheader("Portfolio Drawdown")

    port_returns = returns @ best_weights
    cumulative = (1 + port_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    fig3, ax3 = plt.subplots()
    ax3.plot(drawdown)
    ax3.set_title("Drawdown Curve")
    st.pyplot(fig3)

    # Value at Risk
    st.subheader("Value at Risk (95%)")
    var = np.percentile(port_returns, 5)
    st.write(f"95% VaR: {var:.2%}")

    # Custom Shock
    st.subheader("Custom Market Shock Impact")
    shock_impact = custom_shock / 100
    st.write(f"If market drops {custom_shock}% → Portfolio impact ≈ {shock_impact:.2%}")

    # =====================================
    # PDF REPORT
    # =====================================
    def generate_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        style = ParagraphStyle(
            name='Normal',
            fontSize=12,
            textColor=colors.black
        )

        elements.append(Paragraph("<b>RS Portfolio Risk Engine Report</b>", style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Expected Annual Return: {best_ret:.2%}", style))
        elements.append(Paragraph(f"Expected Annual Volatility: {best_vol:.2%}", style))
        elements.append(Paragraph(f"Sharpe Ratio: {best_sharpe:.2f}", style))
        elements.append(Spacer(1, 0.3 * inch))

        table_data = [weights_df.columns.tolist()] + weights_df.values.tolist()
        table = Table(table_data)
        elements.append(table)

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf = generate_pdf()

    st.download_button(
        "Download Portfolio Report (PDF)",
        data=pdf,
        file_name="RS_Portfolio_Risk_Report.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("---")
st.caption("""
RS Portfolio Risk Engine v1.0  
Developed by Robinjot Singh  
Data Source: Yahoo Finance  
For educational purposes only  
© 2026
""")