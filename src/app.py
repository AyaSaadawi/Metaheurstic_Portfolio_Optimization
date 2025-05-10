import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso import pso_optimize
from ga import genetic_algorithm_optimize
from sa import simulated_annealing_optimize
from hybrid_pso_ga import hybrid_pso_ga_optimize

# --- Title ---
st.set_page_config(page_title="Portfolio Optimizer Dashboard", page_icon="📈", layout="wide")
st.title("📈 Portfolio Optimizer Dashboard")
st.markdown("""
    **Upload asset returns data** and optimize portfolio using **PSO**, **GA**, **SA**, or **Hybrid** approaches.
    <br><br>
    Optimize your portfolio to achieve the best Sharpe ratio!
""", unsafe_allow_html=True)

# --- Upload CSV or Excel ---
uploaded_file = st.file_uploader("🔼 Upload CSV or Excel file with asset returns", type=["csv", "xlsx"])

if uploaded_file:
    # Check the file type and load accordingly
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        returns_df = pd.read_csv(uploaded_file)
    elif file_extension == 'xlsx':
        returns_df = pd.read_excel(uploaded_file)

    # Debug: Show raw dataframe
    st.success("File loaded successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(returns_df.head())

    # Drop non-numeric columns (e.g., date, strings)
    returns_df = returns_df.select_dtypes(include=[np.number])

    # Check for empty or bad data
    if returns_df.empty:
        st.error("❌ Uploaded file contains no numeric return data.")
        st.stop()

    # --- Sidebar Settings ---
    st.sidebar.header("🛠️ Optimization Settings")
    iterations = st.sidebar.slider("Iterations", 50, 500, step=50, value=100)
    optimizer = st.sidebar.selectbox("Select Optimizer", ["PSO", "GA", "SA", "Hybrid"])
    run_button = st.sidebar.button("🚀 Run Optimization")

    # --- Run Optimization ---
    if run_button:
        st.info(f"Running {optimizer}... Please wait.")

        try:
            # Running the chosen optimization algorithm
            if optimizer == "PSO":
                best_weights, best_score, history, exec_time = pso_optimize(returns_df, max_iter=iterations)[:4]

            elif optimizer == "GA":
                best_weights, best_score, history, exec_time = genetic_algorithm_optimize(returns_df, max_iter=iterations)[:4]

            elif optimizer == "SA":
                best_weights, best_score, history, exec_time = simulated_annealing_optimize(returns_df, max_iter=iterations)[:4]

            elif optimizer == "Hybrid":
                best_weights, best_score, history, exec_time = hybrid_pso_ga_optimize(returns_df, max_iter=iterations)[:4]

            # --- Results Section ---
            st.subheader("📊 Optimization Results")
            st.metric("Best Sharpe Ratio", f"{best_score:.5f}")
            st.metric("Execution Time (s)", f"{exec_time:.2f}")

            # --- Portfolio Allocation Pie Chart ---
            st.subheader("📉 Optimal Portfolio Allocation")
            weights = np.maximum(best_weights, 0)  # Make weights non-negative
            if np.sum(weights) == 0:
                st.error("❌ All portfolio weights are zero or negative. Cannot plot pie chart.")
            else:
                weights /= np.sum(weights)  # Normalize
                fig1, ax1 = plt.subplots()
                ax1.pie(weights, labels=returns_df.columns, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
                st.pyplot(fig1)

            # --- Sharpe Ratio Over Iterations Line Plot ---
            st.subheader("📈 Sharpe Ratio Over Iterations")
            fig2, ax2 = plt.subplots()
            ax2.plot(history, color='b', marker='o', markersize=5, linestyle='-', linewidth=2)
            ax2.set_xlabel("Iteration", fontsize=12)
            ax2.set_ylabel("Sharpe Ratio", fontsize=12)
            ax2.set_title("Sharpe Ratio Evolution", fontsize=14)
            st.pyplot(fig2)

            # --- Compare All Optimizers Table ---
            st.subheader("📋 Compare All Optimizers")

            all_results = {}
            for algo_name, algo_func in {
                "PSO": pso_optimize,
                "GA": genetic_algorithm_optimize,
                "SA": simulated_annealing_optimize,
                "Hybrid PSO-GA": hybrid_pso_ga_optimize,
            }.items():
                try:
                    results = algo_func(returns_df, max_iter=iterations)[:4]
                    _, score, _, t = results
                    all_results[algo_name] = {
                        "Sharpe Ratio": round(score, 5),
                        "Time (s)": round(t, 2)
                    }
                except Exception as e:
                    all_results[algo_name] = {
                        "Sharpe Ratio": "Error",
                        "Time (s)": "Error"
                    }
                    st.warning(f"{algo_name} failed: {str(e)}")

            st.table(pd.DataFrame(all_results).T)

        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")

else:
    st.warning("⚠️ Please upload a CSV or Excel file to continue.")