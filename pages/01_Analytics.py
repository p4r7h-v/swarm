import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Set page config
st.set_page_config(page_title="MiniSwarm Analytics", page_icon="📊")

# Title
st.title("MiniSwarm Analytics")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display basic information about the dataset
    st.subheader("Dataset Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Display the first few rows of the dataset
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Data visualization
    st.subheader("Data Visualization")
    
    # Select columns for visualization
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    x_axis = st.selectbox("Choose the X-axis", options=numeric_columns)
    y_axis = st.selectbox("Choose the Y-axis", options=numeric_columns)
    
    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    plt.title(f"{y_axis} vs {x_axis}")
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Custom query
    st.subheader("Custom Query")
    query = st.text_input("Enter a custom query (e.g., SELECT * FROM df LIMIT 5)")
    if query:
        try:
            result = pd.read_sql_query(query, StringIO(df.to_csv(index=False)))
            st.dataframe(result)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")

else:
    st.info("Please upload a CSV file to begin analysis.")

# Add some information about how to use the page
st.sidebar.header("How to use")
st.sidebar.info(
    "1. Upload a CSV file using the file uploader.\n"
    "2. Explore the dataset information and summary statistics.\n"
    "3. Visualize relationships between variables using the scatter plot.\n"
    "4. Examine correlations using the heatmap.\n"
    "5. Run custom SQL-like queries on your data."
)
