import streamlit as st
import plotly.graph_objects as go
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from scipy import stats
from sklearn.linear_model import LinearRegression

# Title and description
st.markdown("""
    <div style="background-color: #2ECC40; padding: 10px; border-radius: 5px; color: white;">
        <h1>Data Analytics with Python</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color: #FF851B; padding: 10px; border-radius: 5px; color: white;">
        An infographic displaying key Python libraries for data analytics.
    </div>
""", unsafe_allow_html=True)

# Set background image
import streamlit as st

# Background image URL
background_image_url = "https://raw.githubusercontent.com/ravindranath8/data-analysis/main/download%20(2).jpg"

# Set background image using custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Your Streamlit app content
st.title("Streamlit with GitHub Image Background")
st.write("This is an example of setting a background image from GitHub.")


# Set the background image
#set_background(r"C:\Users\short\OneDrive\Pictures\Saved Pictures\download (2).jpg")
#https://github.com/ravindranath8/data-analysis/blob/main/download%20(2).jpg
#set_background(https://github.com/ravindranath8/data-analysis/blob/main/download%20(2).jpg)
# Define categories and tools
categories = [
    "Data Visualization", "Data Manipulation", "Statistical Analysis",
    "Web Scraping", "Natural Language Processing", "Time Series Analysis"
]

tools = [
    ["Plotly", "Matplotlib", "Seaborn"],
    ["NumPy", "Pandas", "Polars"],
    ["Statsmodels", "Pingouin", "SciPy"],
    ["Scrapy", "Beautiful Soup", "Selenium"],
    ["TextBlob", "NLTK", "BERT"],
    ["Darts", "Kats", "TSfresh"]
]

colors = ["#3D9970", "#FF4136", "#FF851B", "#2ECC40", "#B10DC9", "#FFDC00"]

# Create a Pie chart
fig = go.Figure(
    data=[go.Pie(
        labels=categories,
        values=[1] * len(categories),
        marker_colors=colors,
        textinfo='label+percent',
        hole=0.3
    )]
)

fig.update_layout(
    title_text="Python Libraries by Category",
    showlegend=False
)

# Display Pie chart
st.plotly_chart(fig)

# Display tools in each category with color boxes around the titles
st.markdown("""
    <div style="background-color: #3D9970; padding: 10px; border-radius: 5px; color: white;">
        <h2>Tools by Category</h2>
    </div>
""", unsafe_allow_html=True)

# Display tools and examples by category
for i, category in enumerate(categories):
    st.markdown(f"""
        <div style="background-color: {colors[i]}; padding: 5px 10px; border-radius: 5px; color: white;">
            <h3>{category}</h3>
        </div>
    """, unsafe_allow_html=True)
    st.write(", ".join(tools[i]))
    
    # Adding examples for each tool within the category
    if category == "Data Visualization":
        st.subheader("Example: Plotly")
        st.write("Plotly is used for creating interactive plots. Here's an example:")
        fig_plotly = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[10, 11, 12])])
        fig_plotly.update_layout(title="Plotly Bar Chart Example")
        st.plotly_chart(fig_plotly)
        
        st.subheader("Example: Matplotlib")
        st.write("Matplotlib is used for creating static, animated, and interactive visualizations.")
        fig_matplotlib, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Matplotlib Line Plot")
        st.pyplot(fig_matplotlib)
        
        st.subheader("Example: Seaborn")
        st.write("Seaborn is a Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.")
        df = sns.load_dataset("iris")
        fig_seaborn = sns.pairplot(df)
        st.pyplot(fig_seaborn)
    
    elif category == "Data Manipulation":
        st.subheader("Example: NumPy")
        st.write("NumPy is used for handling arrays and performing mathematical operations on them.")
        np_array = np.array([1, 2, 3, 4, 5])
        st.write(f"NumPy Array: {np_array}")
        
        st.subheader("Example: Pandas")
        st.write("Pandas is used for data manipulation and analysis.")
        df_pandas = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        st.write("Pandas DataFrame:")
        st.write(df_pandas)
        
        st.subheader("Example: Polars")
        st.write("Polars is a DataFrame library in Python that focuses on performance.")
        import polars as pl
        df_polars = pl.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        st.write("Polars DataFrame:")
        st.write(df_polars)
    
    elif category == "Statistical Analysis":
        st.subheader("Example: Statsmodels")
        st.write("Statsmodels is used for performing statistical tests and data exploration.")
        data = np.random.randn(100)
        st.write("Statsmodels summary:")
        import statsmodels.api as sm
        model = sm.OLS(data, sm.add_constant(np.ones(100)))
        result = model.fit()
        st.write(result.summary())
        
        
        
        
        
        st.subheader("Example: SciPy")
        st.write("SciPy is a library used for scientific and technical computing.")
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(np.random.randn(100), 0)
        st.write(f"T-statistic: {t_stat}, P-value: {p_val}")
    
    elif category == "Web Scraping":
        st.subheader("Example: Scrapy")
        st.write("Scrapy is a web scraping framework.")
        st.write("Scrapy is typically used for scraping data from websites and storing it in various formats like JSON or CSV.")
        st.write("For a full example, refer to the Scrapy documentation.")
        
        st.subheader("Example: Beautiful Soup")
        st.write("Beautiful Soup is used for parsing HTML and XML documents.")
        from bs4 import BeautifulSoup
        html_doc = "<html><head><title>Test Page</title></head><body><p>This is a test.</p></body></html>"
        soup = BeautifulSoup(html_doc, 'html.parser')
        st.write(f"Beautiful Soup parsed HTML: {soup.prettify()}")
        
        st.subheader("Example: Selenium")
        st.write("Selenium is a tool for automating web browsers. It can be used for web scraping or automating web applications.")
        st.write("For a full example, refer to the Selenium documentation.")
    
    elif category == "Natural Language Processing":
        st.subheader("Example: TextBlob")
        st.write("TextBlob is used for text processing tasks like sentiment analysis.")
        blob = TextBlob("Python is great!")
        st.write(f"Sentiment Analysis: {blob.sentiment}")
        
        
    

        
       
        st.subheader("Example: BERT")
        st.write("BERT is a transformer model used for a variety of NLP tasks.")
        st.write("For a full example, refer to the Hugging Face Transformers documentation.")
    
    elif category == "Time Series Analysis":
        st.subheader("Example: Darts")
        st.write("Darts is a library for time series forecasting.")
        st.write("For a full example, refer to the Darts documentation.")
        
        st.subheader("Example: Kats")
        st.write("Kats is a toolkit for time series analysis and forecasting developed by Facebook.")
        st.write("For a full example, refer to the Kats documentation.")
        
        st.subheader("Example: TSfresh")
        st.write("TSfresh is a Python library used for time series feature extraction.")
        st.write("For a full example, refer to the TSfresh documentation.")

# Footer
st.markdown("""
    <div style="background-color: #B10DC9; padding: 10px; border-radius: 5px; color: white;">
        **by Ravindra nath **
    </div>
""", unsafe_allow_html=True)
