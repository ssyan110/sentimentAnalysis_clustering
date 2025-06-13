import json
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load cleaned reviews and pre-computed company features
REVIEWS_PATH = 'clean_reviews.csv'
COMPANY_PATH = 'company_df.csv'
TERMS_PATH = 'cluster_terms.json'

@st.cache_data(show_spinner=False)
def load_data():
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_company = pd.read_csv(COMPANY_PATH)
    try:
        with open(TERMS_PATH, 'r') as f:
            cluster_terms = json.load(f)
    except FileNotFoundError:
        cluster_terms = {}
    return df_reviews, df_company, cluster_terms

df_reviews, company_df, cluster_terms = load_data()

st.title('ITviec Company Insights')
company_list = company_df['CompanyName'].tolist()
company_name = st.sidebar.selectbox('Select company', company_list)
company_row = company_df[company_df['CompanyName'] == company_name].iloc[0]

st.header(company_name)
st.write(f"Company ID: {int(company_row['id'])}")

cluster_cols = [c for c in company_df.columns if c.startswith('cluster_')]
if cluster_cols:
    st.subheader('Cluster assignments')
    for col in cluster_cols:
        st.write(f"{col}: {int(company_row[col])}")

# Sentiment distribution for the selected company
sent_counts = df_reviews[df_reviews['id'] == company_row['id']]['sentiment'].value_counts()
if not sent_counts.empty:
    st.subheader('Sentiment distribution')
    st.bar_chart(sent_counts)

# Word cloud of aggregated reviews
text = company_row['doc'] if 'doc' in company_row else ' '.join(df_reviews[df_reviews['id']==company_row['id']]['clean_review'])
if text:
    st.subheader('Word cloud')
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False)
    wc.generate(text)
    st.image(wc.to_array())

# Top cluster terms if available
if cluster_terms:
    st.subheader('Top TF-IDF terms per cluster')
    cluster_id = int(company_row.get('cluster_kmeans', 0))
    terms = cluster_terms.get(str(cluster_id)) or cluster_terms.get(cluster_id)
    if terms:
        st.write(', '.join(terms))
    else:
        st.write('No terms saved for this cluster.')
