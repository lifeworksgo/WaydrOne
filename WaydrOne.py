import string
import streamlit as st
import oneai
import csv
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go


from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


#import transformers
oneai.api_key = '2caaa40d-bdbe-4a3c-933c-34e28b269983'
pipeline = oneai.Pipeline(steps=[
    oneai.skills.Topics(),
     oneai.skills.Entities(),
     oneai.skills.Sentiments()
])


st.set_page_config(page_title="Classifier", page_icon=None, initial_sidebar_state="expanded", menu_items=None, layout="wide")

st.title("Text Classifier")
my_text = st.text_area('Enter Text Here')

#uploaded_file=st.sidebar.file_uploader("")



if st.button("Start"):
    results = pipeline.run(my_text)
    dfSentiment = pd.DataFrame([sentiment.value for sentiment in results.Sentiments], columns=['Sentiments'])
    dfTopics = pd.DataFrame([topic.value for topic in results.topics], columns=['Themes'])
    dfEntities = pd.DataFrame([entities.name for entities in results.entities], columns=['Entities'])
    
    #pd.pivot_table(dfSentiment, values = 'Value', index=['Country','Year'], columns = 'Indicator').reset_index()

    st.markdown("""
        <style>

        .highlight {
        background-image: linear-gradient(to right, #F27121cc, #E94057cc, #8A2387cc);
        border-radius: 12px;
        padding: 3px 6px;
        }

        body {
        background-color: #110e17;
        }

        p {
        color: #fff;
        text-align: left;
        font-family: sans-serif;
        letter-spacing: 2px;
        padding: 20px 0;
        font-size: 2em;
        }

        </style>
        """, unsafe_allow_html=True)
    # st.write(results)
    st.subheader("Themes Identified")
    st.markdown('<p class="highlight">' + ', '.join(dfTopics["Themes"]) + '</p>', unsafe_allow_html=True)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Entities Identified")
    st.markdown('<p class="highlight">' + ', '.join(dfEntities["Entities"]) + '</p>', unsafe_allow_html=True)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Summary")
    

    
    long_string = my_text
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    wordcloud.to_image()


    st.image(wordcloud.to_image())


    columns = dfSentiment.columns.tolist()
    s = dfSentiment["Sentiments"].str.strip().value_counts()
    trace = go.Bar(x=s.index, y=s.values, showlegend=False)
    layout = go.Layout(title = "No. of comments")
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    #st.plotly_chart(fig)
 
    
    #st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    