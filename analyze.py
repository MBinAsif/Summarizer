import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')

st.title("NLP Text Analyzer")

user_input = st.text_area("Enter your text:", "Type here...")

if user_input:
    st.header("Summary:")
    
    # Load pre-trained BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + user_input, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    st.write(summary)

   # Your previous code for creating the Word Cloud plot
    st.header("Word Cloud:")
    wordcloud = WordCloud(stopwords=set(stopwords.words('english')), background_color='white').generate(user_input)
    plt.figure(figsize=(8, 6))  # Adjust the figsize as needed
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

# Display the Word Cloud plot using st.pyplot() with the explicit figure object
    st.pyplot(plt.gcf())

    st.header("Most Common Words:")
    words = word_tokenize(user_input)  # Tokenize the user input text
    fdist = nltk.FreqDist(words)
    most_common_words = fdist.most_common(10)

    # Prepare data for tabular format
    data = {
        "Word": [word[0] for word in most_common_words],
        "Frequency": [word[1] for word in most_common_words]
    }

    # Display as a table
    st.table(data)
