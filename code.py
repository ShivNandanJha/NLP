# IMPORT NECESSARY LIBRARIES
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from textstat import syllable_count
import requests
from bs4 import BeautifulSoup

# Web Scrapping

# Read the Excel file
df = pd.read_excel("./NLP assignment/input.xlsx")

# Function to extract article text from URL


def extract_article_text(URL):
    try:

        headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
        # Here the user agent is for Edge browser on windows 10. You can find your browser user agent from the above given link.
        # Send a GET request to fetch the webpage content
        response = requests.get(url=URL, headers=headers)
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the article title
        title = soup.find('title').text.strip()
        # Find the article text
        article_text = soup.find(
            'div', class_='td-post-content tagdiv-type').text.strip()
        return title, article_text
    except Exception as e:
        print("Error extracting article from {url}: {e}")
        return None, None


# Create a directory to save text files if it doesn't exist
if not os.path.exists("extracted_articles"):
    os.makedirs("extracted_articles")

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract URL and URL_ID from the DataFrame
    URL = row['URL']
    URL_ID = row['URL_ID']

    # Extract article text from UR
    title, article_text = extract_article_text(URL)

    if title and article_text:
        # Save the extracted article text to a text file
        filename = f"extracted_articles/{URL_ID}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(title + "\n\n")
            file.write(article_text)
        print(f"Article extracted and saved: {filename}")
    else:
        print(f"Article extraction failed for URL_ID: {URL_ID}")

print("Extraction completed.")


## Removing Stopwords
# Load stop words
stop_words_path = './NLP assignment/StopWords/'
stop_words_files = os.listdir(stop_words_path)
stop_words = set()
for file in stop_words_files:
    with open(os.path.join(stop_words_path, file), 'r') as f:
        stop_words.update(f.read().splitlines())
        
# Function to remove stop words
def remove_stop_words(text, stop_words):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

# Load positive and negative words
positive_words_path = './MasterDictionary/positive-words.txt'
negative_words_path = './MasterDictionary/negative-words.txt'
with open(positive_words_path, 'r') as f:
    positive_words = set(f.read().splitlines())
with open(negative_words_path, 'r') as f:
    negative_words = set(f.read().splitlines())
# Function to count positive and negative words
def count_sentiment_words(text, positive_words, negative_words):
    positive_count = sum(1 for word in text.split() if word in positive_words)
    negative_count = sum(1 for word in text.split() if word in negative_words)
    return positive_count, negative_count

stop_words_path = './NLP assignment/StopWords/'
stop_words_files = os.listdir(stop_words_path)
stop_words = set()
for file in stop_words_files:
    def calculate_readability(text):
        # Average Number of Words Per Sentence
        sentences = sent_tokenize(text)
        total_words = len(word_tokenize(text))
        total_sentences = len(sentences)
        average_words_per_sentence = total_words / (total_sentences + 0.000001)

        # Abstract Average Sentence Length
        abstract_average_sentence_length = len(text) / (total_sentences + 0.000001)

        # Complex Word Count and Percentage of Complex Words
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        complex_words = [word for word in words if syllable_count(word) > 2]
        
        complex_word_count = len(complex_words)
        percentage_complex_words = (complex_word_count / (total_words + 0.000001)) * 100

        # Fog Index
        fog_index = 0.4 * (average_words_per_sentence + percentage_complex_words)

        # Word Count
        word_count = total_words

        # Syllable Count Per Word
        syllable_count_per_word = sum(syllable_count(word) for word in words) / (total_words + 0.000001)

        # Personal Pronouns Count
        personal_pronouns_count = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.IGNORECASE))

        # Average Word Length
        average_word_length = sum(len(word) for word in words) / (total_words + 0.000001)

        return average_words_per_sentence, abstract_average_sentence_length, percentage_complex_words, complex_word_count, fog_index, word_count, syllable_count_per_word, personal_pronouns_count, average_word_length

    # Load positive and negative words
    positive_words_path = './NLP assignment/MasterDictionary/positive-words.txt'
    negative_words_path = './NLP assignment/MasterDictionary/negative-words.txt'
    with open(positive_words_path, 'r') as f:
        positive_words = set(f.read().splitlines())
    with open(negative_words_path, 'r') as f:
        negative_words = set(f.read().splitlines())

    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    output_data = {'file_name': [], 'positive': [], 'negative': [], 'polarity_score': [],
                   'subjectivity_score': [], 'average_words_per_sentence': [],
                   'abstract_average_sentence_length': [], 'percentage_complex_words': [],'complex_words_count': [], 'fog_index': [],
                   'word_count': [], 'syllable_count_per_word': [],
                   'personal_pronouns_count': [], 'average_word_length': []}


    articles_folder = './NLP assignment/extracted_articles/'
    for file_name in os.listdir(articles_folder):
        with open(os.path.join(articles_folder, file_name), 'r',  encoding='utf-8' ) as f:
            article_text = f.read()
            # Remove stop words
            clean_text = ' '.join([word for word in word_tokenize(article_text) if word.lower() not in stop_words])
            
            # Calculate sentiment scores
            positive_score, negative_score, polarity_score, subjectivity_score = calculate_derived_variables(word_tokenize(clean_text), positive_words, negative_words, len(word_tokenize(clean_text)))
            
            # Calculate readability metrics
            average_words_per_sentence, abstract_average_sentence_length, percentage_complex_words,  complex_word_count ,fog_index, word_count, syllable_count_per_word, \
            personal_pronouns_count, average_word_length = calculate_readability(article_text)

            output_data['file_name'].append(file_name)
            output_data['positive'].append(positive_score)
            output_data['negative'].append(negative_score)
            output_data['polarity_score'].append(polarity_score)
            output_data['subjectivity_score'].append(subjectivity_score)
            output_data['average_words_per_sentence'].append(average_words_per_sentence)
            output_data['abstract_average_sentence_length'].append(abstract_average_sentence_length)
            output_data['percentage_complex_words'].append(percentage_complex_words)
            output_data['complex_words_count'].append( complex_word_count)
            output_data['fog_index'].append(fog_index)
            output_data['word_count'].append(word_count)
            output_data['syllable_count_per_word'].append(syllable_count_per_word)
            output_data['personal_pronouns_count'].append(personal_pronouns_count)
            output_data['average_word_length'].append(average_word_length)

    # Write output to Excel
output_df = pd.DataFrame(output_data)
output_df.to_excel('output_analyze.xlsx', index=False)
