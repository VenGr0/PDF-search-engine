import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Загрузка необходимых данных для nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_text(text):
    """Предварительная обработка текста: удаление стоп-слов и пунктуации."""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    """Извлечение текста из PDF-документа."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def search_in_text(text, query, top_n=3):
    """Поиск релевантных фрагментов текста по запросу."""
    # Разделение текста на предложения
    sentences = nltk.sent_tokenize(text)
    
    # Предварительная обработка предложений и запроса
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    processed_query = preprocess_text(query)
    
    # Создание TF-IDF матрицы
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences + [processed_query])
    
    # Вычисление косинусного сходства между запросом и предложениями
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Сортировка предложений по убыванию сходства
    ranked_sentences = [sentences[i] for i in cosine_similarities.argsort()[::-1]]
    
    # Возврат топ-N релевантных предложений
    return ranked_sentences[:top_n]

def main():
    # Путь к PDF-документу
    pdf_path = '1984.pdf'  # Замените на путь к вашему PDF-документу
    
    # Извлечение текста из PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Ввод поискового запроса от пользователя
    query = input("Введите поисковый запрос: ")
    
    # Поиск релевантных фрагментов текста
    relevant_sentences = search_in_text(text, query)
    
    # Вывод топ-3 релевантных фрагментов
    print("\nТоп-3 релевантных фрагмента:")
    for i, sentence in enumerate(relevant_sentences, 1):
        print(f"{i}. {sentence}")

if __name__ == "__main__":
    main()