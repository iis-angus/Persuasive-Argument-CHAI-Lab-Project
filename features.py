import pandas as pd
import numpy as np
import re

explain_path = 'results/explain_then_predict_results.csv'
predict_path = 'results/predict_then_explain_results.csv'

#Loads the valence, arousal and dominance lexicons
def load_nrc_vad_lexicon():
    """
    Load NRC-VAD lexicon from the actual lexicon file.
    Returns dictionaries for valence, arousal, and dominance scores.
    """
    lexicon_path = 'original_data/NRC-VAD Lexicon/NRC-VAD-Lexicon-v2.1.txt'
    
    try:
        df = pd.read_csv(lexicon_path, sep='\t')
        
        valence_dict = dict(zip(df['term'], df['valence']))
        arousal_dict = dict(zip(df['term'], df['arousal']))
        dominance_dict = dict(zip(df['term'], df['dominance']))
        
        return valence_dict, arousal_dict, dominance_dict

    except Exception as e:
        print(e)
        return {}, {}, {}

#Loads the concreteness lexicon
def load_concreteness_lexicon():
    """
    Load Brysbaert concreteness ratings from Excel file.
    Returns a dictionary with words as keys and concreteness scores as values.
    """
    lexicon_path = 'original_data/Concreteness Lexicon/13428_2013_403_MOESM1_ESM.xlsx'
    
    try:
        df = pd.read_excel(lexicon_path)
        df['Conc.M'] /= 5 #normalize to range [0, 1]
        concreteness_dict = dict(zip(df['Word'].str.lower(), df['Conc.M']))
        return concreteness_dict
    
    except Exception as e:
        print(e)
        return None

#Splits the text into a list of words
def preprocess_text(text):
    """Cleans text for analysis"""
    if pd.isna(text) or text is None:
        return []

    text = str(text).lower()
    words = text.split()
    
    return words

# ----------- FEATURE FUNCTIONS ----------- #
#Valence
def average_valence_score(text):
    """
    Calculates the average valence score of a text using the NRC-VAD lexicon.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        float: Average valence score, scaled to range [0, 1]
    """
    
    if not valence_lexicon:
        return 0

    words = preprocess_text(text)
    
    if not words:
        return 0
    
    valence_scores = []
    
    for word in words:
        if word in valence_lexicon and abs(valence_lexicon[word]) > 0.7:
            valence_scores.append(valence_lexicon[word])
    
    if valence_scores:
        return (np.mean(valence_scores) + 1)/2
    else:
        return 0

#Arousal
def average_arousal_score(text):
    """
    Calculates the average arousal score of a text using the NRC-VAD lexicon.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        float: Average arousal score, scaled to range [0, 1]
    """    
    if not arousal_lexicon:
        return 0

    words = preprocess_text(text)
    
    if not words:
        return 0
    
    arousal_scores = []
    
    for word in words:
        if word in arousal_lexicon and abs(arousal_lexicon[word]) > 0.7:
            arousal_scores.append(arousal_lexicon[word])
    
    if arousal_scores:
        return (np.mean(arousal_scores) + 1)/2
    else:
        return 0

#Dominance
def average_dominance_score(text):
    """
    Calculates the average dominance score of a text using the NRC-VAD lexicon.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        float: Average dominance score, scaled to range [0, 1]
    """    
    if not dominance_lexicon:
        return 0

    words = preprocess_text(text)
    
    if not words:
        return 0
    
    dominance_scores = []
    
    for word in words:
        if word in dominance_lexicon and abs(dominance_lexicon[word]) > 0.7:
            dominance_scores.append(dominance_lexicon[word])
    
    if dominance_scores:
        return (np.mean(dominance_scores) + 1)/2
    else:
        return 0

#Concreteness
def average_concreteness_score(text):
    """
    Calculates the average concreteness score of a text using the Brysbaert lexicon.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        float: Average concreteness score (typically range 1-5, where higher is more concrete)
    """
    
    if not concreteness_lexicon:
        return 0

    words = preprocess_text(text)
    
    if not words:
        return 0
    
    concreteness_scores = []
    
    for word in words:
        if word in concreteness_lexicon and abs(concreteness_lexicon[word]) > 0.7:
            concreteness_scores.append(concreteness_lexicon[word])
    
    if concreteness_scores:
        return np.mean(concreteness_scores)
    else:
        return 0

#Returns word count of a text
def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

#Returns number of links/url in a text
def count_links(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r'https?://\S+', str(text)))
 
# ----------- MAIN FUNCTION ----------- #
#Processes features (MAIN WRAPPER)
def process_features(csv_path):
    df = pd.read_csv(csv_path)
    
    data = []
    for index, row in df.iterrows():
        r1_word_count = word_count(row['response_1'])
        r2_word_count = word_count(row['response_2'])
        r1_valence = average_valence_score(row['response_1'])
        r2_valence = average_valence_score(row['response_2'])
        r1_arousal = average_arousal_score(row['response_1'])
        r2_arousal = average_arousal_score(row['response_2'])
        r1_dominance = average_dominance_score(row['response_1'])
        r2_dominance = average_dominance_score(row['response_2'])
        r1_concreteness = average_concreteness_score(row['response_1'])
        r2_concreteness = average_concreteness_score(row['response_2'])
        r1_links = count_links(row['response_1'])
        r2_links = count_links(row['response_2'])

        # Determine sign based on prediction
        if row["prediction"] == 1:
            prediction_word_count_diff = r1_word_count - r2_word_count
            prediction_valence_diff = r1_valence - r2_valence
            prediction_arousal_diff = r1_arousal - r2_arousal
            prediction_dominance_diff = r1_dominance - r2_dominance
            prediction_concrete_diff = r1_concreteness - r2_concreteness
            prediction_link_diff = r1_links - r2_links
        else:
            prediction_word_count_diff = r2_word_count - r1_word_count
            prediction_valence_diff = r2_valence - r1_valence
            prediction_arousal_diff = r2_arousal - r1_arousal
            prediction_dominance_diff = r2_dominance - r1_dominance
            prediction_concrete_diff = r2_concreteness - r1_concreteness
            prediction_link_diff = r2_links - r1_links

        row_data = {
            'response_1_word_count': r1_word_count,
            'response_2_word_count': r2_word_count,
            'word_count_prediction_difference': prediction_word_count_diff,
            'response_1_valence': r1_valence,
            'response_2_valence': r2_valence,
            'valence_prediction_difference': prediction_valence_diff,
            'response_1_arousal': r1_arousal,
            'response_2_arousal': r2_arousal,
            'arousal_prediction_difference': prediction_arousal_diff,
            'response_1_dominance': r1_dominance,
            'response_2_dominance': r2_dominance,
            'dominance_prediction_difference': prediction_dominance_diff,
            'response_1_concreteness': r1_concreteness,
            'response_2_concreteness': r2_concreteness,
            'concreteness_prediction_difference': prediction_concrete_diff,
            'response_1_link_count': r1_links,
            'response_2_link_count': r2_links,
            'link_count_prediction_difference': prediction_link_diff
        }
        
        data.append(row_data)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    valence_lexicon, arousal_lexicon, dominance_lexicon = load_nrc_vad_lexicon()
    concreteness_lexicon = load_concreteness_lexicon()
    process_features(explain_path).to_csv('results/explain_features.csv', index=False)
    process_features(predict_path).to_csv('results/predict_features.csv', index=False)
