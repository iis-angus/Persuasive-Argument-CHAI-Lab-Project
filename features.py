import pandas as pd
import numpy as np

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
        
        concreteness_dict = dict(zip(df['Word'].str.lower(), df['Conc.M']))
        concreteness_dict['Conc.M'] /= 5 #normalize to range [0, 1]
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
        if word in valence_lexicon:
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
        if word in arousal_lexicon:
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
        if word in dominance_lexicon:
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
        if word in concreteness_lexicon:
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

# ----------- MAIN FUNCTION ----------- #
#Processes features (MAIN WRAPPER)
def process_features(csv_path):
    df = pd.read_csv(csv_path)
    
    data = []
    for index, row in df.iterrows():
        row_data = {
            'response_1_word_count': word_count(row['response_1']),
            'response_2_word_count': word_count(row['response_2']),
            'response_1_valence': average_valence_score(row['response_1']),
            'response_2_valence': average_valence_score(row['response_2']),
            'response_1_arousal': average_arousal_score(row['response_1']),
            'response_2_arousal': average_arousal_score(row['response_2']),
            'response_1_dominance': average_dominance_score(row['response_1']),
            'response_2_dominance': average_dominance_score(row['response_2']),
            'response_1_concreteness': average_concreteness_score(row['response_1']),
            'response_2_concreteness': average_concreteness_score(row['response_2'])
        }
        
        data.append(row_data)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    valence_lexicon, arousal_lexicon, dominance_lexicon = load_nrc_vad_lexicon()
    concreteness_lexicon = load_concreteness_lexicon()
    process_features(explain_path).to_csv('results/explain_features.csv', index=False)
    process_features(predict_path).to_csv('results/predict_features.csv', index=False)

