import os
from openai import OpenAI
import pandas as pd
import random
import time #I ran into rate limit issues :c

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Basic gpt prompt
def gpt_query(prompt, model="gpt-3.5-turbo", max_tokens=500, temperature=0):
    """
    Runs a basic GPT query with rate limiting handling

    Args:
        prompt (str): The prompt to send to GPT
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 500.
        temperature (float, optional): The temperature to use. Defaults to 0.
    
    Returns:
        str: The model's response.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                print(f"Rate limit hit, waiting 1 second before retry {attempt + 1}...")
                time.sleep(1)
            else:
                raise e

#Prompt outline for "explain-then-predict" method
def prompt_outline_explain_then_predict(response_1, response_2):
    """
    A prompt outline for the "explain-then-predict" method to later plug into
    gpt_query function. Returns a structured format for easy DataFrame storage.

    Args:
        response_1 (str): The first response to the post
        response_2 (str): The second response to the post

    Returns:
        str: The prompt outline for the "explain-then-predict" method
        int: The correct response (1 or 2)
    """

    #shuffling responses
    responses = [response_1, response_2]
    shuffled_responses = responses.copy()
    random.shuffle(shuffled_responses)
    shuffled_response_1 = shuffled_responses[0]
    shuffled_response_2 = shuffled_responses[1]

    return (f"""
    Here is the text from a post on r/ChangeMyView. I will give you two responses
    and your job is to predict which response changed the mind of the OP. Please
    explain your reasoning for why one response is more persuasive, then predict
    which response changed the mind of the OP.
    
    Please format your response EXACTLY as follows:
    [open curly bracket]
    Explanation: [Your reasoning for why one response is more persuasive]
    Prediction: [Either "1" or "2"]
    [closed curly bracket]

    Response 1:
    {shuffled_response_1}

    Response 2:
    {shuffled_response_2}
    """,
    shuffled_responses.index(response_1) + 1)

#Prompt outline for "predict-then-explain" method
def prompt_outline_predict_then_explain(response_1, response_2):
    """
    A prompt outline for the "predict-then-explain" method to later plug into
    gpt_query function. Returns a structured format for easy DataFrame storage.

    Args:
        response_1 (str): The first response to the post
        response_2 (str): The second response to the post

    Returns:
        str: The prompt outline for the "predict-then-explain" method
        int: The correct response (1 or 2)
    """

    #shuffling responses
    responses = [response_1, response_2]
    shuffled_responses = responses.copy()
    random.shuffle(shuffled_responses)
    shuffled_response_1 = shuffled_responses[0]
    shuffled_response_2 = shuffled_responses[1]

    return (f"""
    Here is the text from a post on r/ChangeMyView. I will give you two responses
    and your job is to predict which response changed the mind of the OP. Please
    predict which response changed the mind of the OP, then explain your
    reasoning for why one response is more persuasive
    
    Please format your response EXACTLY as follows:
    [open curly bracket]
    Prediction: [Either "1" or "2"]
    Explanation: [Your reasoning for why one response is more persuasive]
    [closed curly bracket]

    Response 1:
    {shuffled_response_1}

    Response 2:
    {shuffled_response_2}
    """,
    shuffled_responses.index(response_1) + 1)

#Explain_then_predict_experiment
def explain_then_predict_experiment(temperature=0):
    """
    Runs the explain-then-predict experiment on the heldout pair data, then stores
    it into a csv file.

    Args:
        temperature (float, optional): The temperature to use when prompting 
        the ChatGPT model. Defaults to 0.
    """
    data = pd.read_csv("sample_data/heldout_pair_data.csv")
    all_results = []
    
    for index, row in data.iterrows():
        response_1 = row['positive']
        response_2 = row['negative']

        prompt_outline, correct_response = prompt_outline_explain_then_predict(response_1, response_2)
        response = gpt_query(prompt_outline, temperature=temperature)
        
        #Parse response to extract prediction and explanation
        response_clean = response.strip().strip('{}')
        lines = response_clean.split('\n')
        
        explanation = ""
        prediction = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Explanation:'):
                explanation = line.replace('Explanation:', '').strip()
            elif line.startswith('Prediction:'):
                prediction = line.replace('Prediction:', '').strip()
        
        #Store results for this row
        results = {
            'response_1': response_1,
            'response_2': response_2,
            'correct_response': correct_response,
            'prediction': prediction,
            #small mistake here for 'correct', one was a string and one was an
            #int so it would always come up False, fixed it here for visuals but
            #I actually fixed it after I got my results csvs
            'correct': int(correct_response) == int(prediction),
            'explanation': explanation,
            'temperature': temperature
        }
        all_results.append(results)
        
    df = pd.DataFrame(all_results)
    df.to_csv('results/explain_then_predict_results.csv', index=False)

#Predict_then_explain_experiment
def predict_then_explain_experiment(temperature=0):
    """
    Runs the predict-then-explain experiment on the heldout pair data, then stores
    it into a csv file.

    Args:
        temperature (float, optional): The temperature to use when prompting 
        the ChatGPT model. Defaults to 0.
    """
    data = pd.read_csv("sample_data/heldout_pair_data.csv")
    all_results = []
    
    for index, row in data.iterrows():
        response_1 = row['positive']
        response_2 = row['negative']

        prompt_outline, correct_response = prompt_outline_predict_then_explain(response_1, response_2)
        response = gpt_query(prompt_outline, temperature=temperature)
        
        #Parse response to extract prediction and explanation
        response_clean = response.strip().strip('{}')
        lines = response_clean.split('\n')
        
        explanation = ""
        prediction = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Explanation:'):
                explanation = line.replace('Explanation:', '').strip()
            elif line.startswith('Prediction:'):
                prediction = line.replace('Prediction:', '').strip()
        
        #Store results for this row
        results = {
            'response_1': response_1,
            'response_2': response_2,
            'correct_response': correct_response,
            'prediction': prediction,
            'correct': int(correct_response) == int(prediction),
            'explanation': explanation,
            'temperature': temperature
        }
        all_results.append(results)
        
    df = pd.DataFrame(all_results)
    df.to_csv('results/predict_then_explain_results.csv', index=False)
    

if __name__ == "__main__":
    explain_then_predict_experiment()
    predict_then_explain_experiment()