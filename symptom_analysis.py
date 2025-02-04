import openai

open_api_key = "your key"
def analyze_symptom(symptoms_list):
    # Join the list of symptoms into a single string if it's a list
    if isinstance(symptoms_list, list):
        symptom_str = ', '.join(symptoms_list)
    else:
        symptom_str = symptoms_list  # In case it's already a string

    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0125:personal::A8F210Wk",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that diagnoses symptoms."},
            {"role": "user", "content": symptom_str}  # Pass the string of symptoms here
        ],
        max_tokens=200,
        temperature=0  # Lower temperature to make the response deterministic
    )
    return response['choices'][0]['message']['content']





