from pdf_extractor import PDFDatesFinderSpace   
from pdf_extractor import PDFDeductiblesFinder
from postprocessing_functions import post_process_response_dates
from postprocessing_functions import find_dates_regex
import os
import re
from tqdm import tqdm
import json
import openai
import cv2

# OpenAI model settings
API_KEY = "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"
openai.api_key = API_KEY
model_id = 'gpt-3.5-turbo'

# ChatGPT 
def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation
    )
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation


# Insurances folder path
insurances_folder_path = "/mnt/c/Users/e.ceglia/Desktop/EMA/Tesi/Eng Insurances"

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# DATES EXTRACTION using GPT-3.5 turbo
extracted_dates = {}

print('Extracting Dates...')
for root, dirs, files in os.walk(insurances_folder_path):
    for file_name in tqdm(files):

        full_file_path = os.path.join(root, file_name)

        # Dates extraction
        extractor_dates = PDFDatesFinderSpace(full_file_path)
        pages, tables = extractor_dates.extract_mytext()
        paragraphs = [extractor_dates.identify_paragraphs_space(page) for page in pages]

        # keywords filter 1th level
        check_kw = False
        for phrase in paragraphs:
          for sentence in phrase:
            if re.search(r'\bperiod\b', sentence, re.IGNORECASE):
              check_kw = True

        if check_kw:
          # Use a list comprehension with regex to keep only the phrases that contain the word "period" (case-insensitive)
          paragraphs = [sublist for sublist in paragraphs if any(re.search(r'\bperiod\b', phrase, re.IGNORECASE) for phrase in sublist)]

          # Now, further filter each sublist to keep only the phrases that contain the word "period"
          paragraphs = [[phrase for phrase in sublist if re.search(r'\bperiod\b', phrase, re.IGNORECASE)] for sublist in paragraphs]


        responses = []
        for phrase in paragraphs:
          if len(phrase):
            for sentence in phrase:
                conversation = []
                sentcence = sentence.replace(',', '')
                prompt = "Find start date and end date from the following sentence: " + sentence
                conversation.append({'role': 'user', 'content': prompt})
                conversation = ChatGPT_conversation(conversation)
                response = post_process_response_dates(conversation[-1]['content'].strip())

            if response:
                responses.append(response)
                #print(response)

        if len(responses)==0:
          # Parse with regex for prompt fails
          for phrase in paragraphs:
            if len(phrase):
              if len(phrase)>1:
                phrase = [' '.join(phrase)]
              for sentence in phrase:
                  sentence = sentence.replace(',','')
                  response = find_dates_regex(sentence)
                  responses.append(response)

        if responses:
          print(f'DATE {file_name} EXTRACTED!')
          extracted_dates[str(file_name)+"_dates"] = responses
        else:
          print(f'DATE {file_name} NOT FOUND!')


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# DEDUCTIBLES EXTRACTION using GPT-3.5 turbo

deductibles = {}
extracted_deductibles = {}

print('Extracting Deductibles...')
for root, dirs, files in os.walk(insurances_folder_path):
    for file_name in tqdm(files):
        conversation = []
        full_file_path = os.path.join(root, file_name)

        extractor_deductibles = PDFDeductiblesFinder(full_file_path)
        pages, pages_words, tables = extractor_deductibles.extract_mytext()
        pages_with_ded = extractor_deductibles.identify_deductibles_pages(pages, pages_words)
        deductibles[file_name] = pages_with_ded

        prompt = 'User:' + " Extract only information about DEDUCTIBLES from the following text: " + deductibles[file_name]
        conversation.append({'role': 'user', 'content': prompt})
        conversation = ChatGPT_conversation(conversation)
        extracted_deductibles[file_name + "_deductibles"] = [{'Deductible': conversation[-1]['content'].strip()}]

        if len(extracted_deductibles[file_name + "_deductibles"]):
           print(f'DEDUCTIBLES {file_name} EXTRACTED')
        else:
           print(f'DEDUCTIBLES {file_name} NOT FOUND!')



## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Save output as .json file

def Merge(dict_1, dict_2):
	result = dict_1 | dict_2
	return result

final_dict = Merge(extracted_dates, extracted_deductibles)

output_path = "/mnt/c/Users/e.ceglia/Desktop/extractions.json"

# Scrive il dizionario in un file JSON
with open(output_path, 'w') as file:
    json.dump(final_dict, file)

print("File JSON salvato con successo!")