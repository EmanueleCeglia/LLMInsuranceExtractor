from pdf_extractor import PDFDatesFinderSpace   
from pdf_extractor import PDFDeductiblesFinder
from pdf_extractor import PDFSublimitsFinder
from postprocessing_functions import post_process_response_dates
from postprocessing_functions import find_dates_regex
import os
import re
from tqdm import tqdm
import json
import openai
import cv2

# OpenAI model settings
API_KEY = "#######"
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
insurances_folder_path = "./Insurances"
output_dictionary = {}

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Inizialize output dictionary
for root, dirs, files in os.walk(insurances_folder_path):
    for file_name in files:
       output_dictionary[file_name] = {}


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
          output_dictionary[file_name]['Insurance Period'] = response
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

        if len([{'Deductible': conversation[-1]['content'].strip()}]):
           print(f'DEDUCTIBLES {file_name} EXTRACTED')
           output_dictionary[file_name]['Deductibles'] = conversation[-1]['content'].strip()
        else:
           print(f'DEDUCTIBLES {file_name} NOT FOUND!')


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SUBLIMITS EXTRACTION using GPT-3.5 turbo

sublimits = {}
extracted_sublimits = {}

print('Extracting Sublimits...')
for root, dirs, files in os.walk(insurances_folder_path):
    for file_name in tqdm(files):
        conversation = []
        full_file_path = os.path.join(root, file_name)

        extractor_sublimits = PDFSublimitsFinder(full_file_path)
        pages, pages_words, tables = extractor_sublimits.extract_mytext()
        pages_with_sub, sublimit_kw_found = extractor_sublimits.identify_sublimits_pages(pages, pages_words)
        sublimits[file_name] = pages_with_sub

        if len(sublimit_kw_found):
          prompt = 'User:' + f" Extract only information about {sublimit_kw_found[0]} from the following text: " + sublimits[file_name]
          conversation.append({'role': 'user', 'content': prompt})
          conversation = ChatGPT_conversation(conversation)

          if len([{'Sublimits': conversation[-1]['content'].strip()}]):
            print(f'SUBLIMITS {file_name} EXTRACTED')
            output_dictionary[file_name]['Sublimits'] = conversation[-1]['content'].strip()
        
        else:
           print(f'SUBLIMITS {file_name} NOT FOUND!')
        
  

## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Save output as .json file

output_path = "#####"

# Scrive il dizionario in un file JSON
with open(output_path, 'w') as file:
    json.dump(output_dictionary, file)

print("File JSON salvato con successo!")