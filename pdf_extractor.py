import pdfplumber
import re
import numpy as np
from camelot import read_pdf
import os
import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from Levenshtein import distance

class PDFDatesFinderSemanticSearch:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pages_words = None
        self.tables = None
        self.paragraphs = None
        self.embedder = SentenceTransformer('msmarco-distilbert-base-v4')
        
    # Private method to find pages with tables
    def __find_pages_tables(self, len_doc):
        index_pages = []
        for i in range(len_doc):
            table = read_pdf(self.pdf_path, pages=str(i))
            if table.n > 0:
                index_pages.append(i-1)
        return index_pages

    # Private method to extract words and tables
    def __extract_mytext(self):
        tables = read_pdf(self.pdf_path, pages='all')
        with pdfplumber.open(self.pdf_path) as pdf:
            len_doc = len(pdf.pages)
            index_tables = self.__find_pages_tables(len_doc)
            words = []
            for i in range(len_doc):
                if i not in index_tables:
                    words.append(pdf.pages[i].chars)
        return words, tables

    # Method to load the PDF and extract basic information
    def load_pdf(self):
        self.pages_words, self.tables = self.__extract_mytext()
        return "PDF loaded successfully."
    
    # Private method to check if all elements in an array are equal
    def __is_array_uguali(self, array):
        if len(array) == 0:
            return True
        return all(elemento == array[0] for elemento in array)

    # Private method to parse PDF layout
    def __pdf_parser(self, words):
        dict_words = []
        temp_word = ""
        temp_font = []
        temp_height = []
        temp_size = []
        for letter in words:
            if letter['text'] == ' ' and temp_word:
                if self.__is_array_uguali(temp_font):
                    temp_font = temp_font[0]
                if self.__is_array_uguali(temp_height):
                    temp_height = temp_height[0]
                if self.__is_array_uguali(temp_size):
                    temp_size = temp_size[0]
                dict_words.append({'text': temp_word, 'font': temp_font, 'height': temp_height, 'size': temp_size})
                temp_word = ""
                temp_font = []
                temp_height = []
                temp_size = []
            elif letter['text'] == ' ' and temp_word == '':
                continue
            else:
                temp_word += letter['text']
                temp_font.append(letter['fontname'])
                temp_height.append(np.round(letter['height'], 4))
                temp_size.append(np.round(letter['size'], 4))
        return dict_words

    # Private method to identify paragraphs using bold words
    def __identify_paragraph_bold(self, dict_words):
        results = []
        temp_str = ""
        check_var = False
        for word in dict_words:
            p_str = ""
            b_str = ""
            if ('Bold' or 'bold') in word['font']:
                b_str += word['text']
            else:
                p_str += word['text']
            if b_str != "" and p_str == "" and check_var == False:
                temp_str += b_str + " "
            if b_str == "" and p_str != "":
                temp_str += p_str + " "
                check_var = True
            if b_str != "" and p_str == "" and check_var == True:
                results.append(temp_str.strip())
                temp_str = b_str + " "
                check_var = False
        if temp_str:
            results.append(temp_str.strip())
        return results

    # Private method to flatten nested arrays
    def __flatten_array_str(self, arr):
        flattened = []
        for i in arr:
            if isinstance(i, list):
                flattened.extend(self.__flatten_array_str(i))
            else:
                flattened.append(i)
        return flattened

    # Private method to apply paragraph identification to all pages
    def __apply_ipb_all_pages(self):
        paragraphs = []
        for i in range(len(self.pages_words)):
            dict_words = self.__pdf_parser(self.pages_words[i])
            temp_par = self.__identify_paragraph_bold(dict_words)
            paragraphs.append(temp_par)
        return self.__flatten_array_str(paragraphs)

    # Method to process the text and identify paragraphs
    def process_text(self):
        self.paragraphs = self.__apply_ipb_all_pages()
        return "Text processed successfully. Paragraphs identified."
    
        # Private method for semantic search
    def __semantic_search(self, top_k, embed_par, queries):
        dict_dates = {}
        top_k = min(top_k, len(embed_par))
        for metadata in queries.keys():
            dict_kw = {}
            for keyword in queries[metadata]:
                query_embedding = self.embedder.encode(keyword, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_embedding, embed_par)[0]
                top_results = torch.topk(cos_scores, k=top_k)
                temp_list = []
                for score, idx in zip(top_results[0], top_results[1]):
                    temp_list.append(self.paragraphs[idx])
                dict_kw[keyword] = temp_list
            dict_dates[metadata] = dict_kw
        return dict_dates

    # Private method to remove strings without years
    def __remove_strings_without_years(self, outer_dictionary):
        year_pattern = re.compile(r'\b\d{4}\b|\b[5-9][0-9]\b')
        for key, inner_dictionary in outer_dictionary.items():
            for inner_key, string_list in inner_dictionary.items():
                outer_dictionary[key][inner_key] = [s for s in string_list if year_pattern.search(s)]
        return outer_dictionary

    # Method to perform semantic search for dates
    def find_dates(self, top_k=1, dates_kw=None):
        if dates_kw is None:
            dates_kw = {
                'Insurance Validity Period': ['Start Date', 'End Date', 'Period', 'From: To:']
            }
        # Paragraphs embedding
        parag_embeddings = self.embedder.encode(self.paragraphs, convert_to_tensor=True)
        dict_dates = self.__semantic_search(top_k, parag_embeddings, dates_kw)
        dict_dates = self.__remove_strings_without_years(dict_dates)
        return dict_dates



class PDFDatesFinderSpace:
    def __init__(self, path):
        self.path = path

    def find_pages_tables(self, len_doc):
        """Check for tables on each page of the pdf.
        In order to exclude these pages from extraction with fitz library.
        """
        index_pages = []
        for i in range(len_doc):
            table = read_pdf(self.path, pages=str(i))
            if table.n > 0:
                index_pages.append(i - 1)
        return index_pages

    def extract_mytext(self):
        """Extracts tables in pdf file.
        Extracts text present on pages excluding those with tables.
        """
        tables = read_pdf(self.path, pages='all')
        with pdfplumber.open(self.path) as pdf:
            len_doc = len(pdf.pages)
            index_tables = self.find_pages_tables(len_doc)
            pages = []
            len_doc = int(len_doc*0.5)
            for i in range(len_doc):
                if i not in index_tables:
                    pages.append(pdf.pages[i].extract_text(layout=True))
        return pages, tables

    def identify_paragraphs_space(self, page, left_range=60, right_range=40):
        lines = page.split("\n")
        paragraphs = []
        current_paragraph = ""
        for line in lines:
            line_without_spaces = line.strip()
            if line_without_spaces == "":
                dates = re.findall(r'\b\d{4}\b|\b[5-9][0-9]\b', current_paragraph)
                # Check if the current paragraph contains at least two dates
                if current_paragraph != "" and len(dates) >= 2:

                    # If kw period is found we take it plus the rest of phrase ignoring all the part before 
                    match = re.search(r'\bperiod\b', current_paragraph, re.IGNORECASE)
                    if match:
                        start_pos = 0
                        end_pos = len(current_paragraph)
                    # Otherwise we use the right and left range
                    else:
                        positions = [match.start() for match in re.finditer(r'\b\d{4}\b|\b[5-9][0-9]\b', current_paragraph)]
                        start_pos = max(positions[0] - left_range, 0)
                        end_pos = min(positions[1] + right_range, len(current_paragraph))
                    current_paragraph = current_paragraph[start_pos:end_pos]
                    paragraphs.append(re.sub(' +', ' ', current_paragraph.strip()))
                current_paragraph = ""
            else:
                current_paragraph += line + " "
        
        if current_paragraph != "" and len(re.findall(r'\b\d{4}\b|\b[5-9][0-9]\b', current_paragraph)) >= 2:
            positions = [match.start() for match in re.finditer(r'\b\d{4}\b|\b[5-9][0-9]\b', current_paragraph)]
            start_pos = max(positions[0] - left_range, 0)
            end_pos = min(positions[1] + right_range, len(current_paragraph))
            current_paragraph = current_paragraph[start_pos:end_pos]
            paragraphs.append(re.sub(' +', ' ', current_paragraph.strip()))
        
        return paragraphs


class PDFDeductiblesFinder:
    def __init__(self, path):
        self.path = path

    def find_pages_tables(self, len_doc):
        """Check for tables on each page of the pdf.
        In order to exclude these pages from extraction with fitz library.
        """
        index_pages = []
        for i in range(len_doc):
            table = read_pdf(self.path, pages=str(i))
            if table.n > 0:
                index_pages.append(i - 1)
        return index_pages

    def extract_mytext(self):
        """Extracts tables in pdf file.
        Extracts text present on pages excluding those with tables.
        """
        tables = read_pdf(self.path, pages='all')
        with pdfplumber.open(self.path) as pdf:
            len_doc = len(pdf.pages)
            index_tables = self.find_pages_tables(len_doc)
            pages = []
            pages_words = []
            len_doc = int(len_doc)
            for i in range(len_doc):
                if i not in index_tables:
                    pages.append(pdf.pages[i].extract_text(layout=True))
                    pages_words.append(pdf.pages[i].chars)
        return pages, pages_words, tables
    
    def _is_array_uguali(self, array):
        if len(array) == 0:
            return True  # Un array vuoto è considerato formato da elementi uguali

        return all(elemento == array[0] for elemento in array)
    
    def _pdf_parser(self, words): 
        dict_words = []
        temp_word = ""
        temp_font = []
        temp_height = []
        temp_size = []
        for letter in words:
        
            if letter['text'] == ' ' and temp_word:

                #Check font are all equals
                if self._is_array_uguali(temp_font):
                    temp_font = temp_font[0]
            
                #Check hights are all equals
                if self._is_array_uguali(temp_height):
                    temp_height = temp_height[0]
            
                #Check sizes are all equals
                if self._is_array_uguali(temp_size):
                    temp_size = temp_size[0]

                dict_words.append({'text':temp_word, 'font':temp_font, 'height':temp_height, 'size':temp_size})
                #Reset
                temp_word = ""
                temp_font = []
                temp_height = []
                temp_size = []

            elif letter['text'] == ' ' and temp_word=='':
                continue

            else:
                temp_word += letter['text']
                temp_font.append(letter['fontname'])
                temp_height.append(np.round(letter['height'], 4))
                temp_size.append(np.round(letter['size'], 4))
        return dict_words
    
    def _remove_noise_from_pages(self, pages_with_ded, index_ded_bold):
        # Ottenere le righe da ogni pagina
        all_lines = [page.split('\n') for page in pages_with_ded]
        
        # Identificare le righe rumorose
        noisy_lines = set()
        for i, lines in enumerate(all_lines):
            for j, other_lines in enumerate(all_lines):
                if i != j:
                    for line in lines:
                        for other_line in other_lines:
                            if distance(line, other_line) <= 1:
                                noisy_lines.add(line)
        
        # Rimuovere le righe rumorose
        cleaned_pages = []
        for idx, lines in enumerate(all_lines):
            cleaned_page = '\n'.join(line for line in lines if line not in noisy_lines)
            
            # Per le pagine con "deductible" in grassetto, elimina tutto il testo prima di "deductible"
            if idx in index_ded_bold:
                match = re.search(r"(?i)deductible", cleaned_page)
                if match:
                    start_position = match.start()
                    cleaned_page = cleaned_page[start_position:]
        
            cleaned_pages.append(cleaned_page)

        cleaned_pages = "\n\n".join(cleaned_pages)
    
        return cleaned_pages

    def identify_deductibles_pages(self, pages, pages_words):
        words = []
        pages_with_ded = []
        index_ded = []
        index_ded_bold = []
        for i in range(len(pages_words)):
            words.append(self._pdf_parser(pages_words[i]))

        index_bold = 0
        for num_page in range(len(words)):
            for word in words[num_page]:
                if ('Bold' or 'bold') in word['font'] and 'deductible' in word['text'].lower():
                    index_ded.append(num_page)
                    index_ded_bold.append(index_bold)
                    index_bold +=1 
                    if(num_page + 1 < len(pages_words)):
                        index_ded.append(num_page + 1)
                        index_bold += 1
        
        for i in range(len(pages)):
            if i in index_ded:
                pages_with_ded.append(pages[i])

        pages_with_ded = self._remove_noise_from_pages(pages_with_ded, index_ded_bold)

        return pages_with_ded