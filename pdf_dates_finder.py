import pdfplumber
import re
import numpy as np
from camelot import read_pdf
from sentence_transformers import SentenceTransformer, util
import torch

class PDFDatesFinder:
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