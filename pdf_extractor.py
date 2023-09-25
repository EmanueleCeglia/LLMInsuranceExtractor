import pdfplumber
import re
import numpy as np
from camelot import read_pdf
import os
import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from Levenshtein import distance


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

    def _find_pages_tables(self, len_doc):
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
            index_tables = self._find_pages_tables(len_doc)
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
        
        # Remove duplicated pages in index_ded
        index_ded = [x for i, x in enumerate(index_ded) if index_ded.index(x) == i]

        for i in range(len(pages)):
            if i in index_ded:
                pages_with_ded.append(pages[i])

        pages_with_ded = self._remove_noise_from_pages(pages_with_ded, index_ded_bold)

        return pages_with_ded
    


class PDFSublimitsFinder:
    def __init__(self, path):
        self.path = path

    def _find_pages_tables(self, len_doc):
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
            index_tables = self._find_pages_tables(len_doc)
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
    
    def _remove_noise_from_pages(self, pages_with_sub, index_sub_bold, sublimit_kw_found):
        
        # Get rows of each page
        all_lines = [page.split('\n') for page in pages_with_sub]
        
        # Identify noisy rows
        noisy_lines = set()
        for i, lines in enumerate(all_lines):
            for j, other_lines in enumerate(all_lines):
                if i != j:
                    for line in lines:
                        for other_line in other_lines:
                            if distance(line, other_line) <= 1:
                                noisy_lines.add(line)
        
        # Remove noisy rows
        cleaned_pages = []
        for idx, lines in enumerate(all_lines):
            cleaned_page = '\n'.join(line for line in lines if line not in noisy_lines)
            
            # Delete al text before the keyword related to sublimit identified
            if idx in index_sub_bold:
                if len(sublimit_kw_found):
                    match = re.search(r"(?i)"+f"{sublimit_kw_found[0]}", cleaned_page)
                if match:
                    start_position = match.start()
                    cleaned_page = cleaned_page[start_position:]
        
            cleaned_pages.append(cleaned_page)

        cleaned_pages = "\n\n".join(cleaned_pages)
    
        return cleaned_pages

    def identify_sublimits_pages(self, pages, pages_words):
        words = []
        pages_with_sub = []
        index_sub = []
        index_sub_bold = []
        sublimits_single_kw = ['sublimit','sub-limit']
        sublimits_multiple_kw = ['sub limit','inner limit']
        for i in range(len(pages_words)):
            words.append(self._pdf_parser(pages_words[i]))

        index_bold = 0
        sublimit_kw_found = []
        for num_page in range(len(words)):
            for i in range(len(words[num_page])-1):
                word = words[num_page][i]
                next_word = words[num_page][i+1]
                if ('Bold' or 'bold') in word['font']:
                    single_word = word['text'].lower()
                    couple_word = word['text'].lower() + " " + next_word['text'].lower()
                    for single_kw in sublimits_single_kw:
                        if single_kw in single_word.replace(':',''):
                            index_sub.append(num_page)
                            index_sub_bold.append(index_bold)
                            index_bold +=1 
                            if(num_page + 1 < len(pages_words)):
                                index_sub.append(num_page + 1)
                                index_bold += 1

                            sublimit_kw_found.append(single_kw)
                    
                    for multiple_kw in sublimits_multiple_kw:
                        if multiple_kw in couple_word.replace(':',''):
                            index_sub.append(num_page)
                            index_sub_bold.append(index_bold)
                            index_bold +=1 
                            if(num_page + 1 < len(pages_words)):
                                index_sub.append(num_page + 1)
                                index_bold += 1

                            sublimit_kw_found.append(multiple_kw)
        
        # Remove duplicated pages in index_sub and sublimits_kw_found
        index_sub = [x for i, x in enumerate(index_sub) if index_sub.index(x) == i]
        if len(sublimit_kw_found):
            sublimit_kw_found = [x for i, x in enumerate(sublimit_kw_found) if sublimit_kw_found.index(x) == i]
        
        for i in range(len(pages)):
            if i in index_sub:
                pages_with_sub.append(pages[i])

        pages_with_sub = self._remove_noise_from_pages(pages_with_sub, index_sub_bold, sublimit_kw_found)

        return pages_with_sub, sublimit_kw_found
    

