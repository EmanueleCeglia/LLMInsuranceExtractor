import re

# Post processing function for DATES

def post_process_response_dates(text):
    start_date_pattern = r"Start date:\s*([0-9]+[a-z]*\s+[A-Z][a-z]+\s+[0-9]{4})"
    end_date_pattern = r"End date:\s*([0-9]+[a-z]*\s+[A-Z][a-z]+\s+[0-9]{4})"

    start_date_match = re.search(start_date_pattern, text)
    end_date_match = re.search(end_date_pattern, text)

    result = {}
    if start_date_match and end_date_match:
        result['Start date'] = start_date_match.group(1)
        result['End date'] = end_date_match.group(1)
    else:
        return None

    return result

# second level filter with regex if gpt-3.5 cannot find dates
def find_dates_regex(string):
    pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{1,2} [A-Za-z]+ \d{4}\b|\b\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+ \d{4}\b|\b[A-Za-z]+ \d{1,2}(?:st|nd|rd|th)? \d{4}\b'
    dates = re.findall(pattern, string)
    if len(dates) >= 2:
        return f"Start date: {dates[0]} End date: {dates[1]}"
    elif len(dates) == 1:
        return f"Start date: {dates[0]} End date: None"
    else:
        return "No dates found"