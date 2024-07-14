import requests
from bs4 import BeautifulSoup
import json

# Fetch the webpage content
url = "https://excelchamps.com/blog/useful-macro-codes-for-vba-newcomers/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Initialize an empty list to store macro data
macros = []

# Find all macro sections (assuming each macro is within an <h3> tag)
headings = soup.find_all('h3')

for heading in headings:
    title = heading.get_text().strip()
    description_parts = []
    code_block = []
    
    sibling = heading.find_next_sibling()

    # Collect the first description part
    while sibling and sibling.name != 'pre' and sibling.name != 'h3':
        if sibling.name == 'p':
            description_parts.append(sibling.get_text().strip())
        sibling = sibling.find_next_sibling()
    
    # Collect the code block
    if sibling and sibling.name == 'pre':
        code_lines = sibling.get_text().strip().split('\n')
        code_block.extend(code_lines)
        sibling = sibling.find_next_sibling()
    
    # Collect the second description part
    while sibling and sibling.name != 'h3':
        if sibling.name == 'p':
            description_parts.append(sibling.get_text().strip())
        sibling = sibling.find_next_sibling()
    
    description = ' '.join(description_parts)

    macros.append({
        'title': title,
        'description': description,
        'code': code_block
    })

# Save the extracted data to a JSON file
with open('vba_macros.json', 'w') as file:
    json.dump({"macros": macros}, file, indent=4)

print("Macro information saved to vba_macros.json")