import os
import re
import time
import PyPDF2
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# 0. NLTK SETUP & GLOBALS
# ==========================================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
academic_stopwords = {'shall', 'may', 'one', 'two', 'also', 'per', 'hence', 'thus', 'within', 'page'}
stop_words.update(academic_stopwords)

lemmatizer = WordNetLemmatizer()

# Terms we absolutely do not want the length-filter to delete
PROTECTED_TERMS = {'ug', 'pg', 'phd', 'ai', 'ds', 'ee', 'me', 'cse', 'exam'}


clean_docs = []

# ==========================================
# 1. SET YOUR DATA SOURCES HERE
# ==========================================
# LOCAL FILES
my_text_file = ""

my_pdf_files = [
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/2. B.Tech EE_updated 15.3.2022.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/2. B.Tech EE_updated 15.3.2022.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/3. B.Tech CSE_09102020.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/5. B.Tech AIDS.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/Academic_Regulations_Final_03_09_2019.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/CSE-Courses-Details.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/Curriculum-BTech-CSE.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/details-of-the-UG-program-638769639494205696.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic/UG-BTech-EE-Curriculum-638763398808246613.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic_reg/4_Regulation_PG_2022-onwards_20022023.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic_reg/6_2024-04-17-661f605b54457-1713332315.pdf",
    "/home/kanishk/Desktop/NLUAssign2/problem1/pdfs_academic_reg/BTech_Regulations_Latest_21July2018.pdf",
]

# WEB CRAWLER CONFIGURATION
ENABLE_WEB_CRAWLER = True  

CRAWLER_CONFIG = {
    "enabled": ENABLE_WEB_CRAWLER,
    "start_urls": [
        "https://iitj.ac.in/",
        "https://iitj.ac.in/m/Index/main-institute?lg=en",
        "https://iitj.ac.in/m/Index/main-departments?lg=en",
        "https://iitj.ac.in/main/en/faculty-members",
        "https://iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
        "https://iitj.ac.in/Master-of-Technology/en/Master-of-Technology",
        "https://iitj.ac.in/computer-science-engineering/",
        "https://iitj.ac.in/chemical-engineering/",
        "https://iitj.ac.in/electrical-engineering/",
        "https://iitj.ac.in/mechanical-engineering/",
        "https://iitj.ac.in/es/en/engineering-science",
        "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/school-of-artificial-intelligence-and-data-science",
        "https://iitj.ac.in/computer-science-engineering/en/projects",
        "https://iitj.irins.org/faculty/index/Department+of+Chemical+Engineering",
        "https://iitj.ac.in/computer-science-engineering/en/research-area-labs",
        "https://iitj.irins.org/faculty/index/Department+of+Electrical+Engineering",
        "https://iitj.irins.org/faculty/index/Department+of+Bioscience+and+Bioengineering",
        "https://iitj.irins.org/faculty/index/Department+of+Metallurgical+and+Materials+Engineering"
    ],
    "allowed_domain": "iitj.ac.in",
    "max_pages": 700, 
    "delay_between_requests": 0.5 
}

output_file = "corpus.txt"


# ==========================================
# 2. YOUR CUSTOM CLEANING FUNCTION
# ==========================================
def clean_document(text):
    # 1. Lowercase and scrub weird hidden PDF unicode characters
    text = text.lower().encode('ascii', 'ignore').decode('utf-8')

    # 2. Normalize Academic Degrees BEFORE removing punctuation
    text = re.sub(r'\bb\.?\s*tech\b', 'btech', text)
    text = re.sub(r'\bm\.?\s*tech\b', 'mtech', text)
    text = re.sub(r'\bph\.?\s*d\.?\b', 'phd', text)
    text = re.sub(r'\bu\.?\s*g\.?\b', 'ug', text) 
    text = re.sub(r'\bp\.?\s*g\.?\b', 'pg', text) 
    
    # 3. Removal of obvious non-textual content 
    text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', ' ', text)

    # 4. Fix hyphenated words from PDF line breaks
    text = re.sub(r'-\s+', '', text)

    # 5. Remove numbers and all punctuation (keeps only a-z and spaces, plus underscores temporarily)
    text = re.sub(r'[^\w\s]', ' ', text) 

    words = text.split()
    cleaned_words = []
    
    for w in words:
        # Check for underscores
        if w.startswith('_') or w.endswith('_'):
            continue

        # Check for numbers
        if any(char.isdigit() for char in w):
            continue

        # Check for stopwords
        if w in stop_words:
            continue
            
        # Check for short words, but bypass if it is in PROTECTED_TERMS
        if len(w) <= 2 and w not in PROTECTED_TERMS:
            continue

        # 6. Lemmatization (Plurals to Singulars)
        w_lemmatized = lemmatizer.lemmatize(w)
        
        cleaned_words.append(w_lemmatized)

    return cleaned_words


# ==========================================
# 3. PROCESSING FUNCTIONS
# ==========================================
def process_txt(filepath):
    """Reads raw text line-by-line. 1 Line from text file = 1 Document."""
    if not os.path.exists(filepath):
        print(f"Warning: Text file '{filepath}' not found.")
        return
    with open(filepath, 'r', encoding='utf-8', errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            tokens = clean_document(line)
            if len(tokens) >= 3:
                clean_docs.append(tokens)


def process_pdf(filepath):
    """Reads PDFs. 1 ENTIRE PDF FILE = 1 Document."""
    if not os.path.exists(filepath):
        print(f"Warning: PDF file '{filepath}' not found.")
        return
    try:
        full_pdf_text = ""
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            # Accumulate text from all pages first
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_pdf_text += page_text + " "
            
            # Clean and append the ENTIRE pdf as one single document (line)
            tokens = clean_document(full_pdf_text)
            if len(tokens) >= 3:
                clean_docs.append(tokens)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")


# ==========================================
# 4. WEB SCRAPING FUNCTIONS (1 URL = 1 Doc)
# ==========================================
def is_valid_url(url, allowed_domain):
    parsed = urlparse(url)
    if allowed_domain not in parsed.netloc:
        return False
    if url.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.zip', '.rar', '.gif')):
        return False
    return True

def extract_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)

        if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove junk tags
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()

        return soup.get_text(separator=' ')
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return ""

def crawl_website(start_urls, allowed_domain, max_pages, delay=0.5):
    print(f"Starting web crawler for {allowed_domain}...")
    visited_urls = set()
    urls_to_visit = start_urls.copy()

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls: continue
        visited_urls.add(current_url)

        if current_url.lower().endswith('.pdf'):
            print(f"[SKIPPED PDF] {current_url}")
            continue

        try:
            print(f"Scraping ({len(visited_urls)}/{max_pages}): {current_url}")
            time.sleep(delay)

            # 1 URL = 1 Document
            raw_text = extract_from_url(current_url)
            if raw_text:
                tokens = clean_document(raw_text)
                if len(tokens) >= 3:
                    clean_docs.append(tokens)

            # Extract and queue new links
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(current_url, timeout=10, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    full_url = urljoin(current_url, href).split('#')[0]
                    if is_valid_url(full_url, allowed_domain) and full_url not in visited_urls and full_url not in urls_to_visit:
                        urls_to_visit.append(full_url)

        except Exception as e:
            print(f"[ERROR] on {current_url}: {e}")
            
    print(f"\nCrawl complete! Pages scraped: {len(visited_urls)}")


# ==========================================
# 5. EXECUTE PIPELINE AND SAVE
# ==========================================
print("=" * 50)
print("CORPUS PREPARATION PIPELINE")
print("=" * 50)

# 1. LOCAL TEXT FILES
print("\n[STEP 1] Processing local text files...")
process_txt(my_text_file)

# 2. LOCAL PDF FILES
print("\n[STEP 2] Processing local PDF files (1 File = 1 Document)...")
for pdf in my_pdf_files:
    process_pdf(pdf)

# 3. WEB CRAWLER
if CRAWLER_CONFIG["enabled"] and CRAWLER_CONFIG["start_urls"]:
    print("\n[STEP 3] Running web crawler (URL by URL)...")
    crawl_website(
        CRAWLER_CONFIG["start_urls"],
        CRAWLER_CONFIG["allowed_domain"],
        CRAWLER_CONFIG["max_pages"],
        CRAWLER_CONFIG["delay_between_requests"]
    )
else:
    print("\n[STEP 3] Web crawler disabled.")

# 4. Save to the final file
print(f"\n[STEP 4] Saving clean corpus to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for doc in clean_docs:
        f.write(" ".join(doc) + "\n")

print("\n" + "=" * 50)
print(f"SUCCESS! Cleaned corpus saved as '{output_file}'")
print(f"Total valid documents extracted: {len(clean_docs)}")
print("=" * 50)