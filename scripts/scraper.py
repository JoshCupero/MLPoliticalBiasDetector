import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from fake_useragent import UserAgent
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Set up Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in background
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.set_page_load_timeout(30) 

# Set up Logging
logging.basicConfig(filename="data/logs/scraping_log.txt", level=logging.ERROR, format="%(asctime)s - %(message)s")


# Fake User-Agent to Avoid Getting Blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# Define News Sources
STATIC_SOURCES = {
    "BBC": "https://www.bbc.com/news/politics",
    "New York Times": "https://www.nytimes.com/section/politics",
    "NPR": "https://www.npr.org/sections/politics/",
    "Washington Times": "https://www.washingtontimes.com/news/politics/",
    "Newsmax": "https://www.newsmax.com/Politics/",
    "The Guardian": "https://www.theguardian.com/us-news/us-politics",
    "Associated Press": "https://apnews.com/hub/politics",
    "FoxNews": "https://www.foxnews.com/politics",
    "The Daily Wire": "https://www.dailywire.com/",
    "Washington Post": "https://www.washingtonpost.com/politics/",
    "The Hill": "https://thehill.com/homenews/campaign",
    "Breitbart": "https://www.breitbart.com/politics/",
    "USA Today": "https://www.usatoday.com/news/politics/",
    "Reuters": "https://www.reuters.com/world/us/",
    "The Atlantic": "https://www.theatlantic.com/politics/",
    "National Review": "https://www.nationalreview.com/politics-policy/",
    "Vox": "https://www.vox.com/policy-and-politics",
    "The Intercept": "https://theintercept.com/politics/",
    "The Federalist": "https://thefederalist.com/category/politics/",
    "The Week": "https://theweek.com/politics",
    "The American Conservative": "https://www.theamericanconservative.com/articles/",
    "Townhall": "https://townhall.com",
    "Reason": "https://reason.com",
    "The Bulwark": "https://thebulwark.com",
    "The American Prospect": "https://prospect.org/politics/",
    "The Nation": "https://www.thenation.com/subject/politics/",
    "The New Republic": "https://newrepublic.com/politics",
    "Mother Jones": "https://www.motherjones.com/politics/",
    "The American": "https://www.americanmagazine.org/politics",
    "The Spectator": "https://spectator.org/politics/",
    "The Blaze": "https://www.theblaze.com/politics",
    "The Epoch Times": "https://www.theepochtimes.com/politics",
    "The Christian Science Monitor": "https://www.csmonitor.com/USA/Politics",
    "The Washington Examiner": "https://www.washingtonexaminer.com/politics",
    "The New York Post": "https://nypost.com/politics/",
    "The Christian Post": "https://www.christianpost.com/news/politics/",
    "The Daily Caller": "https://dailycaller.com/section/politics/",
    "HuffPost": "https://www.huffpost.com/news/politics",
    "PBS NewsHour": "https://www.pbs.org/newshour/politics",
    "Al Jazeera": "https://www.aljazeera.com/us-canada/",
    "Bloomberg": "https://www.bloomberg.com/politics",
    "Slate": "https://slate.com/news-and-politics",
    "The Economist": "https://www.economist.com/",
    "The Daily Signal": "https://www.dailysignal.com/",
    "The American Spectator": "https://spectator.org/politics/",
    "American Affairs Journal": "https://americanaffairsjournal.org/",
    "The American Thinker": "https://www.americanthinker.com/",
    "Conservative Review": "https://www.conservativereview.com/"
}

DYNAMIC_SOURCES = {
    "CNN": "https://www.cnn.com/politics",
    "MSNBC": "https://www.msnbc.com/politics/"
}

MAX_RETRIES = 3  # Number of retries before giving up
ua = UserAgent()  # Random User-Agent generator
# Function to Get Article Links (Static Sites)

MAX_RETRIES = 3  # Number of retries before giving up
ua = UserAgent()  # Random User-Agent generator

INVALID_URL_KEYWORDS = [
    "javascript", "#", "subscribe", "gift-subscription", "customer-service",
    "/watch-live-news", "/news/", "/sport/", "/business/", "/politics/news/", 
    "shopnpr.org", "help.npr.org", "contact", "signin", "preference", "login",
    "tos", "privacy", "newsletters", "accessibility", "crossword", "sign-in",
    "pricing", "contributers", "membership", "account", "icon", "freetrial", 
    "sign-out"
]
def get_article_links_static(source_name, url):
    """Fetch article links from a static news source with retries and user-agent rotation."""
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"User-Agent": ua.random}  # Randomize User-Agent
            response = requests.get(url, headers=headers, timeout=10)  # 10-second timeout
            response.raise_for_status()  # Raise HTTP errors
            break  # Exit loop if request succeeds
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {url}: {e}")
            logging.error(f"Failed to scrape {url}: {e}")  # Log the error
            time.sleep(random.uniform(5, 15))  # Wait before retrying
    else:
        print(f"❌ Skipping {url} after {MAX_RETRIES} failed attempts.")
        return []  # Return empty list if all retries fail

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for link in soup.select("a[href]"):
        full_link = link["href"]
        full_link = urljoin(url, full_link)  # Ensure absolute URLs

        # **Skip invalid URLs**
        if any(keyword in full_link for keyword in INVALID_URL_KEYWORDS):
            continue

        if full_link not in links:
            links.append(full_link)

    print(f"✅ Found {len(links)} valid articles on {source_name}")
    return links


# Function to Scrape Articles (Static Sites)
def scrape_article_static(source_name, url):
    """Scrape article content with retries and user-agent rotation."""
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"User-Agent": ua.random}  # Randomize User-Agent
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise HTTP errors
            break  # Exit loop if request succeeds
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {url}: {e}")
            logging.error(f"Failed to scrape {url}: {e}")  # Log the error
            time.sleep(random.uniform(5, 15))  # Wait before retrying
    else:
        print(f"❌ Skipping {url} after {MAX_RETRIES} failed attempts.")
        return None  # Skip if all attempts fail

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1").text.strip() if soup.find("h1") else "No Title"
    paragraphs = soup.find_all("p")
    article_text = " ".join([p.text for p in paragraphs])

    return {"source": source_name, "title": title, "url": url, "content": article_text}

# Function to Get Article Links (Dynamic Sites with Selenium)
def get_article_links_dynamic(source_name, url):
    try:
        driver.get(url)
        time.sleep(5)  # Allow JavaScript to load
        
        links = []
        elements = driver.find_elements(By.XPATH, "//a[@href]")
        for elem in elements:
            full_link = elem.get_attribute("href")
            if full_link and "politics" in full_link:
                links.append(full_link)
        
        print(f"✅ Found {len(links)} articles on {source_name}")
        return links
    except Exception as e:
        print(f"⚠️ Failed to load {url}: {e}")
        logging.error(f"Failed to scrape {url}: {e}")  # Log the error
        return []  # Return empty list instead of hanging

# Function to Scrape Articles (Dynamic Sites with Selenium)
def scrape_article_dynamic(source_name, url):
    try:
        driver.get(url)
        time.sleep(5)  # Allow JavaScript to load
        
        title_elem = driver.find_element(By.TAG_NAME, "h1")
        paragraphs = driver.find_elements(By.TAG_NAME, "p")
        
        title = title_elem.text.strip() if title_elem else "No Title"
        article_text = " ".join([p.text for p in paragraphs])
        
        return {"source": source_name, "title": title, "url": url, "content": article_text}
    except Exception as e:
        print(f"⚠️ Failed to scrape {url}: {e}")
        logging.error(f"Failed to scrape {url}: {e}")  # Log the error
        return None

# Scrape Static News Sites
data_static = []
for source, url in STATIC_SOURCES.items():
    article_links = get_article_links_static(source, url)
    for link in article_links[:10]:  # Limit to 10 articles per source for testing
        print(f"Scraping: {link}")
        article = scrape_article_static(source, link)
        if article:
            data_static.append(article)
        if "washingtontimes" in url:
            time.sleep(random.uniform(7, 12))  # Longer delay for Washington Times
        else:
            time.sleep(random.uniform(3, 6))  # Regular delay


# Save Static Data
df_static = pd.DataFrame(data_static)
df_static.to_csv("data/raw_articles/scraped_articles_static.csv", index=False)
print("✅ Static news articles saved to data/scraped_articles_static.csv")

# Scrape Dynamic News Sites
data_dynamic = []
for source, url in DYNAMIC_SOURCES.items():
    article_links = get_article_links_dynamic(source, url)
    for link in article_links[:10]:  # Limit to 10 per source
        print(f"Scraping: {link}")
        article = scrape_article_dynamic(source, link)
        if article:
            data_dynamic.append(article)
        time.sleep(3)  # Avoid detection

# Save Dynamic Data
df_dynamic = pd.DataFrame(data_dynamic)
df_dynamic.to_csv("data/raw_articles/scraped_articles_dynamic.csv", index=False)
print("✅ Dynamic news articles saved to data//scraped_articles_dynamic.csv")

# Close Selenium
driver.quit()

# Merge Static & Dynamic Articles
df_articles = pd.concat([df_static, df_dynamic])

# Save Merged Dataset
df_articles.to_csv("data/raw_articles/scraped_articles.csv", index=False)
print("✅ Merged articles saved to data/scraped_articles.csv")
