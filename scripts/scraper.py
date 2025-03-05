import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


#URL of Allsides Bias Ratings page
URL  = 'https://www.allsides.com/media-bias/media-bias-ratings'

#Use headers to prevent Blocking
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36", 
    "Accept-Language": "en-US, en;q=0.9",
    "Referer": "https://google.com/",
    "DNT": "1", # Do Not Track Request Header
    "Connection": "keep-alive",
}

# Use a session to maintain cookies
session = requests.Session()
session.headers.update(headers)

#Fetch the page Content
time.sleep(3) # Sleep for 3 seconds to avoid being blocked
response = session.get(URL)

#Check if request was successful
if response.status_code != 200:
    print(f"Failed to fetch data. Status code: {response.status_code} ")
else:
    soup = BeautifulSoup(response.text, 'html.parser')

    #Initiliaze data
    data = []

    # Find bias table
    table = soup.find("table", {"class": "views-table"})
    if table:
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td") #Skip header row
            if len(cols) >= 2:
                media_source = cols[0].get_text(strip=True)
                bias = cols[1].get_text(strip=True)
                data.append({"source": media_source, "bias": bias})

    # Save to CSV if data is found
    if data:
        df = pd.DataFrame(data)
        df.to_csv("data/allsides_bias_data.csv", index=False)
        print("Data saved to data/allsides_bias_data.csv")
    else:
        print("No data scraped")
        