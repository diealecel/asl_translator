import requests
from bs4 import BeautifulSoup

# Constants.
ARCHIVE_URL = 'https://engineering.purdue.edu/asldata/ASL-RVL-database/Videos/'

def get_video_links(archive_url):
    # Open connection with |archive_url|.
    response = requests.get(archive_url)
    soup = BeautifulSoup(response.content, 'html5lib')
    
    links = soup.findAll('a')
    video_links = [ archive_url + link['href'] for link in links if link['href'].endswith('avi') ]

    return video_links


if __name__ == '__main__':
    video_links = get_video_links(ARCHIVE_URL)
