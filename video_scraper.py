# -*- coding: utf-8 -*-
# Encoding above is needed for print_progress().

from bs4 import BeautifulSoup   # For accessing the links in the URL.
from mechanize import Browser   # For accessing the URL and providing login credentials.
from base64 import b64encode    # For providing login credentials appropriately.
from sys import stdout          # For print_progress().

# NOTE: the |ARCHIVE_URL| requests a valid username and password. This means that
#       |ARCHIVE_URL| is serving a dual purpose: it serves as a source to download
#       the required videos, and it serves as a point of communication for providing
#       login information.

ARCHIVE_URL = 'https://engineering.purdue.edu/asldata/ASL-RVL-database/Videos/'
LOGIN_USER = 'celis_0168'
LOGIN_PASS = 'c2l3s'
VIDEO_FILE_TYPE = 'avi'

URL_DELIMITER = '/'
SOUP_PARSER = 'html5lib'

SAVE_DIR = 'videos/'

# Prints a progress bar to inform user of work being done.
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 2, bar_length = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    # Print new line when complete
    if iteration == total:
        stdout.write('\n')
    stdout.flush()


# Returns file name at |url|.
def get_file_name(url):
    return url.split(URL_DELIMITER)[-1]


# Downloads all videos pointed to by links in |video_links| using |br|.
def download_videos(br, video_links):
    num_videos = len(video_links)
    print_progress(0, num_videos)

    for i, link in enumerate(video_links):
        file_name = get_file_name(link)
        progress_msg = 'Downloading ' + file_name + '...'

        print_progress(i, num_videos, progress_msg)

        br.open(link)
        data = br.response().read()

        relative_save_path = SAVE_DIR + file_name
        with open(relative_save_path, 'wb') as file: file.write(data)

        if i == num_videos - 1: print_progress(i + 1, num_videos, 'Finished downloading!')


# Returns the video links in |ARCHIVE_URL| using |br|.
def get_video_links(br):
    br.open(ARCHIVE_URL)
    data = br.response().read()

    soup = BeautifulSoup(data, SOUP_PARSER)
    links = soup.findAll('a')

    video_links = [ ARCHIVE_URL + link['href'] for link in links if link['href'].endswith(VIDEO_FILE_TYPE) ]
    return video_links


# Provides |br| login credentials |LOGIN_USER| and |LOGIN_PASS|.
def provide_login_info(br):
    login_str = '%s:%s' % (LOGIN_USER, LOGIN_PASS)
    login = b64encode(login_str)

    login_hdr = 'Authorization', 'Basic %s' % login
    br.addheaders.append(login_hdr)


if __name__ == '__main__':
    br = Browser()
    provide_login_info(br)

    video_links = get_video_links(br)
    download_videos(br, video_links)
