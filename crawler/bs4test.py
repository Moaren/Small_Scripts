from bs4 import BeautifulSoup
import requests

source = requests.get("http://coreyms.com").text

soup = BeautifulSoup(source,'lxml')

# print(soup.prettify())

for article in soup.find_all('article'):
    headline = article.h2.a.text
    print(headline)

    summary = article.find('div',class_ = 'entry-content').p.text
    print(summary)

    vid_src = article.find('iframe', class_ = 'youtube-player')['src']


    vid_id = vid_src.split('/')[4]
    vid_id = vid_id.split('?version')[0]
    # print(vid_id)

    yt_link = f'http://youtube.com/watch?v={vid_id}'
    print(yt_link)

    print('')