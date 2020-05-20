import urllib3
import os
import urllib.request as request

txt_path = "/home/shinkoku/Downloads"
txt_name = "video_links.txt"
output_pth = "/home/shinkoku/materials/GTEA gaze video clips/raw_video"


def download_with_url(output_pth,url):
    list = url.split('/')
    print(url)
    video_name = list[-1][:-6]
    u = request.urlopen(url)
    data = u.read()
    print("downloading:{}".format(video_name))
    with open(os.path.join(output_pth,video_name),'wb') as f:
        f.write(data)

def download_file_from_url(txt_path,txt_name,output_pth):

    file = open(os.path.join(txt_path,txt_name))
    for r in file:
        a = r.replace("dl=0","dl=1")
        print(a)
        download_with_url(output_pth, a)


download_file_from_url(txt_path,txt_name,output_pth)