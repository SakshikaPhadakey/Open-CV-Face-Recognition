import urllib.request
import re


search_keyword=input("Enter the name of youtuber : ")
html=urllib.request.urlopen("https://www.youtube.com/results?search_query="+search_keyword)

video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
print(video_ids)

index=int(input("Pick the video number which you want to play : "))

print("https://www.youtube.com/watch?v=" + video_ids[index-1])