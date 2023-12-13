# Fetch the links related to some specific youtube channel
import timeit
from googleapiclient.discovery import build

api_key = 'AIzaSyBhbYzOh_B3snsiBlCEwI4DdUZbKJVHass'

youtube = build('youtube', 'v3', developerKey=api_key)

# UCA19mAJURyYHbJzhfpqhpCA: action lab shorts
# UCsXVk37bltHxD1rDPwtNM8Q: kurzgesagt in a nutshell
# UCRcgy6GzDeccI7dkbbBna3Q: Lemmino
UserInput = 'UCA19mAJURyYHbJzhfpqhpCA'

start_time = timeit.default_timer()

def fetchVideoUrl(channelId):
  try:
    # fetches the channel's info
    channelRes = youtube.channels().list(
      part='contentDetails', id=channelId
    ).execute()
    
    # uses the channel info and then fetches the playlist 
    if 'items' in channelRes and channelRes['items']:
      playlistId = channelRes['items'][0]['contentDetails']['relatedPlaylists']['uploads']

      # uses that playlist info and then fetches the links of uploaded videos
      playlistResult = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlistId,
        maxResults=50
      ).execute()
      
      # returning an array of videoId
      videoIds = [item['contentDetails']['videoId'] for item in playlistResult.get('items', [])]
      return videoIds
  except Exception as e:
    print(f"An error occured: {e}")

# save the links in json format and in some file
import json
import os

os.chdir('D:/Machine Learning/NLP/')

urlDict = []
videoIdUrls = fetchVideoUrl(UserInput)

vidFetchTime = timeit.default_timer()
print(f"videos fetched in: {vidFetchTime - start_time} secs")

for i in videoIdUrls:
  videoLink = f"https://www.youtube.com/watch?v={i}"
  urlDict.append(videoLink)

def convertToJson(results):
  with open('videoUrls.json', 'w') as outfile:
    json.dump(results, outfile, indent=2)
    print('data written in JSON file successfully')

convertToJson(urlDict)

# get the captions from each video link, if available
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import logging

# Set up logging
logging.basicConfig(filename='youtube_fetch.log', level=logging.ERROR)

# Modify get_captions to store raw transcript data
def get_captions(vidId):
  try:
    raw_transcripts = []

    for ids in vidId:
      try:
        captions = YouTubeTranscriptApi.get_transcript(
          ids, languages=['en'], preserve_formatting=True
        )
        if captions:
          formatted_captions = [{'text': caption['text']} for caption in captions]
          raw_transcripts.append(formatted_captions)
        else:
          continue
      except TranscriptsDisabled as e:
        print(f"Transcripts are disabled for the video: {str(e)}")
      except Exception as e:
        logging.error(f"Error while fetching the videos: {str(e)}")
    return raw_transcripts
  except Exception as e:
    logging.error(f"Error in getting captions: {str(e)}")

# Modify formatToText to handle the correct data structure
from youtube_transcript_api.formatters import TextFormatter
def formatToText(data):
  formatter = TextFormatter()
  captionFormatted = formatter.format_transcript(data)
  return captionFormatted

raw_transcripts = get_captions(videoIdUrls)
caption = formatToText(raw_transcripts)
# caption = get_captions(videoIdUrls)

capFetchTime = timeit.default_timer()
print(f"captions fetched in: {capFetchTime - vidFetchTime} secs")

# save those captions in a file, all of them in one
with open('captions.txt', 'w') as file:
  file.write(caption)
  print('captions file saved successfully!')

  writingTime = timeit.default_timer()
  print(f"file written in: {writingTime - capFetchTime} secs")

end_time = timeit.default_timer()
totalTime = (end_time - start_time)
print(f"total time taken: {totalTime} secs")