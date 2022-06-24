from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from pathlib import Path

def process_sudio(file_name):
    myaudio = AudioSegment.from_file(file_name, "wav")
    chunk_length_ms = 2000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    for i, chunk in enumerate(chunks):
        file_name = Path(os.path.basename(each_file)).stem
        if i != 15:
            chunk_name = './data/chunked_data/' + file_name + "_{0}.wav".format(i)
            print("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")

all_file_names = [] #os.listdir('./data/genres_original/chunked_data/')
for file in os.listdir("./data/genres_original/blues/"):
    if file.endswith(".wav"):
        all_file_names.append(os.path.join("./data/genres_original/blues/", file))


try:
    os.makedirs('data/chunked_data') # creating a folder named chunked
except:
    pass

for each_file in all_file_names:
    process_sudio(each_file)
