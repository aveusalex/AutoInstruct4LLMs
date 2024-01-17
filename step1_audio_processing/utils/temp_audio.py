import os
from fastapi import UploadFile

PATH = os.getcwd().split('AutoInstruct4LLMs')[0] + 'AutoInstruct4LLMs/'
temp_audio_folder = os.path.join(PATH, 'dados/bronze/temp_audios/')


def create_temp_audio(audio, filename) -> str:
    temp_audio_path = temp_audio_folder + filename
    with open(temp_audio_path, 'wb') as temp_audio:
        temp_audio.write(audio.read())
    return temp_audio_path


def delete_temp_audio(filename):
    temp_audio_path = temp_audio_folder + filename
    os.remove(temp_audio_path)