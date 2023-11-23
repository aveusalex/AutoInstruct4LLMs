from audio_processing.denoising import Denoising, denoisar, denoisar_nuvem
import multiprocessing
import threading
import os
import numpy as np
import time
from tqdm import tqdm
import concurrent.futures
from audio_processing.URA.features_extractor import FC_parallel
from audio_processing.URA.URA_removal import RemoveURA
from audio_processing.transcricao import STT


if __name__ == '__main__':

    # path das ligações brutas
    path = './dados/bronze/anatel'

    # perguntando se deseja skipar alguns passos
    skip = input("Deseja skipar alguns passos?\n"
                 "1- Denoiser\n"
                 "2- Extração de features\n"
                 "3- Remoção de URA\n"
                 "4- Transcrição\n")
    skip = skip.split(', ')

    # ############################## DENOISER ##################################
    # instanciando o objeto do modelo de denoising
    if not (skip and '1' in skip):
        dns = Denoising()

        # listando os arquivos de áudio no diretório
        audios = os.listdir(path)
        audios = [audio for audio in audios if audio.endswith('.opus')]
        print(f"Total de áudios: {len(audios)}")

        # verificando quais já foram denoised
        try:
            denoised = os.listdir('./dados/silver/anatel')
            audios = [audio for audio in audios if audio.replace('.opus', '_cliente_denoised.wav') not in denoised]
        except FileNotFoundError:
            pass
        print(f"Total de áudios a serem denoised: {len(audios)}")

        start = time.time()
        # denoising de cada áudio de forma sequencial
        for audio in tqdm(audios):
            denoisar_nuvem(audio, dns, path_bronze=path, path_silver="./dados/silver/anatel")
        print(f"Tempo total sequencial: {time.time() - start}")

    # ############################## REMOÇÃO DE URA ##################################
    MAX_THREADS = 4
    # extraindo features dos áudios
    path_ligacoes_denoised = "./dados/silver/anatel"  # fonte das ligações denoised
    path_save_features_ = "./dados/silver/anatel/features_audios"  # onde irá salvar os csvs com as features
    if not (skip and '2' in skip):
        audios_list_ = os.listdir(path_ligacoes_denoised)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
        #     # executando as threads
        #     for audio in audios_list_:
        #         executor.submit(FC_parallel, audio, path_ligacoes_denoised, path_save_features_)
        for audio in tqdm(audios_list_):
            FC_parallel(audio, path_ligacoes_denoised, path_save_features_)

    if not (skip and '3' in skip):
        # removendo a URA dos áudios
        remover = RemoveURA(path_save_features_)
        df = remover.run_default(5)
        # percorrendo o df e recortando os áudios
        path_save_cut_audios_ = r"audios_analise_qualidade_cut"
        for idx, row in df.iterrows():
            audio = row["audio"].replace("features_", "").replace(".csv", "")
            inicio_atendimento = row["inicio_atendimento"]
            # recortando o audio
            remover.cut_audio(audio, path_audios=path_ligacoes_denoised,
                              time=inicio_atendimento, path_save=path_save_cut_audios_)

    # ############################## TRANSCRIÇÃO ##################################
    if not (skip and '4' in skip):
        # instanciando o objeto do modelo de transcrição
        path_audios = "./dados/silver/anatel"
        STT = STT(path_voicerecords=path_audios)
        # listando os arquivos de áudio no diretório
        audios = os.listdir("./dados/silver/anatel")
        audios = [audio for audio in audios if (audio.endswith('.mp3') or
                                                audio.endswith('.wav') or
                                                audio.endswith('.opus'))]
        print(f"Total de áudios para fazer transcrição: {len(audios)}")
        for audio in tqdm(audios):
            path = os.path.join(path_audios, audio)
            STT.get_transcription_nuvem(path)


