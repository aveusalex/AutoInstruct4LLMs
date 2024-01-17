import pandas as pd
from step1_audio_processing.denoising import Denoising, denoisar, denoisar_nuvem
import os
import time
from tqdm import tqdm
import concurrent.futures
from step1_audio_processing.URA.features_extractor import FC_parallel
from step1_audio_processing.URA.URA_removal import RemoveURA
from step1_audio_processing.channel_separation import separa_canais
from tools import check_path_exists
from step1_audio_processing.transcricao import STT

"""
path_audio = './dados/bronze/TESTE'
path_audio_denoised = './dados/silver/teste_denoised'
path_features = './dados/silver/teste_features_audios'
path_cut_audios = './dados/silver/teste_cut_audios'
path_transcricoes = 'dados/gold/teste_transcricoes_backup'
"""


def main(path, path_denoised, path_features, path_cut_audios, path_transcricoes):
    # path_audio das ligações brutas
    check_path_exists(path, path_denoised, path_features, path_cut_audios, path_transcricoes)

    # perguntando se deseja skipar alguns passos
    # skip = input("Deseja skipar alguns passos?\n"
    #              "1- Denoiser\n"
    #              "2- Extração de features\n"
    #              "3- Remoção de URA\n"
    #              "4- Separar os canais\n"
    #              "5- Transcrição\n")

    skip = ""
    # ############################## STEP 1: DENOISER ##################################
    # instanciando o objeto do modelo de denoising
    if not (skip and '1' in skip):
        print("\033[92m" + "Denoisando..." + "\033[0m")
        dns = Denoising()

        # listando os arquivos de áudio no diretório
        audios = os.listdir(path)
        audios = [audio for audio in audios if audio.endswith('.opus')]
        print(f"Total de áudios: {len(audios)}")

        # verificando quais já foram denoised
        try:
            denoised = os.listdir(path_denoised)
            audios = [audio for audio in audios if audio.replace('.opus', '_cliente_denoised.wav') not in denoised]
        except FileNotFoundError:
            pass
        print(f"Total de áudios a serem denoised: {len(audios)}")

        start = time.time()
        # denoising de cada áudio de forma sequencial
        for audio in tqdm(audios):
            denoisar(audio, dns, path_bronze=path, path_silver=path_denoised)
        print(f"Tempo total sequencial: {time.time() - start}")

    # ############################## STEP 2: EXTRAÇÃO DE FEATURES ##################################
    MAX_THREADS = 4
    # extraindo features dos áudios
    path_ligacoes_denoised = path_denoised  # fonte das ligações denoised
    path_save_features_ = path_features  # onde irá salvar os csvs com as features
    if not (skip and '2' in skip):
        print("\033[92m" + "Extraindo features..." + "\033[0m")
        audios_list_ = os.listdir(path_ligacoes_denoised)
        audios_list_ = [audio for audio in audios_list_ if (audio.endswith('.wav') or
                                                            audio.endswith('.mp3') or
                                                            audio.endswith('.opus'))]
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
            # executando as threads
            for audio in audios_list_:
                executor.submit(FC_parallel, audio, path_ligacoes_denoised, path_save_features_)
        # for audio in tqdm(audios_list_):
        #     FC_parallel(audio, path_ligacoes_denoised, path_save_features_)

    # ############################## STEP 3: REMOÇÃO DA URA ##################################
    if not (skip and '3' in skip):
        print("\033[92m" + "Removendo URA..." + "\033[0m")
        # removendo a URA dos áudios
        remover = RemoveURA(path_save_features_)
        df = remover.run_default(5)
        # percorrendo o df e recortando os áudios
        path_save_cut_audios_ = path_cut_audios
        for idx, row in df.iterrows():
            audio = row["audio"].replace("features_", "").replace(".csv", "")
            inicio_atendimento = row["inicio_atendimento"]
            # recortando o audio
            remover.cut_audio(audio, path_audios=path_ligacoes_denoised,
                              time=inicio_atendimento, path_save=path_save_cut_audios_)

    # ########################## STEP 4: SEPARANDO OS CANAIS ##############################
    # separando os canais de áudio
    PATH_SRC = path_cut_audios
    PATH_DST = path_cut_audios
    if not (skip and '4' in skip):
        print("\033[92m" + "Separando os canais..." + "\033[0m")
        separa_canais(PATH_SRC, PATH_DST)

    # ############################## STEP 5: TRANSCRIÇÃO ##################################
    PATH_GOLD = path_transcricoes
    if not (skip and '5' in skip):
        print("\033[92m" + "Transcrevendo..." + "\033[0m")
        # instanciando o objeto do modelo de transcrição
        path_audios = path_cut_audios
        stt = STT(path_voicerecords=PATH_DST)
        # listando os arquivos de áudio no diretório
        audios_dirs = os.listdir(PATH_DST)
        audios_dir = [audio_dir for audio_dir in audios_dirs if
                      os.path.isdir(os.path.join(path_audios, audio_dir))]  # filtrando apenas pastas

        print(f"Total de áudios para fazer transcrição: {len(audios_dir)}")

        for audio in tqdm(audios_dir):
            # transcrevendo o atendente
            path_atendente = os.path.join(PATH_DST, audio, audio + "_atendente_cut.mp3")
            transcricao_atendente = stt.get_local_insanely_fast_whisper_transcription(path_atendente)
            # transcrevendo o cliente
            path_cliente = os.path.join(PATH_DST, audio, audio + "_cliente_cut.mp3")
            transcricao_cliente = stt.get_local_insanely_fast_whisper_transcription(path_cliente)

            # colocando as transcrições no df
            df_aux_atendente = pd.DataFrame(transcricao_atendente, columns=["start", "end", "transcription"])
            df_aux_atendente["role"] = "atendente"
            df_aux_cliente = pd.DataFrame(transcricao_cliente, columns=["start", "end", "transcription"])
            df_aux_cliente["role"] = "cliente"
            # concatenando os df
            df_aux = pd.concat([df_aux_atendente, df_aux_cliente], ignore_index=True)
            # ordenando o df
            df_aux = df_aux.sort_values(by=["start"])
            # salvando o df
            df_aux.to_csv(os.path.join(PATH_GOLD, audio + ".csv"), index=False)


if __name__ == '__main__':
    path = '../dados/bronze/anatel_big'
    path_denoised = '../dados/silver/anatel_big_denoised'
    path_features = '../dados/silver/anatel_teste_final_features'
    path_cut_audios = '../dados/silver/anatel_teste_final_cut'
    path_transcricoes = '../dados/gold/anatel_teste_final_transcricoes'
    main(path, path_denoised, path_features, path_cut_audios, path_transcricoes)
