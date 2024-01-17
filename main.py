from step1_audio_processing.main_audio import main as main_audio
from step3_elastic.step0_mapping import embedding_mapping
from step4_information_retrieval.step1_identificador_demanda import IdentificadorDemanda
from step4_information_retrieval.step2_demandas_to_embedding import main as main_demandas_emb
from step4_information_retrieval.step3_falas_atendentes_to_embeddings import main as main_respostas_emb
from step4_information_retrieval.step4_busca_respostas_pelas_demandas import main as main_busca_respostas
from step4_information_retrieval.step5_refina_candidatos_para_cada_demanda import main as main_refina_candidatos
from step4_information_retrieval.step6_combina_respostas_refinadas import main as main_combina_candidatos
from step5_instruct_formatter.step1_instruct_func import add_instruction_to_dataframe
from elasticsearch import Elasticsearch
from pprint import pprint
import os
import pandas as pd
cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


OPERACAO = "teste"  # selecione aqui o nome da operação. Deve ser o mesmo nome da pasta onde os áudios estão (bronze)

check_path = f"./checkpoints_{OPERACAO}.txt"
if os.path.exists(check_path):
    checkpoint = open(check_path).read()
else:
    checkpoint = "0"
    with open(check_path, "w") as file:
        file.write(checkpoint)

# definindo as variáveis de paths
path_audio = f'./dados/bronze/{OPERACAO}'
path_audio_denoised = f'./dados/silver/{OPERACAO}_denoised'
path_audio_features = f'./dados/silver/{OPERACAO}_features'
path_audio_cut = f'./dados/silver/{OPERACAO}_cut'
path_audio_transcricoes = f'./dados/gold/{OPERACAO}_transcricoes'
path_demandas = f"/dados/gold/demandas_{OPERACAO}"

anatel_index = f'{OPERACAO}_index'
password_elastic = "Vlihd6gtJorsSwrEjSKL"

# ####################### AUDIO STEPS ############################
if checkpoint == "0":

    main_audio(path_audio, path_audio_denoised, path_audio_features, path_audio_cut, path_audio_transcricoes)

    checkpoint = "1"
    with open(check_path, "w") as file:
        file.write(checkpoint)

# ####################### INDEXING STEPS ##########################
if checkpoint == "1":
    client = Elasticsearch("https://localhost:9200",
                           ca_certs="./http_ca.crt",
                           basic_auth=("elastic", password_elastic),
                           verify_certs=False)

    if not client.indices.exists(index=anatel_index):
        client.indices.create(index=anatel_index, settings=embedding_mapping['settings'], mappings=embedding_mapping['mappings'])
        pprint(client.indices.get_mapping(index=anatel_index))
        pprint(client.indices.get_settings(index=anatel_index))
    else:
        print("Index already created.")

    checkpoint = "2"
    with open(check_path, "w") as file:
        file.write(checkpoint)

# ########### BUSCA POR DEMANDAS E RESPOSTAS ########################
if checkpoint == "2":
    print("\033[92mIniciando a identificação de demandas...\033[0m")
    idd = IdentificadorDemanda(path_audio_transcricoes, api_key_path="./key.txt")
    idd.identificar_demandas(path_to_save=path_demandas)
    checkpoint = "3"
    with open(check_path, "w") as file:
        file.write(checkpoint)

if checkpoint == "3":
    # indexando as demandas achadas no elastic search e calculando seus embeddings
    print("\033[92mIniciando a indexação de demandas no elasticsearch...\033[0m")
    main_demandas_emb(path_demandas, password_elastic, es_index=anatel_index)
    checkpoint = "4"
    with open(check_path, "w") as file:
        file.write(checkpoint)

if checkpoint == "4":
    # indexando as respostas (falas atendentes) no elastic search e calculando seus embeddings
    print("\033[92mIniciando a indexação de respostas dos atendentes no elasticsearch...\033[0m")
    main_respostas_emb(path_transcricoes=path_audio_transcricoes, password_elastic=password_elastic,
                       es_index=anatel_index)
    checkpoint = "5"
    with open(check_path, "w") as file:
        file.write(checkpoint)

if checkpoint == "5":
    # buscando respostas para as demandas através da busca semântica
    print("\033[92mIniciando a busca semântica por respostas para as demandas...\033[0m")
    main_busca_respostas(path_demandas=path_demandas, password_elastic=password_elastic)
    checkpoint = "6"
    with open(check_path, "w") as file:
        file.write(checkpoint)

if checkpoint == "6":
    # refinando as respostas encontradas através da busca semântica
    print("\033[92mIniciando o refinamento das respostas...\033[0m")
    main_refina_candidatos(path_demandas=path_demandas)
    checkpoint = "7"
    with open(check_path, "w") as file:
        file.write(checkpoint)

if checkpoint == "7":
    # combinando as respostas refinadas em uma única resposta, para ficar o par demnada - resposta
    print("\033[92mIniciando a combinação das respostas refinadas...\033[0m")
    main_combina_candidatos(n_threads=30, path_demandas=path_demandas)
    checkpoint = "8"
    with open(check_path, "w") as file:
        file.write(checkpoint)

if checkpoint == "8":
    # adicionando as intruções ao dataset
    add_instruction_to_dataframe(df_path=path_demandas + "/demandas_respostas_final.csv")
    checkpoint = "9"
    with open(check_path, "w") as file:
        file.write(checkpoint)

if checkpoint == "9":
    print("\033[92mProcesso finalizado com sucesso!\033[0m")
    df = pd.read_csv(cwd + path_demandas + "/demandas_respostas_final_instruct.csv")
    amostras = df.sample(2)
    for i, amostra in amostras.iterrows():
        print(f'Exemplos:\nInstrução: {amostra["instruction"]}\n'
              f'Demanda: {amostra["fala_reescrita_cliente"]}\n'
              f'Resposta: {amostra["respostas_unificadas"]}\n')