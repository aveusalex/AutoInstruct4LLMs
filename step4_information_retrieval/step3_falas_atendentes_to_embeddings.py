import pandas as pd
from LLMs import embedding_calculator
import os
from tqdm import tqdm
from elasticsearch import Elasticsearch
from step2_text_preprocessing.main_preprocessing import Preprocessamento
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"
if not os.path.exists(f"{cwd}/dados/gold/embeddings"):
    os.makedirs(f"{cwd}/dados/gold/embeddings")


def main(path_transcricoes: str = "dados/gold/anatel_transcricoes/", password_elastic=None, es_index=None):
    """
    Essa função tem por objetivo transformar as falas dos atendentes em embeddings e indexá-las no elasticsearch.
    """

    assert password_elastic is not None, "Deve fornecer a senha do elasticsearch."
    assert es_index is not None, "Deve fornecer o indice elasticsearch."
    # instanciando o objeto de preprocessamento
    obj_preprocessamento = Preprocessamento()

    # instancia o objeto do elasticsearch

    es = Elasticsearch("https://localhost:9200",
                       ca_certs=f"{cwd}/step3_elastic/http_ca.crt",
                       basic_auth=("elastic", password_elastic),
                       verify_certs=False)

    # carregando as transcrições
    transcricoes = os.listdir(f"{cwd}/{path_transcricoes}")
    obj = embedding_calculator.Ada002(api_key_path=f"{cwd}/key.txt")

    for transc in tqdm(transcricoes):
        # carregando as transcricoes do atendente como uma frase
        df = pd.read_csv(f"{cwd}/{path_transcricoes}/{transc}")
        # selecionando as falas do atendente
        df = df[df.role == "atendente"]
        # agrupando as falas do atendente em uma string
        fala_atendente = " ".join(df.transcription)

        # preprocessando o texto
        fala_atendente = obj_preprocessamento.preprocessar_texto(fala_atendente, transc)
        # remover numeros
        fala_atendente = obj_preprocessamento.remove_numeros(fala_atendente)

        # calculando o embedding da fala do atendente
        embedding, _ = obj.get_embedding(fala_atendente)
        # salvando embedding no dataframe geral
        doc = {
            "segmento_de_texto": fala_atendente,
            "persona": "atendente",
            "arquivo": transc,
            "fala_original": "N/A",
            "embed": embedding
        }

        es.index(index=es_index, document=doc)

    # salvando o dataframe
    print(f"Tokens usados: {obj.tokens_in} - Preço: U${obj.tokens_in * 0.0001/1000:.3f}")


if __name__ == '__main__':
    main()
