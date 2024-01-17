import pandas as pd
from LLMs import embedding_calculator
import os
from tqdm import tqdm
from elasticsearch import Elasticsearch
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"
if not os.path.exists(f"{cwd}/dados/gold/embeddings"):
    os.makedirs(f"{cwd}/dados/gold/embeddings")


def main(path_demandas: str = None, password_elastic=None, es_index=None):
    """
    Essa função tem por objetivo transformar as demandas identificadas em embeddings e indexá-las no elasticsearch.
    """
    assert password_elastic is not None, "Deve disponibilizar a senha da instância elasticsearch"
    assert path_demandas is not None, "Deve disponibilizar o diretório das demandas identificadas."
    assert es_index is not None, "Defina o indice elastic search"
    es = Elasticsearch("https://localhost:9200",
                       ca_certs=f"{cwd}/step3_elastic/http_ca.crt",
                       basic_auth=("elastic", password_elastic),
                       verify_certs=False)

    # carregando as demandas extraídas das transcrições
    df = pd.read_csv(f"{cwd}/{path_demandas}/demandas_identificadas.csv")
    # criando coluna embeddings no dataframe
    df["embedding"] = None
    df['embedding'] = df['embedding'].astype('object')
    # criando os embeddings para cada demanda
    obj = embedding_calculator.Ada002(api_key_path=f"{cwd}/key.txt")
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        embedding, tokens_in = obj.get_embedding(row["fala_reescrita"])
        doc = {
            "segmento_de_texto": row.fala_reescrita,
            "persona": "cliente",
            "categoria": "demanda",
            "arquivo": row.arquivo,
            "fala_original": row.fala_cliente,
            "embed": embedding
        }

        es.index(index=es_index, document=doc)

    # salvando o dataframe
    print(f"Tokens usados: {obj.tokens_in} - Preço: U${obj.tokens_in * 0.0001/1000:.3f}")


if __name__ == '__main__':
    main()
