"""
Esse código tem por objetivo realizar, para cada demanda, a busca semântica no elasticsearch por respostas.
"""
from step3_elastic.step2_search import DocumentsSearch
from time import time
import pandas as pd
import os
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


def main(path_demandas=None, password_elastic=None):
    """
    Essa função tem por objetivo buscar as respostas mais semelhantes às demandas feitas pelos clientes.
    """
    assert path_demandas is not None, "Especifique o caminho para as demandas"
    assert password_elastic is not None, "Defina a senha do elasticsearch"
    # instanciando o objeto de busca semântica:
    anatel_search = DocumentsSearch("anatel_teste_final_index", password_elastic)

    # buscando as demandas dos clientes
    df = pd.read_csv(f"{cwd}/{path_demandas}/demandas_identificadas.csv")

    # instanciando o dataframe final
    df_final = pd.DataFrame(columns=["arquivo", "fala_reescrita_cliente", "respostas_candidatas", "tempo_busca", "similaridade_max"])

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        frase = row.fala_reescrita
        persona = "atendente"

        start = time()  # medindo o tempo que leva para a busca
        results = anatel_search.semantic_search(frase, persona)
        tempo = time() - start

        # verificando
        resultado_pesquisa = ""
        similaridade_maxima = -1
        for resultado in results:
            texto_candidato = resultado[0]['segmento_de_texto']
            similaridade = resultado[1]
            if similaridade > similaridade_maxima:
                similaridade_maxima = similaridade

            resultado_pesquisa += texto_candidato + "<sep>"

        # Coletando o nome do arquivo e a fala reescrita do cliente
        arquivo = row['arquivo']  # Substitua 'arquivo' pelo nome da coluna correspondente, se for diferente
        fala_reescrita_cliente = row['fala_reescrita']  # Presumo que esta seja a coluna correta

        # Incluindo os dados no dataframe
        df_aux = pd.DataFrame({
            "arquivo": [arquivo],
            "fala_reescrita_cliente": [fala_reescrita_cliente],
            "respostas_candidatas": [resultado_pesquisa],
            # ou a melhor resposta, se você a estiver armazenando em outra variável
            "tempo_busca": [tempo],
            "similaridade_max": [similaridade_maxima]
        })

        df_final = pd.concat([df_final, df_aux], axis=0, ignore_index=True)

    # salvando o dataframe
    df_final.to_csv(f"{cwd}/{path_demandas}/demandas_e_respostas_candidatas.csv", index=False)
