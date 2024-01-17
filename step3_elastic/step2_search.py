from elasticsearch import Elasticsearch
import os
from LLMs import embedding_calculator
import time

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


class DocumentsSearch:
    def __init__(self, index_name: str, password_elastic):
        """
        :param index_name: nome do índice
        """
        assert password_elastic is not None, "Defina a senha do elasticsearch"
        self.es = Elasticsearch("https://localhost:9200",
                                ca_certs=f"{cwd}/step3_elastic/http_ca.crt",
                                basic_auth=("elastic", password_elastic),
                                verify_certs=False)
        self.index_name = index_name
        self.emb = embedding_calculator.Ada002(api_key_path=f"{cwd}/key.txt")

    def semantic_search(self, phrase: str, persona: str, top_n=10):
        """
        Perform a semantic search in the given index using the provided embedding.

        :param phrase: A frase para buscar
        :param persona: Valor do campo 'persona' para filtrar (e.g., "atendente", "cliente")
        :param top_n: Número de resultados top para retornar
        """
        # Calculando o embedding da frase
        embedding, _ = self.emb.get_embedding(phrase)
        query = {
            "script_score": {
                "query": {
                    "bool": {
                        "must": {
                            "match_all": {}
                        },
                        "filter": {
                            "term": {
                                "persona": persona
                            }
                        }
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embed') + 1.0",
                    "params": {"query_vector": embedding}
                }
            }
        }

        response = self.es.search(
            index=self.index_name,
            body={"query": query, "size": top_n},
            _source=False  # Change to True if you want to see the source documents
        )

        respostas = []
        for resposta in response['hits']['hits']:
            hit_id = resposta["_id"]
            semelhanca = resposta['_score']
            source = self.es.get(index=self.index_name, id=hit_id)["_source"]
            respostas.append([source, semelhanca])

        return respostas


# Execução principal
if __name__ == "__main__":
    anatel_search = DocumentsSearch("anatel_index")
    frase = "Quero reclamar da vivo, eles tão me cobrando caro demais e não foi combinado esse preço"
    persona = "atendente"  # Exemplo: filtrar por documentos relacionados a 'cliente'
    start = time.time()
    results = anatel_search.semantic_search(frase, persona)
    print(f"{time.time() - start:.2f} segundos")
    for resultado in results:
        print(resultado)
