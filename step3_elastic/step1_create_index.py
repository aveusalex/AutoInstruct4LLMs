from step0_mapping import embedding_mapping
from elasticsearch import Elasticsearch
from pprint import pprint


if __name__ == '__main__':
    anatel_index = 'anatel_teste_final_index'
    password_elastic = "Vlihd6gtJorsSwrEjSKL"

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
        # perguntando se quer apagar o index
        quest = "y" #input("Do you want to delete the index? (y/n) ")
        if quest == 'y':
            print("Deleting index...")
            client.indices.delete(index=anatel_index)
