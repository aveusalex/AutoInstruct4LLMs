embedding_mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "text_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "stemmer_text"
                    ]
                }
            },
            "filter": {
                "stemmer_text": {
                    "type": "stemmer",
                    "language": "brazilian"
                }
            }
        }
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "segmento_de_texto": {  # o texto transcrito do segmento de áudio
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword"
                    }
                },
            },
            "persona": {  # atendente ou cliente
                "type": "keyword"
            },
            "categoria": {  # demanda ou solução / resposta
                "type": "keyword"
            },
            "arquivo": {
                "type": "keyword"
            },
            "fala_original": {
                "type": "text"
            },
            "embed": {
                "type": "dense_vector",
                "dims": 1536
            }
        }
    }
}