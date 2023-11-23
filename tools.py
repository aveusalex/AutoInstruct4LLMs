import os


def verify_and_create_dir(path: str) -> None:
    """
    Função que verifica se um diretório existe e, caso não exista, cria o mesmo.
    :param path: o caminho do diretório a ser verificado/criado.
    :return: None
    """
    if not os.path.exists(path):
        print(f"O diretório {path} não existe. Criando...")
        try:
            # criando o diretório de logs
            os.makedirs(path)
        except Exception as e:
            print(f"Não foi possível criar o diretório {path}. Erro: {e}")
            exit(1)
