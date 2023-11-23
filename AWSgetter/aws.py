import boto3
import os
from tqdm import tqdm


class S3download:
    def __init__(self, path_key):
        with open(path_key, "r") as f:
            lines = f.readlines()
            self.access_key_id = lines[1].split(",")[0].strip()
            self.secret_key = lines[1].split(",")[1].strip()
            del lines
        self.s3 = boto3.client('s3',
                               aws_access_key_id=self.access_key_id,
                               aws_secret_access_key=self.secret_key)

    def download(self, bucket: str, filename: str, path_dest: str) -> None:
        """
        Faz o download de um arquivo do S3.
        :param bucket: o nome do bucket
        :param filename: o nome do arquivo
        :param path_dest: o caminho de destino para salvar o arquivo
        :return: none
        """
        # verificando se path_dest existe
        if not os.path.exists(path_dest):
            os.makedirs(path_dest)

        if not filename.split("/")[-1].strip():  # verifica se o nome é um diretório ou arquivo isolado
            path_dest = os.path.join(path_dest, filename)
        else:
            path_dest = os.path.join(path_dest, filename.split("/")[-1].strip())
        try:
            self.s3.download_file(bucket, filename, path_dest)
        except Exception as e:
            print(f'Erro ao baixar o arquivo {filename}: {e}')
        print(f'{filename} baixado com sucesso!')

    def download_pasta(self, bucket: str, path: str, path_dest: str) -> None:
        """
        Faz o download de todos os arquivos em uma pasta do S3.
        :param bucket: o nome do bucket
        :param path: o caminho da pasta
        :param path_dest: o caminho de destino para salvar os arquivos
        :return: none
        """
        paginator = self.s3.get_paginator('list_objects')
        iterable = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=path)
        for result in iterable:
            if result.get('CommonPrefixes') is not None:
                for subdir in result.get('CommonPrefixes'):
                    self.download(bucket, subdir.get('Prefix'), path_dest)
            for file in tqdm(result.get('Contents', []), desc=f'Fazendo download da pasta {path}'):
                destination_file_name = os.path.join(path_dest, os.path.basename(file.get('Key')))
                if not os.path.exists(path_dest):
                    os.makedirs(path_dest)
                try:
                    self.s3.download_file(bucket, file.get('Key'), destination_file_name)
                except Exception as e:
                    print(f'Erro ao baixar o arquivo {file.get("Key")}: {e}')
        print(f'Pasta {path} baixada com sucesso!')

    def list_s3_objects(self, bucket_name, prefix='', include_files=True):
        paginator = self.s3.get_paginator('list_objects_v2')
        folders = []

        # Adjust the pagination parameters as per your requirements
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # printando os folders e criando uma lógica de printar arquivos somente se include_files for True
                    if not key.endswith('/'):
                        if include_files:
                            print(key)
                        paths = "/".join(key.split('/')[:-1]) + "/"
                        if paths not in folders:
                            folders.append(paths)
                    elif key.endswith('/'):
                        if key not in folders:
                            folders.append(key)
        print("############ Folders ############")
        for folder in folders:
            print(folder)

    def count_qtd_audios(self, bucket: str, nome_empresa: str = "") -> int:
        # lista os objetos no bucket
        qtd = 0
        response = self.s3.list_objects(Bucket=bucket)
        if 'Contents' in response:
            for obj in response['Contents']:
                if nome_empresa in obj['Key'] and (obj['Key'].endswith('.opus') or obj['Key'].endswith('.wav')):
                    qtd += 1
        else:
            print('O bucket está vazio ou não existe')

        return qtd

    def personalized_function(self, function, *args):
        return function(self.s3, *args)


if __name__ == '__main__':
    listar = input('Deseja listar os objetos no bucket? (s/[n]): ').lower().strip() == 's'

    # Substitua 'NOME_DO_BUCKET' pelo nome real do seu bucket S3
    nome_do_bucket = 'ceia-bucket'

    # Substitua 'NOME_DO_OBJETO' pelo nome do objeto que deseja baixar
    nome_da_pasta = 'hermes-pardini/2023/6/15/'

    # Especifique o local onde deseja salvar o objeto baixado e o nome do arquivo
    caminho_destino = './downloads/pardini'

    # Instancia a classe S3download
    s3 = S3download('../key.csv')

    if listar:
        # Lista os objetos no bucket
        # importante preencher o prefixo e delimiter para listar os objetos de uma pasta específica!!!
        s3.list_bucket(nome_do_bucket, prefix='', delimiter="")

    else:
        # Faz o download do objeto
        s3.download_pasta(nome_do_bucket, nome_da_pasta, caminho_destino)