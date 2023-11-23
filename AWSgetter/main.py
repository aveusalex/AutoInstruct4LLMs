from AWSgetter.aws import S3download

bucket = 'ceia-bucket'
s3 = S3download('../key.csv')
empresa = 'anatel/'
# descobrindo quais pastas temos da empresa
s3.list_s3_objects(bucket, prefix=empresa, include_files=False)

caminho_destino = '../dados/bronze/anatel'

# s3.download_pasta(bucket, empresa, caminho_destino)
pastas = [
    "anatel/2023/6/15/",
    "anatel/2023/6/19/",
    "anatel/2023/6/20/",
    "anatel/2023/6/21/",
    "anatel/2023/6/22/",
    "anatel/2023/6/26/",
    "anatel/2023/6/27/",
    "anatel/2023/6/29/",
    "anatel/2023/6/30/",
    "anatel/2023/7/3/",
    "anatel/2023/7/4/",
    "anatel/2023/7/5/"]

for pasta in pastas:
    s3.download_pasta(bucket, pasta, caminho_destino)
