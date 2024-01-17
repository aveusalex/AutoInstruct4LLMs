# Código extraído do repositório Data-Juicer, disponível em:
# https://github.com/alibaba/data-juicer/blob/main/data_juicer/ops/mapper/fix_unicode_mapper.py
import ftfy


def fix_unicode(text):
    text = ftfy.fix_text(text)
    return text
