FROM tfjob:base
MAINTAINER ma.xiangxiang@eisoo.com
ENV LANG en_US.utf8
ENV PYTHONPATH /tfNLP
ADD __init__.py /tfNLP/__init__.py
ADD classiffier /tfNLP/classiffier
ADD data_processor /tfNLP/data_processor
ADD modeling /tfNLP/modeling
ADD ner /tfNLP/ner
ADD kubeflow /tfNLP/kubeflow

ENTRYPOINT ["python3", "/tfNLP/kubeflow/dist-only-attention.py"]
