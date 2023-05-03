FROM jupyter/datascience-notebook:latest 

USER python

COPY requirements_conj.txt /home/python/satpoint/requirements.txt
COPY data/ /home/python/satpoint/data/
COPY modules /home/python/satpoint/modules/
COPY *.ipynb /home/python/satpoint/
WORKDIR /home/python/satpoint
USER python
 
RUN pip install -r requirements.txt

CMD jupyter-notebook
