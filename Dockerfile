FROM python:3.8
#The python image uses /usr/src/app as the default run directory:
WORKDIR /usr/src/app
#Copy from the local current dir to the image workdir:
COPY . .
#Install any dependencies listed in our ./requirements.txt:
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader stopwords
#Run api.py on container startup:
CMD [ "python", "./launch.py" ]