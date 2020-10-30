# Natural language processing API Flask

- Année : **M2 WEB HITEMA**
- Matière: *Machin Learning*
- Projet : *Flask API*

## Auteur(s)

|Nom|Prénom|
|--|--|
*EL ASSOURI* | *Mohammed*|

## Les routes

- path: "/entrainement"
    -
    La route "/entrainement" permet d'entrainer le model avec le corpus de base (news.csv). 
    Elle renvoie l'accuracy et le nb de document sous forme d'objet : {'Le training est fini le score est de: ': {'accuracy': 0.7925925925925926, 'size': 1347}}
    **Exemple de requete**
    ```python
      route='/entrainement'
      url='http://127.0.0.1:5000'+route
      r=requests.post(url)
      r.json()
    ``` 
- path: "/prediction"
    -
    La route "/prediction" permet d'analyser un texte et de classer si c'est une Fakenews, soit Fake News soit Vrai News.
    Elle renvoie une reponse sous la forme d'un objet: {'Le texte est une Vrai News'}
    **Exemple de requete**
    ```python
      route='/prediction'
      url='http://127.0.0.1:5000'+route
      param={'input_text':'Donald trump en ralation avec une organisation de burger musulman' }
      r=requests.post(url,data=param)
      r.json()
    ```
## Les méthodes
- doTraining(data)
    -
    La méthode doTraining(data) prend en parametre le DataSet avec lequel on doit entraîner le model.
    elle vectorise les données textes et enregistre le vocabulaire (x), puis elle labellise les données de matching (y).
    elle retourn un objet metrics: {"accuracy": 0.86545465465, "size": 1347}
     
- nettoyage(string)
    -
    La méthode nettoyage(string) prend en parametre le texte a nettoyer.
    elle retourn le texte tokenizé dans un tableau.