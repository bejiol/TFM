# Offline topic detection and clustering for time limited events in Twitter

This repository keeps the source code and data employed in this paper.


### Description of the main files:
- **twitter-listener.py** : Script employed for downloading event tweets. 
- **credenciales-twitter.py** : Used by **twitter-listener.py**, defines the variables needed to access to the API of Twitter. They can be obtained after defining a  [Twitter Application](https://apps.twitter.com/).
- **tweet-import-preprocess.py**: Imports the tweets into MongoDB (which is assumed to be installed) and applies the preprocessing  operations described in  **preprocesado.js**. 
- **tweet-analysis.py**: Script devoted to topic detection, starts with some time interval of the event. Imports the modules **dateTimePicker.py** and **scroll.py**.
- **config.json**: Configuration file employed by the script **tweet-analysis.py** during the topic detection phase.
- **tweet-time-clustering.py**:  Script responsible of the temporal aggregation of the detected topics and of the visualization of the moments obtained.
- **tweet-textual-clustering.py**: Script responsible of the textual aggregation of the detected topics and of the visualization of the thematic clusters obtained.





