import tweepy
import datetime
import sys
import logging
import logging.handlers
import json
import requests
import traceback
from credenciales_twitter import (CONSUMER_TOKEN, CONSUMER_SECRET, ACCESS_TOKEN,
    ACCESS_TOKEN_SECRET) # credenciales como cadenas de texto


#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def __init__(self, api):
        super().__init__(api)
        self.logTwitter = logging.getLogger('twitterMsg')
        self.logError = logging.getLogger('error')
        self.total = 0
        self.contador = 0
        self.minuto = datetime.datetime.now().minute

    def on_status(self, status):
        try: 
          self.logTwitter.info(json.dumps(status._json))
        except:
          self.logError.error("Error al volcar el tweet {0}".format(json.dumps(status._json)) )
          
        self.contador += 1
        self.total += 1
        date = datetime.datetime.now()
        minuto = date.minute
        if minuto != self.minuto:
            self.logError.info( "{0} tweets  -- Total {1} tweets".format(self.contador, self.total) )
            self.minuto = minuto
            self.contador = 0
        
    def on_error(self, status_code):
        if status_code == 420: #rate limited error
            self.logError.critical("You are being rate limited. Exiting.")
            sys.exit()
            return False
        
        
def logs_setup(folder, basename):
    twitterMsg_log_handler = logging.handlers.TimedRotatingFileHandler(folder + basename, when='h', interval=1)
    twitterMsg_log = logging.getLogger('twitterMsg')
    twitterMsg_log.addHandler(twitterMsg_log_handler)
    twitterMsg_log.setLevel(logging.INFO)
    
    error_log = logging.getLogger('error')
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(folder + 'ERROR_MESSAGES', mode='w')
    fileHandler.setFormatter(formatter)
    error_log.setLevel(logging.DEBUG)
    error_log.addHandler(fileHandler)    
      
    
def search(keywords, api):
    logError = logging.getLogger('error')
    while(True):
        try:
            logError.info("Connecting to the stream, track=" + str(keywords))
            myStreamListener = MyStreamListener(api)
            myStream = tweepy.Stream(auth = api.auth,listener=myStreamListener)
            myStream.filter(track=keywords, stall_warnings=True)
        except requests.exceptions.ConnectionError:
            logError.critical("ERROR: Max retries exceeded. Connection by Twitter refused. Exiting...")
            sys.exit()
        except KeyboardInterrupt:
            logError.info("Exiting normally")
            myStream.disconnect()
            break
        except Exception as e:
            """The most common exceptions are raised when the connection
                is lost or the tweets are arriving faster than they can
                be processed. We can ignore these exceptions, reconnect
                and continue collecting tweets."""
            logError.error("Exception: {0} {1}.\n\n{2}\nNot critical. Continuing...".format(type(e), e, traceback.format_exc()))
            continue


def usage(exec_name):
  print("   *** El número de parámetros es incorrecto ***\n")
  print("   Llamada:")
  print("      {0} <ruta_carpeta> <nombre_ficheros> <término1> ...\n".format(exec_name))
  print("   Ejemplo:")
  print("      {0} /home/kike/twitterMsgs/ Twitterclasico clásico clasico elclásico elclasico\n".format(exec_name))
  

if __name__ == "__main__":
    if len(sys.argv) < 4:
      usage(sys.argv[0])
      sys.exit(-1)
      
    logs_setup(sys.argv[1], sys.argv[2])

    # usa los credenciales cargados desde credenciales_twitter
    auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET) 
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
   
    search(sys.argv[3:], api)
    #GEOBOX_WORLD = [-180,-90,180,90]
    #GEOBOX_GERMANY = [5.0770049095, 47.2982950435, 15.0403900146, 54.9039819757]
       
    #myStreamListener = MyStreamListener()
    #myStream = tweepy.Stream( auth, listener=MyStreamListener())
    #myStream.filter(locations=GEOBOX_WORLD)
    #myStream.filter(track =  ['vistalegre2'])
