import pymongo
import datetime, time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import operator
import math
import time
import json
import collections
import bisect
from tqdm import tqdm
import sys
from dateTimePicker import EventDetailsPicker, main
from scroll import ScrollMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import scipy.sparse as sparse


########################## COMPARACION GRUPOS TWEETS ###################


# Obtenemos los tweets en una ventana determinada, conociendo el número de ventana por orden cronológico
def getWindow(fechainicial, fechafin, num_salto):
	lista = []
	proy = { "id": True, "text": True, "text3": True,"created_at": True, "user": True,"num_RT": True}
	query = {"created_at": 	{"$gte": fechainicial,"$lt": fechafin}, "RT": False,  "repeat": {"$exists": False},"voidtext": {"$exists": False}}
	s = [('created_at', pymongo.ASCENDING)]	

	try:
		res = db.tweets.find(query, proy).sort(s).skip(num_salto * config['tweets_ventana']).limit(config['tweets_ventana'])
	except Exception as e:
		print("Error:", type(e), e)

	for r in res:
		lista.append(r)
	
	return lista

# De un tema, tomar num_muestra_tema muestras aleatorias (indices), todo el tema si n < len(tema)
def muestra_random_tema(grupo):
	if len(grupo) <= config['num_muestra_tema']:
		return grupo
	else:
		return random.sample(grupo, k=config['num_muestra_tema'])

# Calcula la similaridad entre dos tweets
def similaridadTweets(tw1, tw2):
	# Fase de procesamiento de ventanas
	if not fusion: 
		text_sim = similarity_matrix[tw1 - (num_ventana * config['tweets_ventana']), tw2 - (num_ventana * config['tweets_ventana'])]
	# Fase de fusión de ventanas
	else:
		text_sim = similarity_matrix[diccionarioInd[tw1], diccionarioInd[tw2]]


	if text_sim >= 0.1:
		text_sim = math.log10(text_sim) + 1
	else:
		text_sim = 0
	return text_sim

# Dados dos nucleos, calculamos su proximidad media
def proximidadMediaNucleos(nucleo1, nucleo2):
	suma = 0
	for e1 in nucleo1:
		for e2 in nucleo2:
			suma += similaridadTweets(e1, e2)

	return suma / (len(nucleo1) * len(nucleo2))

# Calcula si un tweet es similar a un grupo de tweets, 
# devolviendo True en caso de ser similar y False en caso contrario
def esSimilarAlGrupo(tw, grupo, threshold):
	similares = 0
	suma  = 0.0
	for e in grupo:
		pr = similaridadTweets(tw, e)
		if pr >=  threshold:
			similares +=1
		suma += pr

	media = suma / len(grupo)
	
	if len(grupo) >= config['min_tweets_similares'] and similares >= config['min_tweets_similares'] and media >= threshold:
		return True
	elif len(grupo) <= config['min_tweets_similares'] and media >= threshold:
		return True
	else:
		return False

def word_tokenizer(text):
    tokens = word_tokenize(text.lower())
    # STEMMER DELETED
    tokens = [t for t in tokens if t not in stopwords.words(config['lang'])]
    return tokens

# Dado un conjunto de tweets, calcula la matriz tf-idf de sus textos
def tfidf_calc():
	print("CALCULANDO MATRIZ TF-IDF")
	sentences = []
	for t in tweets:
		sentences.append(re.sub(r'[^\w\s]','',t['text3']))

	tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
	                                    stop_words=stopwords.words(config['lang']),
	                                    # max_df=0.9,
	                                    # min_df=0.0001,
	                                    norm='l2',
	                                    # use_idf=False,
	                                    lowercase=True)
	start = time.time()
	tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
	end = time.time()
	print("---- CALCULADA EN ", round((end - start) / 60.0, 2), " minutos")
	print("---- Matrix shape" , tfidf_matrix.shape)
	return tfidf_matrix

# Calcula la matriz de similaridades global, para la fase de fusión entre ventanas.
# Devuelve el diccionario para poder establecer la relación con los índices de tweet originales
def matrizSimilaridadesGlobal(ventanas):
	global similarity_matrix
	diccionarioIndicesTweets = {}
	count = 0
	for i in range(len(ventanas)):
		grupos = ventanas[i]
		for grupo in grupos:
			if count == 0:
				new = tfidf_matrix[grupo[0]]
				diccionarioIndicesTweets[grupo[0]] = count
				count +=1
				inicio = 1
			for j in range(inicio, len(grupo)):
				new = sparse.vstack((new, tfidf_matrix[grupo[j]]))
				diccionarioIndicesTweets[grupo[j]] = count
				count += 1
			if inicio == 1:
				inicio = 0
	print(new.shape)
	similarity_matrix = cosine_similarity(new, dense_output = False)
	return diccionarioIndicesTweets


################################### PROCESAMIENTO VENTANAS ###############################

# Carga el fichero de configuración con los parámetros de procesamiento
def load_config(filename):
	print("Leyendo configuración de " + filename + "...")
	config = json.loads(open(filename).read())
	print(config)
	return config

# Bucle que obtiene los datos de los tweets de cada una de las ventanas de tamaño tam_ventana
def carga_tweets(horaInicio, horaFin):
	# Dividimos el dia en ventanas del tamaño indicado
	tweets = [] # Almacena los datos completos de los tweets de cada ventana
	info_ventanas = [] #Almacenar las fechas inicio-fin de ventanas y su número de tweets
	print("++++++++++++++++++++VENTANAS+++++++++++++++++")
	num_salto = 0
	w = getWindow(horaInicio, horaFin, num_salto)
	while w != []:
		tweets = tweets + w
		horaInicioVentana = w[0]['created_at']
		horaFinVentana = w[-1]['created_at']
		info_ventanas.append((horaInicioVentana, horaFinVentana, len(w)))
		print(horaInicioVentana, horaFinVentana, len(w))
		num_salto +=1
		w = getWindow(horaInicio, horaFin,num_salto)
	print("+++++++++++++++++++++++++++++++++++++++++++++")
	return tweets, num_salto, info_ventanas

# Dado un número de ventana, calcula la media y la desviación típica de sus frecuencias de twitteo
def frecVentana(num_ventana):
	inicio = info_ventanas[num_ventana][0]
	fin = info_ventanas[num_ventana][1] 
	frecuencias = []
	while inicio <= fin:
		if inicio.strftime('%Y-%m-%dT%H:%M') in info_frecuencias_tweets:
			frecuencias.append(info_frecuencias_tweets[inicio.strftime('%Y-%m-%dT%H:%M')])
		inicio += datetime.timedelta(minutes= 1)
	
	media = np.mean(frecuencias)
	std = np.std(frecuencias)
	return media, std

# Función que realiza el procesamiento de las ventanas de manera contigua
def procesa_tweets():
	global similarity_matrix
	global  num_ventana
	grupos = []
	centros_grupos = []
	inicio_ventana = 0
	for i in range(len(info_ventanas)):
		start = time.time()
		print("---------------------------------------------")
		print("Iniciamos ventana ", i," a las ", time.ctime())
		num_ventana = i
		# Calculamos la matriz de similaridades por coseno de la ventana que toque
		similarity_matrix  = cosine_similarity(tfidf_matrix[inicio_ventana:inicio_ventana + info_ventanas[i][2]], dense_output = False)
		grs_ventana, analiz_ventana = gruposIniciales(info_ventanas[i][2], inicio_ventana, i)
		grs_ventana = unirAGrupos(grs_ventana, analiz_ventana, info_ventanas[i][2], inicio_ventana) 
		centros_grs_ventana = calcularCentros(grs_ventana)
		grs_ventana, centros_grs_ventana = compactar(grs_ventana, centros_grs_ventana)
		grs_ventana, centros_grs_ventana =  fusionInternaGrupos(grs_ventana, centros_grs_ventana)
		grs_ventana = ordenaProximidadCentro(centros_grs_ventana, grs_ventana)
		# Anotamos resultados: grupos, centros y núcleos
		grupos.append(grs_ventana)
		centros_grupos.append(centros_grs_ventana)
		inicio_ventana += info_ventanas[i][2]
		end = time.time()
		print("Acabada ventana ", i, " a las ", time.ctime(), " en ", round((end - start) / 60.0, 2), " minutos")
		print("---------------------------------------------")
	return grupos, centros_grupos


# Obtenemos la información de las frecuencias de twitteo
def informacion_twitteo(horaInicio, horaFin):
	horaFin = horaFin.replace(second = 0) + datetime.timedelta(minutes= 1)
	match = {"$match": {"created_at": {"$gte": horaInicio,"$lt": horaFin}, "RT": False,  "repeat": {"$exists": False},"voidtext": {"$exists": False}}}
	group = {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%dT%H:%M", "date": "$created_at"}}, "count": {"$sum": 1}}}
	sort = {"$sort": {"_id": 1}}
	pipeline = [match, group, sort]
	try:
		res = db.tweets.aggregate(pipeline)
	except Exception as e:
		print("Error:", type(e), e)

	resultado = collections.OrderedDict()
	for r in res:
		resultado[r['_id']] = r['count']

	return resultado

# Para la ventana dada, devuelve una lista de todos los temas de tweets similares, cada tema una lista de indices de tweets de la ventana
def gruposIniciales(numTweetsVentana, posInicial, num_ventana):
	print("Calculamos los temas iniciales de la ventana, con un total de ", numTweetsVentana, " tweets")
	grupos = []
	hit = 0

	# Tomar muestra aleatoria de muestra_ventana tweets (sus indices) por orden de la ventana 
	if numTweetsVentana >= config['muestra_ventana']:
		sample = random.sample(range(posInicial, posInicial + numTweetsVentana), config['muestra_ventana'])
		sample.sort()	
	else: # Si no llegan a X, tomara todos
		sample = range(posInicial, posInicial + numTweetsVentana)

	count = 0
	for tw_ind in tqdm(sample, total=len(sample), unit="tweets"):	
		similares = 0
		for g in grupos:
			group_sample = muestra_random_tema(g)
			# Se puede añadir a ún grupo existente
			if esSimilarAlGrupo(tw_ind, group_sample, config['umbral_inicial']):
				g.append(tw_ind)
				similares +=1
				hit +=1
				break
		# No se ha añadido a ningun grupo, forma un grupo nuevo por si solo		
		if similares == 0:
			nuevo_grupo = []
			nuevo_grupo.append(tw_ind)
			grupos.append(nuevo_grupo)
		# Mostrar progreso porque esto es desesperantemente lento...
		if count % config['filtrar_unitarios'] == 0:
			num_eliminados, grupos = filtrarGruposVentana(grupos)
		count = count + 1

	print("Temas inicialmente formados: ", len(grupos))
	num_eliminados, grupos = filtrarGruposVentana(grupos)
	centros_grupos = calcularCentros(grupos)
	grupos, centros_grupos = fusionInternaGrupos(grupos, centros_grupos)
	num_eliminados, grupos = filtrarGruposFinal(grupos)
	print("Temas tras filtrado: ", len(grupos))
	return grupos, sample

# Tratar de unir los tweets no tomados como muestra inicial a los temas creados
# Estos tweets no podran crear temas propios en caso de no pertenecer a ninguno
def unirAGrupos(grupos, analizados, numTweetsVentana, posInicial):
	print("Tratamos de aumentar los temas creados con tweets de la ventana")
	hit = 0
	count = 0

	analizados_set = set(analizados)
	tweets_restantes = [x for x in range(posInicial, posInicial + numTweetsVentana) if x not in analizados_set]

	for tw_ind in tqdm(tweets_restantes, total=len(tweets_restantes), unit="tweets"):
		similares = 0
		for g in grupos:
			sample = muestra_random_tema(g)
			# Solo se añade a grupos existentes, no se crean grupos nuevos
			if esSimilarAlGrupo(tw_ind, sample, config['umbral_restantes']):
				g.append(tw_ind)
				similares +=1
				hit +=1
	print("Tweets añadidos a los temas existentes ", hit)	
	return grupos

# Fusionamos temas de diferentes ventanas
def fusionExterna(grupos, centros_grupos):
	print("Tratamos de fusionar temas de ventanas diferentes...")
	for i in range(len(info_ventanas)-1):
		gruposi = grupos[i]
		gruposj = grupos[i+1]
		centrosi = centros_grupos[i]
		centrosj = centros_grupos[i+1]
		gruposi, centrosi, gruposj, centrosj = fusionVentanas(gruposi, centrosi, gruposj, centrosj)
		print("Fusión ", i, i+1)
		grupos[i] = gruposi
		grupos[i+1] = gruposj
		centros_grupos[i] = centrosi
		centros_grupos[i+1] = centrosj
	for i in range(len(grupos)):
		grupos[i] = ordenaProximidadCentro(centros_grupos[i], grupos[i])
		i +=1
	return grupos, centros_grupos

# Fusionamos los grupos de dos ventanas consecutivas
def fusionVentanas(grupos_w1, centros_w1, grupos_w2, centros_w2):
 	iw1 = 0
 	iw2 = 0
 	delete_list = []
 	for iw1 in range(len(centros_w1)):
 		iw2 = 0
 		ok = True
 		n1 = nucleoGrupo(centros_w1[iw1], grupos_w1[iw1])
 		while ok and iw2 < len(centros_w2):
 			n2 = nucleoGrupo(centros_w2[iw2],grupos_w2[iw2])
 			if proximidadMediaNucleos(n1, n2) >= config['umbral_fusion_externa']:
 				union = grupos_w1[iw1] + grupos_w2[iw2]
 				centro_union = calculaCentroGrupo(union)
 				num_elim, union_comp, centro_union = compactaGrupo(union, centro_union)
	 			if num_elim <= len(union) * 0.10:
	 				grupos_w2[iw2] = union_comp
	 				centros_w2[iw2] = centro_union
	 				nucleo_comp = nucleoGrupo(centro_union, union_comp)
	 				delete_list.append(iw1)
	 				ok = False
 			iw2 += 1

 	g1_new = []
 	c1_new = []
 	n1_new = {}
 	if len(delete_list) > 0:
	 	for i in range(len(grupos_w1)):
	 		if i not in delete_list:
		 		g1_new.append(grupos_w1[i])
		 		c1_new.append(centros_w1[i])
 	else:
 		g1_new = grupos_w1
 		c1_new = centros_w1

 	return g1_new, c1_new,grupos_w2, centros_w2

################################### OPERACIONES SOBRE TEMAS ###############################

# Calculamos la densidad del tema: num_tweets / tiempo_vida
def densidadGrupo(grupo):
	return len(grupo) / calcularVidaGrupo(grupo).total_seconds()

# Calcula la longitud media de los tweets de un tema, contando con palabras diferentes
def longitudMedia(grupo):
	longitudes = []
	for t in grupo:
		l = set()
		tokens = re.sub(r'[^\w\s]','',tweets[t]['text3']).split()
		for w in tokens:
			if w not in l:
				l.add(w)
		longitudes.append(len(l))

	return np.sum(longitudes, axis=0) / len(grupo)

def numeroUsuarios(grupo):
	numUsuarios = 0
	users = set()
	for e in grupo:
		if tweets[e]['user']['screen_name'] not in users:
			users.add(tweets[e]['user']['screen_name'])
			numUsuarios +=1

	return numUsuarios

def numRTGrupo(grupo):
	numRTs = 0
	for e in grupo:
		numRTs += tweets[e]['num_RT']
	return numRTs

# Devuelve el número de tweets escritos en el rango de minutos indicado
def tweetsRango(mindate, maxdate):
	mindate = mindate.replace(second = 0)
	maxdate = maxdate.replace(second = 0)
	total = 0
	while mindate <= maxdate:
		if mindate.strftime('%Y-%m-%dT%H:%M') in info_frecuencias_tweets:
			total += info_frecuencias_tweets[mindate.strftime('%Y-%m-%dT%H:%M')]
		mindate += datetime.timedelta(minutes= 1)

	return total

# Calcula el centro temporal del tema, que es el tweet que queda en la mediana
def centroTemporal(grupo):
	minutos = []
	for e in grupo:
		time = tweets[e]['created_at']
		bisect.insort(minutos, time)
	ln = len(minutos)
	if ln % 2 == 0:
		m1 = minutos[ln // 2 - 1]
		m2 = minutos[ln // 2]
		mediavida = (m2 - m1).total_seconds() / 2
		m1 += datetime.timedelta(seconds = mediavida)
		mediana = m1
	else:
		mediana = minutos[(ln + 1 ) // 2 - 1]

	return mediana

# Calcula la similaridad media aproximada de los tweets del grupo, tomando una muestra de como máximo 50 tweets
def similaridadMedia(grupo):
	suma = 0.0
	if len(grupo) > 50:
		sample_grupo = random.sample(grupo, k=50)
	else:
		sample_grupo = grupo

	i = 0
	count = 0
	while i < len(sample_grupo) - 1:
		pr = similaridadTweets(sample_grupo[i], sample_grupo[i+1])
		# print(tweets[sample_grupo[i]]['text3'], "<-->", tweets[sample_grupo[i+1]]['text3'], pr)
		suma  +=  pr
		i +=2
		count +=1

	return suma / count

# Calcula el número de tweets diferentes (texto distinto) que hay en un grupo
def numeroTweetsDiferentes(grupo):
	l = []
	for e in grupo:
		l.append(tweets[e]['text3'].replace("\n", " ").replace("\t", " ").replace("\r", " "))
	return len(set(l))


# Devuelve una puntuación asignada al grupo en función de distintos factores:
# - Número de tweets
# - Frecuencia del grupo 
# - Longitud media de los tweets del grupo
# - Similaridad media respecto al centro
# - Tiempo de vida
# - Número de RT
# - Frecuencia absoluta
# - Similaridad media aproximada
def evaluaGrupo(grupo):
	num_tweets = len(grupo)
	# centro  = calculaResumenGrupo(grupo)
	centro  = calculaCentroGrupo(grupo)
	mindate, maxdate = calcularVidaGrupo(grupo)
	tiempo_vida = (maxdate - mindate + datetime.timedelta(seconds = 1)).total_seconds()
	frec_grupo =  num_tweets / tiempo_vida
	long_media = longitudMedia(grupo)
	num_retweets = numRTGrupo(grupo)
	frec_absoluta =  num_tweets / tweetsRango(mindate, maxdate)
	similaridad_media = similaridadMedia(grupo)
	tweets_diferentes = numeroTweetsDiferentes(grupo)
	return num_tweets, frec_grupo, long_media, centro[0], tiempo_vida, num_retweets, frec_absoluta, similaridad_media, tweets_diferentes

# Filtrar los tweets, eliminando los que hayan formado tema unitario
def filtrarGruposVentana(grupos):
	eliminados = 0
	filtrados = []
	for g in grupos:
		if len(g) > 1:
			filtrados.append(g)
		else:
			eliminados +=1
	return eliminados, filtrados

# Filtrar los temas de la fase de formación de temas inicial, según tipo de evento
def filtrarGruposFinal(grupos):
	eliminados = 0
	filtrados = []

	if tipo_evento == 1:
		longitud_media_minima = 3
	else:
		longitud_media_minima = 2

	for g in grupos:
		if len(g) > 1:
			num_tweets, frec_grupo, long_media, centro, tiempo_vida, num_retweets, frec_absoluta, similaridad_media, tweets_diferentes = evaluaGrupo(g)
			# Al menos 5 tweets
			# De 5 usuarios distintos
			# Que el 10% de los tweets sean diferentes en texto
			# Que la longitud media (palabras únicas) sea de un mínimo de 2 palabras
			if num_tweets >= 5 and numeroUsuarios(g) >= 5 and tweets_diferentes > 1 and tweets_diferentes >= num_tweets * 0.10 and long_media > longitud_media_minima:
					filtrados.append(g)
			else:
				eliminados +=1
		else:
			eliminados +=1
	print(len(grupos) - len(filtrados), "/", len(grupos)," temas muertos")
	return eliminados, filtrados

# Dado un tema, calcula su tiempo de vida total
def calcularVidaGrupo(grupo):
	maxdate = datetime.datetime(2001, 1, 1, 0, 0)
	mindate = datetime.datetime.now()

	for e in grupo:
		if tweets[e]['created_at'] < mindate:
			mindate = tweets[e]['created_at']
		if tweets[e]['created_at'] > maxdate:
			maxdate = tweets[e]['created_at']

	return mindate, maxdate

# Calcular el centro de un tema dado
# Se calcula la matriz de proximidades
# Para cada tweet:
# 	- Se calcula la media y desv. típica de le sus proximidades al resto de tweets
# Centro: mayor media de proximidades
# Devuelve (índice del tweet centro del tema, proximidad media, desviación típica, mediana)
def calculaCentroGrupo(grupo):
	prox_matrix = {}
	ln_grupo = len(grupo)

	# Calculamos matriz total de distancias
	for i in range(ln_grupo):
		j = i + 1
		while j < ln_grupo:
			prox = similaridadTweets(grupo[i], grupo[j])
			prox_matrix[(i, j)] = prox
			prox_matrix[(j, i)] = prox
			j+= 1
	# Calculamos proximidades medias		
	medias = []
	for i in range(ln_grupo):
		suma = 0
		for j in range(ln_grupo):
			if i != j:
				suma += prox_matrix[(i, j)]
		medias.append(suma / (ln_grupo - 1))	

	# El centro será el que tenga mayor media, obtenemos su indice en el grupo
	centro = np.argmax(medias)	

	# Calculamos la mediana de las distancias respecto al centro (mayor media)
	ordenados = []
	for j in range(len(grupo)):
		if centro != j:
			# Inserción ordenada de los valores
			bisect.insort(ordenados, prox_matrix[(centro, j)])
	mediana = np.median(ordenados)

	# Calculamos la desviación típica del centro
	suma = 0
	for j in range(ln_grupo):
		if centro != j:
			suma =  suma + (prox_matrix[(centro, j)] - medias[centro])**2
	desv_tipica = math.sqrt(suma / (ln_grupo - 1))

	return (grupo[centro], medias[centro], desv_tipica, mediana)

# Dado un tema calcular su resumen calculando la matriz, haciendo la suma de componentes 
def calculaResumenGrupo(grupo):
	ln_grupo = len(grupo)
	ot = []
	for t in grupo:
		ot.append(re.sub(r'[^\w\s]','',tweets[t]['text3']))

	tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,stop_words=stopwords.words(config['lang']),# max_df=0.9,
                                    min_df=0.0001, norm='l2',lowercase=True)
	matrix = tfidf_vectorizer.fit_transform(ot)
	# Más palabras comunes, no dando ventaja a las cortas
	l = [len(word_tokenizer(j)) for j in ot]

	sums = np.sum(matrix, axis=1) 
	# Buscamos valor más pequeño
	r = [x/(y*1.0) for x, y in zip(sums, l)]
	ind_centro = r.index(np.min(r))
	centro = grupo[ind_centro]

	suma  = 0
	ordenados = []
	# Proximidad media de los tweets respecto al centro
	for t in grupo:
		if t != centro:
			pr = similaridadTweets(centro, t)
			suma  +=  pr
			bisect.insort(ordenados, pr) 

	media = suma / (ln_grupo - 1)

	# Mediana
	mediana = np.median(ordenados)

	# Desviación típica de proximidades respecto al centro
	suma = 0
	for t in ordenados:
			suma =  suma + (t - media)**2
	desv_tipica = math.sqrt(suma / (ln_grupo - 1))


	return (centro, media, desv_tipica, mediana)

# Calcular el centro de todos los temas obtenidos en la ventana
def calcularCentros(grupos):
	print("Calculamos los centros de los temas de la ventana una vez aumentados")
	centros = []
	for g in grupos:
		centros.append(calculaCentroGrupo(g))
	return centros

# Calcular el resumen de todos los temas obtenidos en la ventana
def calcularResumenes(grupos):
	print("Calculamos los centros de los temas de la ventana una vez aumentados")
	centros = []
	for g in grupos:
		centros.append(calculaResumenGrupo(g))
	return centros

# Se eliminan de cada tema los tweets que estén a una distancia menor a m - 2s, la mayor diferencia aceptable respecto al centro
# Se devuelven los temas ya filtrados con sus centros recalculados
def compactar(grupos, centros):
	print("Tratamos de compactar los temas de la ventana, exigiendo que sean próximos al centro")
	grupos_compactados = []
	nuevos_centros = []
	eliminados = 0

	count = 0
	for g in grupos:
		elim_grupo, grupo_new, centro_new = compactaGrupo(g, centros[count])
		# Comprobamos que el grupo no haya quedado con un solo tweet
		if len(grupo_new) > 1:
			grupos_compactados.append(grupo_new)
			nuevos_centros.append(centro_new)
		eliminados += elim_grupo
		count +=1

	print("Eliminados tras compactar ", eliminados, " tweets")
	return grupos_compactados, nuevos_centros
	
# Dado un tema, lo compactamos, recalculando su centro en caso de ser necesario
# Salida  num_eliminados, grupo_compactado, nuevo_centro
def compactaGrupo(grupo, centro):
	maxdif = centro[1] - 2 * centro[2]
	eliminados = 0
	grupo_new = []
	for e in grupo:
		if similaridadTweets(centro[0],e) >= maxdif:   
			grupo_new.append(e)
		else:
			eliminados += 1
	if eliminados != 0:
		centro_new = calculaCentroGrupo(grupo_new)
	else:
		centro_new = centro
	
	return eliminados, grupo_new, centro_new 

# Devolvemos los temas de manera que cada tema se ordene por orden de proximidad al centro
def ordenaProximidadCentro(centros, grupos):
	print("Ordenamos los temas de la ventana, devolviendo los tweets ordenados por distancia al centro")
	count = 0
	cercanos = []
	for grupo in grupos:
		grupo_sort = ordenaProximidadCentroGrupo(centros[count], grupo)
		cercanos.append(grupo_sort)
		count +=1
	return cercanos

# Dado un tema, lo devolvemos ordenado por proximidad de los tweets al centro del tema
def ordenaProximidadCentroGrupo(centro, grupo):
	proximidades = []
	for elem in grupo:
		proximidades.append((elem, similaridadTweets(centro[0], elem)))

	ordenado = sorted(proximidades, key=lambda x: x[1], reverse=True)
	return [x[0] for x in ordenado] 

# Devuelve los X tweets más representativos (cercanos al centro)
def nucleoGrupo(centro, grupo):
	proximidades = []
	for elem in grupo:
		proximidades.append((elem, similaridadTweets(centro[0], elem)))

	ordenado = sorted(proximidades, key=lambda x: x[1], reverse=True)
	return [x[0] for x in ordenado][:config['tweets_nucleo']]

# Tratamos de fusionar temas dentro de una ventana, exigiendo que compartan el X% de los tweets
# o que sus nucleos (X tweets más centrales) se parezcan respecto a determinado umbral
def fusionInternaGrupos(grupos, centros):
	ok = False
	ic = 0
	while ic < len(grupos) and not ok:
		ix = ic + 1
		n1 = nucleoGrupo(centros[ic], grupos[ic])
		while ix < len(grupos) and not ok:
			n2 = nucleoGrupo(centros[ix], grupos[ix])
			interseccion = set(grupos[ic]) & set(grupos[ix])
			union = set(grupos[ic]) | set(grupos[ix])
			if len(interseccion) >= config['porcentaje_fusion_interna'] * len(union) or proximidadMediaNucleos(n1, n2) >= config['umbral_fusion_interna']:
				"""
				print("----------------", proximidadMediaNucleos(n1,n2))
				for e in grupos[ic]:
					print(ic, tweets[e]['text3'])
				print("----------------")
				for e in grupos[ix]:
					print(ix, tweets[e]['text3'])
				"""
				ok = True
				ind = (ic, ix)	
			ix +=1
		ic +=1
	# Hemos encontrado temas fusionables, caso recursivo
	if ok:
		# El grupo fusionado incluye todos los tweets de los dos, eliminando repetidos y compactando
		nuevo_elim, nuevo_grupo, nuevo_centro = compactaGrupo(list(union), calculaCentroGrupo(list(union)))
		grupos[ind[0]] = nuevo_grupo
		centros[ind[0]] = nuevo_centro
		del grupos[ind[1]]
		del centros[ind[1]]
		return fusionInternaGrupos(grupos, centros)
	else:
		print("Temas después de fusionar internamente en la ventana ", len(grupos))
		return grupos, centros	

################################### MOSTRAR Y ALMACENAR RESULTADOS ###############################

# Almacenar los temas en ficheros
def save_temas(grupos, centros, ruta, num_ventana):
	i = 0
	with open(ruta, 'a') as f:
		media_ventana, desv_ventana = frecVentana(i)
		f.write("Inicio ventana: " + str(info_ventanas[num_ventana][0]) + "\n")
		f.write("Fin ventana: " + str(info_ventanas[num_ventana][1]) + "\n")
		f.write("Fecuencia media ventana: "+ str(media_ventana)+  "\n" )
		f.write("Desviación típica de frecuencias ventana:"+ str(desv_ventana)+"\n")
		for g in grupos:
			num_tweets, frec_grupo, long_media, centro, tiempo_vida, num_retweets, frec_absoluta, similaridad_media, tweets_diferentes = evaluaGrupo(g)
			f.write("--------------------TEMA " + str(i+1) + "-------------------\n")
			f.write("\tTWEETS: \n")
			maxdate = datetime.datetime(2001, 1, 1, 0, 0)
			mindate = datetime.datetime.now()
			for tw in g:
				# str(tweets[tw]['id']) + 
				info = " (" + str(tweets[tw]['created_at'])+ ") --> " + str(tweets[tw]['user']['screen_name']) + ": " +  str(tweets[tw]['text'].replace("\n", " ").replace(",", " ").replace("\t", " ").replace("\r", " ")) + "\n"
				f.write(info)
				time = tweets[tw]['created_at']
				if time < mindate:
					mindate = time
				if time > maxdate:
					maxdate = time
			f.write("\tHora inicio: " + str(mindate)  + "\n")
			f.write("\tHora fin: " +  str(maxdate)  + "\n")
			# Centro: Mayor proximidad media a todos
			f.write("\tCentro grupo: " +  str(tweets[centros[i][0]]['text'].replace("\n", " ").replace(",", " ").replace("\t", " ").replace("\r", " "))  + "\n")
			# Resumen: TF-IDF
			textoResumen = calculaResumenGrupo(g)[0]
			f.write("\tResumen grupo: " + str(tweets[textoResumen]['text'].replace("\n", " ").replace(",", " ").replace("\t", " ").replace("\r", " "))  + "\n" )
			centroTemp = centroTemporal(g)
			f.write("\tCentro temporal: " + str(centroTemp.strftime('%Y-%m-%dT%H:%M:%S')) + "\n")
			f.write("Factores de evaluación:\n")
			f.write("\tTamaño del grupo: " + str(num_tweets) + "\n")
			f.write("\tTiempo de vida (segundos): " + str(tiempo_vida) + "\n")
			f.write("\tFrecuencia de grupo: " + str(frec_grupo) + "\n")
			f.write("\tLongitud media de los tweets (palabras): " + str(long_media) + "\n")
			f.write("\tProximidad media al centro:" + str(centro) + "\n")
			f.write("\tNúmero de RT: " + str(num_retweets) + "\n")
			f.write("\tFrecuencia absoluta: " + str(frec_absoluta) + "\n")
			f.write("\tSimilaridad media: " + str(similaridad_media) + "\n")
			f.write("\tTweets diferentes: " + str(tweets_diferentes) + "\n")
			f.write("\tCentro timestamp:" + str(centroTemp.replace(tzinfo=datetime.timezone.utc).timestamp()) + "\n")
			f.write("-------------------- FIN TEMA " + str(i+1) + "-------------------\n")
			i += 1
	f.close()

# Almacenar la información de las frecuencias de twitteo del periodo
def saveInfoTwitteo(info, ruta):
	with open(ruta, 'w') as f:
		for minute,count in info_frecuencias_tweets.items():
			print(minute, count)
			f.write(minute + "," + str(count) + "\n")
	f.close()

################################### REPRESENTACIÓN GRÁFICA INICIAL ###############################


def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([info_grupos[n][0] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

def showThemeInfo(ind):
    lst = []
    for n in ind['ind']:
        tweets = info_grupos[n][1]['tweets']
        inicio =  info_grupos[n][1]['inicio']
        fin =  info_grupos[n][1]['fin']
        centro = info_grupos[n][1]['centro']
        resumen =  info_grupos[n][1]['resumen']
        centro_temporal = info_grupos[n][1]['centro_temporal']
        lst.append("INICIO TEMA: " + str(inicio))
        lst.append("FIN TEMA: " + str(fin))
        lst.append("TWEET CENTRO: " + str(centro))
        lst.append("TWEET RESUMEN: " + str(resumen))
        lst.append("CENTRO TEMPORAL DEL TEMA: " + str(centro_temporal))
        lst.append("Número de tweets: " + str(len(tweets)))
        lst.append("Tiempo de vida: " + str(info_grupos[n][1]['tiempo_vida']) + " segundos.")
        lst.append("Longitud media de los tweets: " + str(info_grupos[n][1]['long_media']) + " palabras.")
        lst.append("Número de RT total: " + str(info_grupos[n][1]['num_retweets']))
        lst.append("\n")
        for t in tweets:
            lst.append(t)
        lst.append("\n")
        lst.append("\n")

    gui = ScrollMessageBox(ind['ind'], lst, None)
    gui.exec_()

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    if event.inaxes == ax and event.button == 3:
        cont, ind = sc.contains(event)
        showThemeInfo(ind)

# ################################## ########################## ############################# ##################### 
# ################################## ########################## ############################# ##################### 
# ################################## ########################## ############################# ##################### 
# ################################## ########################## ############################# ##################### 

if len(sys.argv) != 4:
	print('Number of arguments:', len(sys.argv), 'arguments.')
	print("Uso: python tweet-analysis.py <ruta_trabajo> <nombre_bd> <puerto>")
	sys.exit(0)

# Conexión con MongoDB
puerto = int(sys.argv[3])
con = pymongo.MongoClient('localhost',puerto)
# Nombre de la base de datos pertinente
nombre_bd = sys.argv[2]
db = con[nombre_bd]
# Ruta a la carpeta de trabajo donde se encontrará también el fichero de configuración config.json
ruta = sys.argv[1]

horaInicio, horaFin, status = main()
config = load_config(ruta + 'config.json')

if status == False: 
	horaInicio = datetime.datetime(config['start_year'], config['start_month'], config['start_day'], config['start_hour'], config['start_minute'])
	horaFin = datetime.datetime(config['end_year'], config['end_month'], config['end_day'], config['end_hour'], config['end_minute'])

print("Analizando tweets desde ", horaInicio, " hasta ", horaFin)
# Si el evento es de más de 5 horas lo tomaremos como evento largo, que tiene unas restricciones de filtrado diferentes
if (horaFin - horaInicio).total_seconds() > 18000:
	print("Evento largo")
	tipo_evento = 1
else:
	print("Evento corto")
	tipo_evento = 0

fusion = False
# Almacenamos la informacion de frecuencia de twitteo por minuto en el tiempo analizado
info_frecuencias_tweets = informacion_twitteo(horaInicio, horaFin)
saveInfoTwitteo(info_frecuencias_tweets, ruta + "info_frecuencias.csv")

# Cargamos los tweets con la hora de inicio indicada, según el fichero de configuración
tweets, num_ventanas, info_ventanas = carga_tweets(horaInicio, horaFin)
print("Total de ventanas: ", num_ventanas , " de longitud de aproximadamente", config['tweets_ventana'], "tweets cada una. TOTAL TWEETS:", len(tweets))
# Calculamos la matriz tf-idf
tfidf_matrix = tfidf_calc()
similarity_matrix = ""
num_ventana = 0

# grupos --> Almacena los grupos de cada ventana, en cada posicion una ventana, con su lista de grupos
# centros_grupos --> Almacena los centros de los grupos formados
grupos, centros_grupos = procesa_tweets()
num_ventana = 0
diccionarioInd = matrizSimilaridadesGlobal(grupos)
fusion = True
# Una vez tenemos los grupos de cada hora, tratamos de fusionar parejas de grupos de distintas ventanas
grupos, centros_grupos = fusionExterna(grupos, centros_grupos)

# Mostramos los resultados finales
print("Almacenamos los temas finales")
for i in range(len(grupos)):
	save_temas(grupos[i], centros_grupos[i], ruta + "salida.txt", i)

# Representamos gráficamente los resultados
x_frec = []
y_frec = []
for k,v in info_frecuencias_tweets.items():
	x_frec.append(datetime.datetime.strptime(k, '%Y-%m-%dT%H:%M'))
	y_frec.append(v)
fig, ax = plt.subplots()
plt.plot(x_frec, y_frec)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn
total = 0
for grs in grupos:
	total += len(grs)
		
print("TOTAL DE TEMAS: ", total)
c = np.random.randint(1,5,size=total)

plt.title("Tweets por minuto " +str(horaInicio) + " a "+ str(horaFin) + " y temas detectados")   # Establece el título del gráfico
plt.xlabel("minuto")   # Establece el título del eje x
plt.ylabel("tweets")   # Establece el título del eje y
# Dibujar los centros temporales de grupos
x = []
y = []
info_grupos = []
for grs in grupos:
	for g in grs:
		aux = centroTemporal(g)
		ct = aux.strftime('%Y-%m-%dT%H:%M:%S')[-8:]
		minuto = ct[:5]
		segundos = ct[-2:]
		x.append(aux)
		y.append(len(g))
		resumen = calculaResumenGrupo(g)
		info = str(len(g)) + " - " + tweets[resumen[0]]['text']
		info_grupo = {}
		num_tweets, frec_grupo, long_media, centro_grupo, tiempo_vida, num_retweets, frec_absoluta, similaridad_media, tweets_diferentes = evaluaGrupo(g)
		maxdate = datetime.datetime(2001, 1, 1, 0, 0)
		mindate = datetime.datetime.now()
		tweets_grupo = []
		for tw in g:
			info_tweet = " (" + str(tweets[tw]['created_at'])+ ") --> " + str(tweets[tw]['user']['screen_name']) + ": " +  str(tweets[tw]['text'])
			tweets_grupo.append(info_tweet)
			time = tweets[tw]['created_at']
			if time < mindate:
				mindate = time
			if time > maxdate:
				maxdate = time
		info_grupo["inicio"] = str(mindate)
		info_grupo["fin"] = str(maxdate)
		info_grupo["resumen"] = tweets[resumen[0]]['text']
		info_grupo["centro"] = tweets[centro_grupo]['text']
		info_grupo["tweets"] = tweets_grupo
		info_grupo["num_tweets"] = num_tweets
		info_grupo["long_media"] = long_media
		info_grupo["tiempo_vida"] = tiempo_vida
		info_grupo["num_retweets"] = num_retweets
		info_grupo["centro_temporal"] = str(centroTemporal(g))
		res = (info, info_grupo)
		info_grupos.append(res)
sc = plt.scatter(x, y, cmap=cmap, norm=norm, alpha=0.5, s=100)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)
plt.locator_params(axis='x', nbins=20)
plt.xlim(horaInicio, horaFin)
fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
