import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.sparse as sparse
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import bisect
import math
import datetime, time
import sys
from scroll import ScrollMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def loadNews(file):
  grupos = []
  parameters = 10
  state = 0
  count_grupos = 0
  with open(file, encoding="utf-8") as f:
    for l in f:
      l2 = l[:-1]
      if state ==1:
        if l2.startswith(" ("): # Es un tweet
          hora = l2.split(")")[0][2:]
          user = l2.split("> ")[1].split(":")[0]
          texto = l2.split("> ")[1].split(":")[1:]
          texto = ":".join(texto)
          tweet = {}
          tweet['hora'] = hora
          tweet['user'] = user
          tweet['texto'] = texto
          grupos[count_grupos]['tweets'].append(tweet)
        if l2.startswith("\tHora inicio:"): # Es hora inicio
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['inicio'] = cadena
        if l2.startswith("\tHora fin:"): # Es hora fin
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['fin'] = cadena
        if l2.startswith("\tCentro grupo:"): # Es centro del tema
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['centro'] = cadena
        if l2.startswith("\tResumen grupo:"): # Es resumen del tema
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['resumen'] = cadena
        if l2.startswith("\tCentro temporal:"): # Es centro temporal
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['centro_temporal'] = cadena
      if state ==2:
        if i<parameters: # todavía es válido
            i = i+1
            index = l2.index(':') # buscamos el valor
            if index!=-1:
                s+=l2[index+1:]+"," # lo añadimos a s
            else:
                print("No encuentro : en ",l2)
        else: # Fin de los factores
            factores = s[:-1]
            grupos[count_grupos]['factores'] = factores
            s = ""
            i = 0
            state = 0
            count_grupos +=1
      elif l2.startswith("\tTWEETS:"): # Empieza un nuevo grupo
        state = 1 # Tweets read state
        grupo = dict()
        grupo['tweets'] = []
        grupo['factores'] = ""
        grupo['inicio'] = ""
        grupo['fin'] = ""
        grupo['centro'] = ""
        grupo['resumen'] = ""
        grupo['centro_temporal'] = ""
        grupos.append(grupo)
        s = ""
        #print(l2)
      elif l2.startswith("Factores de evaluación:"):
        state = 2 # factores
        i = 0
        s = ""
  f.close()
  return grupos

def word_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('spanish')]
    return tokens

def tf_idfCalc(sentences):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                    stop_words=stopwords.words('spanish'),
                                    use_idf=False, 
                                    norm='l2',
)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    return tfidf_matrix

def extraeCentros(tfidf_matrix, grupos_indices):
    indices_internos_centros = []
    sentences_centros = []
    ind_centro, ind_interno =  calculaCentroGrupo(grupos_indices[0])
    indices_internos_centros.append(ind_interno)
    sentences_centros.append(sentences[ind_centro])
    print("Centro tema ", 0 , ": ", ind_centro , sentences[ind_centro])
    new = tfidf_matrix[ind_centro]
    for i in range(1, len(grupos_indices)):
        ind_centro, ind_interno =  calculaCentroGrupo(grupos_indices[i])
        indices_internos_centros.append(ind_interno)
        sentences_centros.append(sentences[ind_centro])
        print("Centro tema ", i , ": ", ind_centro , sentences[ind_centro])
        new = sparse.vstack((new, tfidf_matrix[ind_centro]))
    return new, sentences_centros, indices_internos_centros


def palabras_clave(cluster):
    wordcount = {}
    for e in cluster:
        gr = grupos[e]
        for t in gr['tweets']:
            # Palabras al diccionario de cada grupo, para calcular las más comunes no stopwords
            for word in re.sub(r'[^\w\s]','',t['texto'].lower()).split():
                if word not in  stopwords.words("spanish"):
                    if word not in wordcount:
                        wordcount[word] = 1
                    else:
                        wordcount[word] += 1    
        # Palabras más comunes
        word_counter = collections.Counter(wordcount)
        line = ""
        for word, times in word_counter.most_common(10):
            line =  line + word + ": " + str(times) + "| "
    return line


def cluster_sentences_minibatch(tfidf_matrix, nb_of_clusters=5):
    kmeans = MiniBatchKMeans(n_clusters=nb_of_clusters, init='k-means++', n_init=1,init_size=1000, batch_size=1000)
    kmeans.fit(tfidf_matrix)
    s = metrics.silhouette_score(tfidf_matrix, kmeans.labels_, metric='euclidean')
    ch = metrics.calinski_harabaz_score(tfidf_matrix.toarray(), kmeans.labels_)
    print("K=", nb_of_clusters," s=" , s, " ch=", ch)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
    return dict(clusters), tfidf_matrix, kmeans, s, ch    


def similaridadTweets(tw1, tw2):
    text_sim = similarity_matrix[tw1, tw2]
    if text_sim >= 0.1:
        text_sim = math.log10(text_sim) + 1
    else:
        text_sim = 0
    return text_sim

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

    # El centro será el que tenga mayor media, obtenemos su indice en el tema
    centro = np.argmax(medias)  

    return grupo[centro], centro

def centroTemporal(grupo):
    minutos = []
    for e in grupo['tweets']:
        bisect.insort(minutos, datetime.datetime.strptime(e['hora'], '%Y-%m-%d %H:%M:%S'))
    ln = len(minutos)
    if ln % 2 == 0:
        m1 = minutos[ln // 2 - 1]
        m2 = minutos[ln // 2]
        mediavida = (m2 - m1).total_seconds() / 2
        m1 += datetime.timedelta(seconds = mediavida)
        mediana = m1
    else:
        mediana = minutos[(ln + 1 ) // 2 - 1]

    return mediana.replace(microsecond = 0)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join(["(Cluster " + str(kmeans.labels_[n]) + ") "+ centros_temporales[n] + ": "+ grupos[n]['resumen'] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))

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
        tweets = grupos[n]['tweets']
        inicio =  grupos[n]['inicio']
        fin =  grupos[n]['fin']
        centro = grupos[n]['centro']
        resumen =  grupos[n]['resumen']
        centro_temporal = grupos[n]['centro_temporal']
        factores = grupos[n]['factores'].split(",")
        lst.append("INICIO TEMA: " + str(inicio))
        lst.append("FIN TEMA: " + str(fin))
        lst.append("TWEET CENTRO: " + str(centro))
        lst.append("TWEET RESUMEN: " + str(resumen))
        lst.append("CENTRO TEMPORAL DEL TEMA: " + str(centro_temporal))
        lst.append("Número de tweets: " + str(len(tweets)))
        lst.append("Tiempo de vida: " + str(factores[1]) + " segundos.")
        lst.append("Longitud media de los tweets: " + str(factores[3]) + " palabras.")
        lst.append("Número de RT total: " + str(factores[5]))
        lst.append("\n")
        for t in tweets:
            lst.append(str(t['hora']) + "- (" + t['user'] + ") -"+   t['texto'])
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


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print("Uso: python tweet-textual-clustering.py <ruta_resultados_deteccion>")
        sys.exit(0)

    path = sys.argv[1]


    grupos = loadNews(path + "salida.txt")
    # nclusters = int(input("Enter a number of clusters: "))
    sentences =[]
    ind = 0
    grupos_indices = []
    centros_temporales = []
    for g in grupos:
        i = 0
        ind_group = []
        centros_temporales.append(g['centro_temporal'])
        for t in g['tweets']:
            ind_group.append(ind)
            texto = g['tweets'][i]['texto']
            sentences.append(texto)
            i +=1
            ind +=1
        grupos_indices.append(ind_group)

    x = []
    intertia = []
    silhouette_scores = []
    ch_scores = []
    tffss = []
    resultados = []
    centros_resultados = []
    labels_resultados = []


    num_initial_clusters = 3
    num_final_clusters = 21
    tfidf_matrix = tf_idfCalc(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output = False)
    tfidf_matrix, sentences_centers, indices_centros = extraeCentros(tfidf_matrix, grupos_indices)

    for nclusters in range(num_initial_clusters, min(num_final_clusters, len(grupos) - 1)):
        clusters, tfs, kmeans, s, ch = cluster_sentences_minibatch(tfidf_matrix, nclusters)
        x.append(nclusters)
        intertia.append(kmeans.inertia_)
        silhouette_scores.append(s)
        ch_scores.append(ch)
        resultados.append(clusters)
        centros_resultados.append(kmeans.cluster_centers_)
        tffss.append(tfs)
        labels_resultados.append(kmeans.labels_)

    best_s = np.argmax(silhouette_scores)
    best_ch = np.argmax(ch_scores)
    print("Mejor k según silhouette_score: ", best_s + num_initial_clusters )
    print("Mejor k según calinski_harabaz_score: ", best_ch + num_initial_clusters)
    true_k = max(best_s, best_ch)
    
    print("Guardando ", true_k + num_initial_clusters, "-kmeans...")
    clusters = resultados[true_k]
    centros_clusters = centros_resultados[true_k]
    kmeans.labels_ = labels_resultados[true_k]

    closest, _ = pairwise_distances_argmin_min(centros_clusters, tfidf_matrix)
    
    with open(path + "resumenes-temas-relacionados-"+ str(true_k + num_initial_clusters)+"-means.txt", 'w') as f:
        with open(path + "centros-temas-relacionados-"+ str(true_k + num_initial_clusters)+"-means.txt", 'w') as f2:
            for cluster in range(len(clusters)):
                f.write("Cluster " + str(cluster + 1) + ":  " + sentences_centers[closest[cluster]] + "\n")
                f2.write("Cluster " + str(cluster + 1) + ":  " +  sentences_centers[closest[cluster]] +  "\n")
                cl = []
                for i,sentence in enumerate(clusters[cluster]):
                    c = indices_centros[sentence]
                    cl.append((centroTemporal(grupos[sentence]), grupos[sentence]['resumen'], grupos[sentence]['tweets'][c]['texto'] ))
                cl_ordenados = sorted(cl, key=lambda x: x[0])
                i =0
                for e in cl_ordenados:
                    f.write("\t "+ str(i) +": " + str(e[0]) + "-->" + e[1] + "\n")
                    f2.write("\t "+ str(i) +": " + str(e[0]) + "-->" + e[2] + "\n")
                    i +=1
                f.write("\t" + palabras_clave(clusters[cluster]) + "\n")
                f2.write("\t" + palabras_clave(clusters[cluster]) + "\n")
    f.close() 
    f2.close()       

    k = 100
    tfs_reduced = TruncatedSVD(n_components=k, random_state=0).fit_transform(tffss[true_k])
    tfs_embedded = TSNE(n_components=2, perplexity=30, verbose=2, metric ="cosine").fit_transform(tfs_reduced)
    fig, ax = plt.subplots()
    plt.title("Relación textual")

    annot = ax.annotate("", xy=(0, 0), 
                bbox=dict(boxstyle="round", fc="w"),
                xytext=(0, 0), 
                  arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    cmap = plt.cm.Wistia
    norm = plt.Normalize(1,4)
    c = np.random.randint(1,5,size=len(sentences))
    # colors = cm.hsv(np.array(y) / float(max(np.array(y))))

    colors = ['red','blue','gold','green','cyan',
    'magenta', 'black', 'gray', 'sienna', 'orange',
    'darkblue', 'mediumslateblue', 'lime', 'lightcoral', 'khaki',
    'blueviolet', 'aqua', 'springgreen', 'orchid', 'salmon' ]  

    cluster_colors = []
    for l in kmeans.labels_:
        cluster_colors.append(colors[l])

    sc = plt.scatter(tfs_embedded[:, 0], tfs_embedded[:, 1], marker = "o", c = cluster_colors)
    # 3d --> sc = ax.scatter(tfs_embedded[:, 0], tfs_embedded[:, 1], tfs_embedded[:, 2], marker = "o", c = cluster_colors)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
