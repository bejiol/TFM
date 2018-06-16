import numpy as np
import csv
import datetime, time
import math
import matplotlib.pyplot as plt
import bisect
from collections import OrderedDict
import sys


# Carga los datos de los temas del archivo ruta + salida.txt
def loadNews(file):
  grupos = []
  values = []
  parameters = 10
  state = 0
  count_grupos = 0
  with open(file, encoding="utf-8") as f:
    for l in f:
      l2 = l[:-1]
      if state ==1:
        if l2.startswith(" ("): # Es un tweet
          hora = l2.split(")")[0][1:]
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
        if l2.startswith("\tCentro grupo:"): # Es centro del grupo
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['centro'] = cadena
        if l2.startswith("\tResumen grupo:"): # Es resumen del grupo
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['resumen'] = cadena
        if l2.startswith("\tCentro temporal:"): # Es hora inicio
          cadena = l2.split(":")[1:]
          cadena = ':'.join(cadena)
          grupos[count_grupos]['centro_temporal'] = cadena
      if state ==2:
        if i<parameters: # todavía es válido
            i = i+1
            index = l2.index(':') # buscamos el valor
            if index!=-1:
                s+=l2[index+1:]+"," # lo añadimos a s
                # Es el último, el centro en timestamp
                if i == parameters:
                  scts = math.ceil(float(l2[index+1:]))
                  dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=scts)
                  ts = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                  values.append((str(count_grupos), dt ))
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
  return grupos, values

# Carga la información temporal del evento
def loadFrecuencies(path):
  frec = OrderedDict() 
  with open(path, encoding="utf-8") as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
      if i == 0:
        inicio = datetime.datetime.strptime(row[0], '%Y-%m-%dT%H:%M')
      d = datetime.datetime.strptime(row[0], '%Y-%m-%dT%H:%M')
      frec[d] = int(row[1])
      fin = datetime.datetime.strptime(row[0], '%Y-%m-%dT%H:%M')
      i +=1
  f.close()      
  return frec, inicio, fin

# Calcula los grupos del intervalo [inicio, inicio + intervalo] a partir de los datos de valores
def gruposIntervalo(inicio, intervalo, valores):
  grupos = []
  fin_intervalo = inicio + datetime.timedelta(minutes = intervalo)
  i = 0
  while i < len(valores) and valores[i][1] < fin_intervalo:
    if valores[i][1] >= inicio:
      grupos.append(valores[i])
    i +=1
  return grupos, (inicio, fin_intervalo)

# Calcula un nivel dado, encontrando los grupos de temas de num_grupos de cada intervalo intervalo_minutos
def calculaNivel(inicio_evento, fin_evento, num_grupos, intervalo_minutos, grupos):
  grupos_nivel = []
  grupos_asignados = []
  restantes = []
  info = []
  while inicio_evento < fin_evento:
    g, intervalo = gruposIntervalo(inicio_evento, intervalo_minutos, grupos)
    if len(g) >= num_grupos:
      grupos_nivel.append(g)
      info.append((intervalo, len(g)))
      for e in g:
        grupos_asignados.append(e)
    inicio_evento += datetime.timedelta(minutes = intervalo_minutos)

  for g in grupos:
    if g not in grupos_asignados:
      restantes.append(g)

  return grupos_nivel, restantes, info

# Dado un conjunto de temas que componen un momento, calcula su centro temporal
def centroTemporal(grupos):
  minutos = []
  for e in grupos:
    bisect.insort(minutos, e[1])
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
 

# Almacena en el fichero de salida los niveles 0...N con sus respectivos momentos
def saveNiveles(ruta, grupos, niveles, info_niveles):
  with open(ruta, 'w') as f:
    f.write("Niveles con sus momentos ordenados cronológicamente de las noticias obtenidas de " + ruta + "\n\n")
    nvl = 0
    for nivel in niveles:
      if nivel != []:
        f.write("NIVEL " + str(nvl) + "("+ str(math.ceil(info_niveles[nvl]['tam'])) + " grupos en " + str(math.ceil(info_niveles[nvl]['intervalo'])) + " minutos)\n")
        momento = 1
        for mnt in nivel:
          f.write("\tMOMENTO " + str(momento) +"\n")
          for m in mnt:
            gr = grupos[int(m[0])]
            # f.write("\t\t"+ str(m[1]) + ":" + gr['tweets'][0]['texto'] + " (" + str(len(gr['tweets']))+" tweets)\n")
            f.write("\t\t"+ str(m[1]) + ":" + gr['resumen'] + " --- " + gr['centro'] + " (" + str(len(gr['tweets']))+" tweets)\n")
          momento +=1
      nvl +=1
  f.close()

# Almacena en el fichero de salida los momentos encontrados ordenados cronológicamente
def saveNivelesCronologicamente(ruta, grupos, niveles):
  momentos = []
  for nivel in niveles:
    if nivel != []:
      momento = 1
      for mnt in nivel:
        momentos.append(mnt)

  ordenados = sorted(momentos, key=lambda x: x[0][1])
  mnt = 0
  with open(ruta, 'w') as f:
    f.write("Momentos ordenados cronológicamente de las noticias obtenidas de " + ruta + "\n\n")
    for momento in ordenados:
      f.write("\tMOMENTO " + str(mnt) +"\n")
      for m in momento:
        gr = grupos[int(m[0])]
        # f.write("\t\t"+ str(m[1]) + ":" + gr['tweets'][0]['texto'] + " (" + str(len(gr['tweets']))+" tweets)\n")
        f.write("\t\t"+ str(m[1]) + ":" + gr['resumen'] + " --- " + gr['centro'] + " (" + str(len(gr['tweets']))+" tweets)\n")
      mnt +=1
  f.close()

# Almacena los temas encontrados ordenados cronológicamente
def saveCentrosCronologicamente(ruta, grupos, sorted_timestamp):
    with open(ruta, 'w') as f:
      f.write("Noticias obtenidas de " + ruta + "\n\n")
      for e in sorted_timestamp:
        gr = grupos[int(e[0])]
        # f.write("\t"+ str(e[1]) + ":" + gr['tweets'][0]['texto'] + " (" + str(len(gr['tweets']))+" tweets)\n")
        f.write("\t"+ str(e[1]) + ":" + gr['resumen'] + " --- " + gr['centro'] + " (" + str(len(gr['tweets']))+" tweets)\n")
    f.close()

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([anotaciones[n] for n in ind["ind"]]))
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

# Using a closure to access data. Ideally you'd use a "functor"-style class.
def formatter(**kwargs):
    dist = abs(np.array(x) - kwargs['x'])
    i = dist.argmin()
    return '\n'.join(anotaciones[i])


#################################################################################################################################                
#################################################################################################################################                
#################################################################################################################################                
#################################################################################################################################                
#################################################################################################################################                
#################################################################################################################################                
#################################################################################################################################                
#################################################################################################################################                


if len(sys.argv) != 3:
	print('Number of arguments:', len(sys.argv), 'arguments.')
	print("Uso: python tweet-time-clustering.py <ruta_resultados_deteccion> <num_niveles>")
	sys.exit(0)

path = sys.argv[1]
num_max_niveles = int(sys.argv[2])


grupos, values = loadNews(path + "salida.txt")
frecuencias, inicio_evento, fin_evento = loadFrecuencies(path + "info_frecuencias.csv")
print("Ruta: " + path)
print("Grupos leídos: ", len(grupos))


sorted_timestamp = sorted(values, key=lambda x: x[1].replace(tzinfo= datetime.timezone.utc).timestamp())
for e in sorted_timestamp:
  print(e[0], str(e[1]))


tiempo_total = (fin_evento - inicio_evento).total_seconds() / 60
total_grupos = len(values)
intervalo_minutos = 1
num_grupos = total_grupos
nivel = 0
niveles = []
info_niveles = []
info_intervalos = []

print(inicio_evento, fin_evento, tiempo_total)

t = tiempo_total**(1 / (num_max_niveles - 1))
g = num_grupos**(1 / (num_max_niveles - 1))

restantes_timestamp = sorted_timestamp
while nivel < num_max_niveles:
  print("-----------------------------------------------")
  print("Nivel", nivel, "busca" ,num_grupos, " temas en",  intervalo_minutos, "minutos")
  grupos_nivel, restantes_timestamp, info_intervalos_nivel = calculaNivel(inicio_evento, fin_evento, math.ceil(num_grupos), math.ceil(intervalo_minutos), restantes_timestamp)
  print("Quedan ", len(restantes_timestamp), " temas")    
  print("Momentos nivel " + str(nivel) + " --> " + str(len(grupos_nivel)))
  info = {}
  info['intervalo'] = math.ceil(intervalo_minutos)
  info['tam'] = num_grupos
  info_niveles.append(info)
  niveles.append(grupos_nivel)
  info_intervalos.append(info_intervalos_nivel)
  nivel +=1
  intervalo_minutos = t**nivel
  num_grupos = total_grupos / (g**nivel) 

saveCentrosCronologicamente(path + "sorted-centers.txt", grupos, sorted_timestamp)
saveNiveles(path + str(num_max_niveles) + "niveles-timestamp.txt", grupos, niveles, info_niveles)
saveNivelesCronologicamente(path + str(num_max_niveles) + "niveles-sorted-timestamp.txt", grupos, niveles)


fig, ax = plt.subplots()
plt.plot(frecuencias.keys(), frecuencias.values())
plt.title("Momentos temporales")
x = []
y = []
s = []
c = []
anotaciones = []
nivel = 0
for grupos_nivel in niveles:
  for e in grupos_nivel:
    cadena = "NIVEL " + str(nivel) + " (" + str(len(e)) + " temas en "  + str(info_niveles[nivel]['intervalo']) + " minutos)\n"
    suma = 0
    for m in e:
      cadena += " "+ grupos[int(m[0])]['centro_temporal'] +": " + grupos[int(m[0])]['resumen'] + " (" + str(len(grupos[int(m[0])]['tweets'])) + " tweets)\n"
      suma += len(grupos[int(m[0])]['tweets'])
    x.append(centroTemporal(e))
    y.append((suma / info_niveles[nivel]['intervalo']))
    s.append(suma)
    anotaciones.append(cadena)
  nivel +=1


for i in info_intervalos:
  for inf in i:
    print(inf)

medio = (fin_evento - inicio_evento).total_seconds() / 3
medio = inicio_evento + datetime.timedelta(seconds = medio)   
annot = ax.annotate(cadena, xy=(medio, 200), 
                bbox=dict(boxstyle="round", fc="w"),
                xytext=(medio, 200), 
                  arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)
cmap = plt.cm.Wistia
norm = plt.Normalize(1,4)
c = np.random.randint(1,num_max_niveles,size=len(x))

sc = plt.scatter(x, y, c=c, s = s,alpha=0.5, marker = "o")

plt.xlim(inicio_evento, fin_evento)
plt.tight_layout()
fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()

