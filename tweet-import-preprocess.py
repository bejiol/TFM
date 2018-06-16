from subprocess import call
import os
import sys
import time

if len(sys.argv) != 4:
	print('Number of arguments:', len(sys.argv), 'arguments.')
	print("Uso: python tweet-preprocess.py <ruta_carpeta> <nombre_bd> <ruta_tweets>")
	sys.exit(0)

ruta_carpeta = sys.argv[1]
nombre_bd = sys.argv[2]
ruta_tweets = sys.argv[3]

# Crear la carpeta donde almacenar los datos si no existe ya
if not os.path.exists(ruta_carpeta):
	print("Creada la carpeta que contendrá la base de datos: " + ruta_carpeta)
	os.makedirs(ruta_carpeta)
else:
	text = input("La carpeta " + ruta_carpeta +" existe, desea borrar su contenido antes S/N? ")  # Python 3
	if text == "S" or text == "s":
		os.system("rm -rf " + ruta_carpeta)
		os.makedirs(ruta_carpeta)
	elif text =="N" or text=="n":
		print("Se debe borrar primero la carpeta.")
		sys.exit(0)
	else:
		print("Opción incorrecta.")
		sys.exit(0)

# Lanzar MongoDB sobre <ruta_carpeta>
os.system("mongod --fork --logpath "+  ruta_carpeta+"/mongod-temp.log --dbpath=" + ruta_carpeta)
time.sleep(2)

# Importar los tweets de <ruta_tweets> a la base de datos <nombre_bd> que se almacenará en <ruta_carpeta>
call(["mongoimport", "--db", nombre_bd, "--collection", "tweets", ruta_tweets])

# Ejecutar .js con las operaciones de preprocesado
os.system("mongo --shell " + nombre_bd + " preprocesado.js")
