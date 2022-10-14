import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

img1= Image.open('Streamlit/vocales.png')
img2 = Image.open('Streamlit/figuras.png')
img3= Image.open('Streamlit/plantilla.png')
img4= Image.open('Streamlit/3.png')
img5= Image.open('Streamlit/4.png')
img6= Image.open('Streamlit/5.png')
img7= Image.open('Streamlit/5_1.png')
img8= Image.open('Streamlit/6.png')
img2_1= Image.open('Streamlit/2_1.png')
img3_1= Image.open('Streamlit/3_1.png')
img4_1= Image.open('Streamlit/4_1.png')
img5_1= Image.open('Streamlit/5_1.png')
img6_1= Image.open('Streamlit/6_1.png')

st.write('''
        # Proyecto 2 - INFO 1128
        Diego Alveal, Patricio Arratia
        ''')
colT1,colT2 = st.columns([1,8])
with colT2:
        status= st.selectbox('Selecciona la pregunta:',('Pregunta 1','Pregunta 2','Pregunta 3','Pregunta 4','Pregunta 5','Pregunta 6'))
        if(status=='Pregunta 1'):
                st.write('1. Dada la siguiente figura obtenga los momentos invariantes de Hu (H1-H7) y la Tabla Resumen. Programe un script en Python que obtenga los Hu(i=1..7) de cada una de las vocales. Puede utilizar CV2.')
                st.image(img1)
                code=('''
from math import copysign,log10
import cv2

#Cargamos la imagen y la transformamos a escala de grises
img = cv2.imread('vocales.png',cv2.IMREAD_GRAYSCALE)

#Recortamos las imagenes por cada letra de las vocales y se agregan a una lista
imgvoc = []
imgvoc.append(img[0:700,0:120])
imgvoc.append(img[0:700,120:235])
imgvoc.append(img[0:700,224:320])
imgvoc.append(img[0:900,320:420])
imgvoc.append(img[0:900,430:550])

#Dependiendo del valor entregado a la funcion, se imprimiran los momentos de Hu de cierta vocal
def Letra(n):
    #Creamos un limite para poder tranfromar la imagen en binaria
    _,img= cv2.threshold(imgvoc[n],128,255, cv2.THRESH_BINARY)

    #Momento
    momento = cv2.moments(img)

    #Realizamos el Momento de Hu el cual tiene un rango muy amplio
    MomentoHu = cv2.HuMoments(momento)

    #Creamos un ciclo en el cual dejara los valores en escala comparable y se imprimen por pantalla
    for i in range(0,7):
        MomentoHu[i] = -1* ((copysign(1.0, MomentoHu[i])) * log10(abs(MomentoHu[i])))
        print('log(H'+str(1+i)+')',MomentoHu[i])
    print('')

#Se escriben los momentos de Hu de cada letra
print('Momentos de Hu de la letra A')
Letra(0)

print('Momentos de Hu de la letra E')
Letra(1)

print('Momentos de Hu de la letra I')
Letra(2)

print('Momentos de Hu de la letra O')
Letra(3)

print('Momentos de Hu de la letra U')
Letra(4)''')
                st.code(code,language='python')
                datos=[{'Momentos de Hu de la letra A': 2.77830974, 'Momentos de Hu de la letra E': 2.86291884, 'Momentos de Hu de la letra I': 2.95491292,'Momentos de Hu de la letra O':2.75387676,'Momentos de Hu de la letra U':2.77946426},
                        {'Momentos de Hu de la letra A': 6.5706167, 'Momentos de Hu de la letra E': 6.70057832, 'Momentos de Hu de la letra I': 7.61838186,'Momentos de Hu de la letra O':7.16440321,'Momentos de Hu de la letra U':6.64003929},
                        {'Momentos de Hu de la letra A': 10.35482723, 'Momentos de Hu de la letra E': 11.02737094, 'Momentos de Hu de la letra I': 11.78272893,'Momentos de Hu de la letra O':10.95607962,'Momentos de Hu de la letra U':10.90729631},
                        {'Momentos de Hu de la letra A': 10.50378482, 'Momentos de Hu de la letra E': 11.59710193, 'Momentos de Hu de la letra I': 11.76680958,'Momentos de Hu de la letra O':10.38736457,'Momentos de Hu de la letra U':10.42598156},
                        {'Momentos de Hu de la letra A': 20.93311608, 'Momentos de Hu de la letra E': -22.95178118, 'Momentos de Hu de la letra I': -24.48425653,'Momentos de Hu de la letra O':21.36119473,'Momentos de Hu de la letra U':21.77007561},
                        {'Momentos de Hu de la letra A': 13.94454143, 'Momentos de Hu de la letra E': -14.95779045, 'Momentos de Hu de la letra I': -15.86175456,'Momentos de Hu de la letra O':-14.94348371,'Momentos de Hu de la letra U':-13.96408849},
                        {'Momentos de Hu de la letra A': 22.90050784, 'Momentos de Hu de la letra E': -23.2846893, 'Momentos de Hu de la letra I': -23.54442487,'Momentos de Hu de la letra O':-21.12119787,'Momentos de Hu de la letra U':-21.1024293}]
                df=pd.DataFrame(datos,index= ('Log(H1)','Log(H2)','Log(H3)','Log(H4)','Log(H5)','Log(H6)','Log(H7)'))
                if st.checkbox("Mostrar Resultados"):
                        st.table(df)
        elif(status=='Pregunta 2'):
                st.write('2. Coloque cada una las siguientes imágenes en la posición señalada dentro de la plantilla de salida. Debe redimensionar y rotar las figuras. Programe un script en Python + Pygame + PIL.')
                st.image(img2)
                st.image(img3)
                code=('''
from  PIL import Image

#Cargamos la imagen de figuras y la convertimos en RGBA para obtener el canal alpha
img=Image.open('figuras.png').convert('RGBA')
#Cargamos la imagen de plantilla
imgplantilla = Image.open('plantilla.png')


#Cortamos la imagen
img1=img.crop((10, 10, 210, 220))
#Cambiamos el tamano de la imagen
img1_tamano= img1.resize((116, 116))
#Rotamos la imagen y la expandimos para arreglar esos recortes en negro
img1_rotada= img1_tamano.rotate(75, expand=True)
#Pegamos la imagen en la plantilla en las cordenadas dadas
imgplantilla.paste(img1_rotada,(10,16))


img2=img.crop((242, 10, 484, 207))
img2_tamano= img2.resize((116, 116))
img2_rotada= img2_tamano.rotate(100, expand=True)
imgplantilla.paste(img2_rotada,(198,16))


img3=img.crop((503, 23, 723, 198))
img3_tamano= img3.resize((116, 116))
img3_rotada= img3_tamano.rotate(45, expand=True)
imgplantilla.paste(img3_rotada,(378,8))


img4=img.crop((733, 19, 980, 202))
img4_tamano= img4.resize((116, 116))
img4_rotada= img4_tamano.rotate(120, expand=True)
imgplantilla.paste(img4_rotada,(590,10))

#guardamos la imagen
imgplantilla= imgplantilla.save('2_1.png')''')
                st.code(code,language='python')
                if st.checkbox("Mostrar Resultados"):
                        st.image(img2_1)
        elif(status=='Pregunta 3'):
                st.write('3.  Aplique Least Square Polymonial mediante poly1d() y polyfit(). Utilice f1.npy y f2.npy para obtener el siguiente gráfico. Utilice x = np.arange(start=1,stop=50,step=1')
                st.image(img4)
                code=('''
import matplotlib.pyplot as plt
import numpy as np
from numpy import *

#Cargamos los archivos numpy
f1 = np.load("f1.npy")
f2 = np.load("f2.npy")

#Genermaos un conjunto de numeros entre el valor de inicio y uno final especificando un incremento entre los valores
#al no ingresar el ultimo dato se sobreentiende que el incremento es 1
x= np.arange(start=1,stop=50)
#Sacamos los coeficientes de un polinomio p(x) de grado n con los datos de f1.npy y f2.npy
pol1= np.polyfit(x,f1,1)
pol2=np.polyfit(x,f2,2)

#Creamos una figura 
plt.figure(figsize=(10,4))
plt.title("Least Square Polymonial")
plt.ylabel('F(x)')
plt.xlabel('T(s)')
plt.plot(x,f1,'o')
plt.plot(x,f2,'o')
plt.plot(x,polyval(pol1,x))
plt.plot(x,polyval(pol2,x))
plt.grid()
plt.show()''')
                st.code(code,language='python')
                if st.checkbox("Mostrar Resultados"):
                        st.image(img3_1)
        elif(status=='Pregunta 4'):
                st.write('4.  Dada la señal signal.npy aplique los filtros Median y Wiener para obtener el siguiente gráfico. Investigue sobre el módulos scipy.signal.')
                st.image(img5)
                code=('''
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.signal import wiener
import numpy as np

#Cargamos el archivo numpy
signals= np.load('signal.npy')

#Realizamos un filtro de mediana en una matriz N-Dimensional
Media_Signal=medfilt(signals)

#Realizamos un filtro wieneer en una matriz N-Dimensional
Wiener_Signal = wiener(signals)

#Creamos la figura
plt.figure(figsize=(10,3))
#Asignamos el nombre de la grafica
plt.title("Signal Filter")
#Generamos las etiquetas para el eje Y, X
plt.ylabel("Signal")
plt.xlabel("T(s)")
#Ingresamos las senales para graficarlas y le asignamos sus nombres
plt.plot(signals, label='Signal Original')
plt.plot(Media_Signal,label='Median Filter')
plt.plot(Wiener_Signal, label='Wiener Filter')
#Cramos el recuadro para que aparezcan los nombres asignando
plt.legend()
plt.grid()
#Guardamos la figura
plt.show()''')
                st.code(code,language='python')
                if st.checkbox("Mostrar Resultados"):
                        st.image(img4_1)
        elif(status=='Pregunta 5'):
                st.write('5. Separe la tendencia de la señal. Obtenga un gráfico similar. Complete el código.')
                st.image(img6)
                st.image(img7)
                code=(''' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

#Agregamos a una array que comienze con 0, termina en 5, con numero de valores de 100
t = np.linspace(0,5,100)

#Se crea una matriz de forma especifica y la llena con numero aleatorio que es de parte normal
x= t +np.random.normal(size=100)

#Elimina la tendendia lineal a lo largo del eje de los datos
x_Tendencia=signal.detrend(x)

#Creamos la figura  
plt.figure(figsize=(10,3)) 
plt.plot(t,x,label='Serie con tendencia')
plt.plot(t,x_Tendencia,label='Serie sin Tendencia')
plt.legend()
plt.grid()
#Guardamos la figura
plt.show()''')
                st.code(code,language='python')
                if st.checkbox("Mostrar Resultados"):
                        st.image(img5_1)
        elif(status=='Pregunta 6'):
                st.write('6.  Obtenga la Interpolación de Chebyshev desde cheby.npy. ¿Qué conclusiones obtiene? ¿Escriba el polinomio con sus coeficientes.')        
                st.image(img8)
                code=('''
import numpy as np
from numpy import polynomial as P
from scipy import linalg
import matplotlib.pyplot as plt

#Cargamos el archivo cheby.npy
cheby=np.load('cheby.npy')
#Separamos por ejes
x=cheby[0]
y=cheby[1]
#Se define el grado
deg=len(x)-1
#Se usa chebyshev para el calculo de la interpolacion y se resuelve la ecuacion lineal entre el polinomio A y el eje y
A= P.chebyshev.chebvander(x, deg)
c = linalg.solve(A, y)
#Se guarda en una variable la solucion a la ecuacion para su futuro uso en un polinomio
f = P.Chebyshev(c)
#Sacamos 100 numeros aleatorios entre los intervalos de x.min, x.max 
xx = np.linspace(x.min(), x.max(), 100)

#Creamos la figura
plt.figure(figsize=(10,3))
#Usando la solucion almacenada en f, se muestra la interpolacion de chebyshev utilizando los valores del eje x
plt.plot(xx, f(xx), '-',label='Interpolation  chebyshave')
plt.plot(x, y,'o', label='Data puntos')
plt.grid()
plt.xlabel('T(s)')
plt.ylabel('F(x)')
plt.legend()
plt.xticks(range(11))
#Guardamos la figura
plt.show()''')
                st.code(code,language='python')
                if st.checkbox("Mostrar Resultados"):
                        st.image(img6_1)
