import numpy as np 
import cv2 
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)
cap.set(3, 300)


while True:

# ----------------------- FILTROS LINEALES ------------------------
    
  ret,frame = cap.read()
  
  if ret==True:

      cv2.imshow('Video', frame)

####################### RUIDO GAUSSIANO #####################
      
      kernel = np.ones((5,5),np.float32)/25
      Gaus = cv2.filter2D(frame,-1,kernel)

      cv2.imshow('Filtro Gaussiano', Gaus)

####################### BLUR #####################

      blur = cv2.blur(frame,(1,1))
      cv2.imshow('Filtro Blur', blur)
      
####################### GAUSSIANO BLUR #####################

      Gausblur = cv2.GaussianBlur(frame,(5,5),0)

      cv2.imshow('Filtro Gaussiano Blur', Gausblur)

####################### DENOISING COLORED #####################

      denoising = cv2.fastNlMeansDenoisingColored(frame,None,3,3,1,1)

      cv2.imshow('Filtro Denoising', denoising)


#~~~~~~~~~~~~~~~~~~~~~~~~ IMAGEN GRIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

      #cv2.imshow('Escala de Grises', gray)

####################### SAL Y PIMIENTA #####################

      prob = 0.3

      row = gray.shape[0]
      col = gray.shape[1]

      s_vs_p = 0.5
      ruido = np.copy(gray)

      num_salt = np.ceil(prob * gray.size * s_vs_p)

      coords = [np.random.randint(0,i-1,int(num_salt))for i in gray.shape]
      ruido[tuple(coords)] = 1

      num_pepper = np.ceil(prob * gray.size * (1 - s_vs_p))
      coords = [np.random.randint(0,i -1,int (num_pepper))for i in gray.shape]
      ruido[tuple(coords)] = 0

      cv2.imshow('Ruido',ruido)

      blurr = cv2.medianBlur(ruido,3)

      cv2.imshow('Median Blur',blurr)
      
# ------------------------- FILTROS MORFOLOGICOS -------------------------------

      ker = np.ones((1,5),np.uint8)
      
####################### EROSION #####################

      erosion = cv2.erode(frame,ker,iterations = 1)

      cv2.imshow('Erosion',erosion)

####################### DILATACION #####################

      dilatacion = cv2.dilate(frame,ker,iterations = 1)
      cv2.imshow('Dilatacion',dilatacion)

####################### APERTURA #####################

      apertura = cv2.morphologyEx(frame, cv2.MORPH_OPEN, ker)
      cv2.imshow('Apertura',apertura)

####################### CIERRE #####################

      cierre = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, ker)   
      cv2.imshow('Cierre',cierre)

####################### GRADIENTE MORFOLOFICO #####################

      gradiente = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, ker)  
      cv2.imshow('Gradiente morfologico',gradiente)
    
      if cv2.waitKey(30) & 0xFF == ord('m'):

          break

        


    
cap.release()
cv2.destroyAllWindows()
