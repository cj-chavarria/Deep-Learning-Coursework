#Importar bibliotecas
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
from datetime import datetime

def main(marks=True,save_m=False,save_t=False):
  cap = cv2.VideoCapture(0)
  detector = HandDetector(detectionCon=0.8, maxHands=1)
  key=0
  while key != 27:
    # Obtener frame
    success, img = cap.read()
    img_aux=np.copy(img)
    if marks:
      # Encontrar la mano y sus puntos de referencia
      hands, img = detector.findHands(img)  # Dibujar puntos de referencia
    else:
      hands = detector.findHands(img, draw=False)  # Sin dibujo de puntos

    if hands:
      hand1 = hands[0]
      lmList1 = hand1["lmList"]  # Lista de 21 puntos de referencia
      bbox1 = hand1["bbox"]  # Cuadro delimitador info x,y,w,h
      centerPoint1 = hand1['center']  # Centro de la mano cx,cy
      handType1 = hand1["type"]  # Tipo de mano: Left or Right

      box_x = bbox1[0]
      box_y = bbox1[1]
      box_w = bbox1[2]
      box_h = bbox1[3]
      img_x = img.shape[1]
      img_y = img.shape[0] 
      pad = 60

      #Condicion para cuando la mano no se sale de la imagen
      if box_x>0 and box_y>0 and box_x+box_w < img_x and box_y+box_h< img_y:
        try:
          hand_cap = img_aux[box_y-pad : box_y+box_h+pad, box_x-pad:box_x+box_w+pad]
          if handType1 == 'Left':
            #Se hace el reflejo en el eje x
            hand_cap = hand_cap[:,::-1]
          #Marco de la imagen para hacerla cuadrada  
          hand_cap = imgfiller(hand_cap)
          # Mostrar la zona de la imagen con la mano
          cv2.imshow("Image2", hand_cap)
          # Enmarcado
              # Rectangulo
          cv2.rectangle(img, (box_x - pad, box_y - pad),
                        (box_x + box_w + pad, box_y + box_h + pad),
                        (255, 0, 255), 2)
              # Tipo de mano
          #cv2.putText(img, handType1, (box_x - pad - 10, box_y - pad - 10), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
              
          # Para guarda la imagen con la mano
          # Se presiona la letra cuya seña corresponde a la imagen que se quiere guardar
          #key=None
          key = cv2.waitKey(1)
          if key!=-1:
            #print('hey')
            if 65<=key<=90 or 97<=key<=122:
              save_image(save_m, save_t, key,hand_cap)
            
        except: pass
      else:
        try:
            cv2.destroyWindow("Image2")
        except: pass
    # Display
    cv2.imshow("Image", img)
    key=cv2.waitKey(1)
  cap.release()
  cv2.destroyAllWindows()



def imgfiller(img):
  if img.shape[1]>=img.shape[0]:
    scale_factor = 200/img.shape[1]
    img = cv2.resize(img,(200, int(scale_factor*img.shape[0])), interpolation = cv2.INTER_AREA)
    padx1= 0
    padx2= 0
    pady1= int((200-img.shape[0])/2)
    pady2= 200-(padx1+img.shape[0])
  else:
    scale_factor = 200/img.shape[0]
    img = cv2.resize(img,(int(scale_factor*img.shape[1]), 200), interpolation = cv2.INTER_AREA)
    padx1=int((200-img.shape[1])/2)
    padx2= 200-(padx1+img.shape[1])
    pady1=0
    pady2=0
  BLUE = [255,0,0]
  img = cv2.copyMakeBorder(img,pady1,pady2,padx1,padx2,cv2.BORDER_CONSTANT,value=BLUE)
  img = cv2.resize(img,(200,200))
  return img

def save_image(save_m, save_t, key, hand_cap):
  now = datetime.now()
  tiempo = now.strftime("%H%M%S%f")
  #Crear las carpetas papel, tijera,  piedra , test
  #file_path = "dataset/papel/image_{}_"+tiempo+".jpg" # Tecla P
  file_path = "dataset/tijera/image_{}_"+tiempo+".jpg" # Tecla T
  #file_path = "dataset/piedra/image_{}_"+tiempo+".jpg" # Tecla R
  #file_path = "dataset/test/image_{}_"+tiempo+".jpg" # Tecla a
  if save_m: # Guardar para entrenamiento
    cv2.imwrite(file_path.format(chr(key).upper()),hand_cap)
    print('Guardado')

            

if __name__=='__main__':
  main(marks=True,save_m=True,save_t=False)


