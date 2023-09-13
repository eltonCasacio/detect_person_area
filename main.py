import cv2
from ultralytics import YOLO
import winsound
import threading

alarmeCtl = False
PERSON_CLS = 0

yolo_versions = ['YOLOv8n.pt',	'YOLOv8m.pt']

def start():
    modelo = YOLO("YOLOv8x.pt")
    area_interesse = {
        "X-INICIO": 100,
        "Y-INICIO": 190,
        "X-FIM":1150,
        "Y-FIM":700
    }


    def alarme():
        global alarmeCtl
        for _ in range(7):
            winsound.Beep(2500,500)
        alarmeCtl = False


    def dispatch_alarm():
        global alarmeCtl
        if not alarmeCtl:
            alarmeCtl = True
            threading.Thread(target=alarme).start()


    def apply_transparency(img1:cv2.Mat, img2:cv2.Mat):
        return cv2.addWeighted(img2,0.5,img1,0.5,0)


    def extract_coordinates(dados):
        x,y,w,h = dados.xyxy[0]
        return int(x),int(y),int(w),int(h)
        

    def draw_area(mat:cv2.Mat, color:(int, int, int)):
        cv2.rectangle(mat,
                    (area_interesse['X-INICIO'], area_interesse['Y-INICIO']),
                    (area_interesse['X-FIM'], area_interesse['Y-FIM']),
                     color, -1)
        return mat


    def extract_center_object(x, y, w, h):
         return (x+w)//2, (y+h)//2
        

    def draw_rectangle_in_person(img, x, y, w, h, color, thickness):
        cv2.rectangle(img, (x, y), (w, h), color, thickness)

    
    def person_inside_area(cx, cy):
        return cx >= area_interesse['X-INICIO'] and \
               cy >= area_interesse['Y-INICIO'] and \
               cx <= area_interesse['X-FIM']    and \
               cy <= area_interesse['Y-FIM']
    

    while True:
        cap = cv2.VideoCapture(0)

        if (cap.isOpened()):
            print("Conectado com webcam")
        
        while cap.isOpened():
            ret,img = cap.read()
            if(ret):
                img  = cv2.resize(img,(1270,720))
                img2 = draw_area(img.copy(), (0,255,0))

                model_data = modelo(img)

                for objeto in model_data:
                    boxe = objeto.boxes
                    for dado in boxe:
                        classe = int(dado.cls[0]) # Pega a classe  
                        if classe == PERSON_CLS:
                            x,y,w,h = extract_coordinates(dado)
                            cx, cy  = extract_center_object(x,y,w,h)
                            draw_rectangle_in_person(img, x, y, w, h, (255, 0, 0), 3)

                            if person_inside_area(cx, cy):
                                draw_area(img2, (0,0,255))
                                cv2.rectangle(img,(100,30),(470,80),(0,0,255),-1)
                                cv2.putText(img,'INVASOR DETECTADO',(105,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
                                dispatch_alarm()

                imgFinal =  apply_transparency(img, img2)

                cv2.imshow('img',imgFinal)
                key = cv2.waitKey(5)
                if key == 27:
                    break
            
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start()