#importing libraries
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

#func image

def image_process(image_path, model_path):
    #cpu yu aktif etme
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path) #yolo modelini yükleme
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (768,768))
    if image is None:
        print("image not found")
        return
    cukur = {}
    results = model.track(image, persist=True, tracker="bytetrack.yaml", conf = 0.544, device=device)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()
        for box,id in zip(boxes,ids):
            x1,y1,x2,y2 = map(int,box)
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0),2)
            cv2.putText(image, f"ID : {id}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cukur[id] = cukur.get(id, 0) + 1

    cv2.imshow("cukur detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return results




#func video

def video_proc(video_path,model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("video bulunamadı")
        return
    cukur = {}
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (768,768))
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf = 0.544, device=device)
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                for box,id in zip(boxes,ids):
                    x1,y1,x2,y2 = map(int,box)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0),2)
                    cv2.putText(frame, f"ID : {id}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cukur[id] = cukur.get(id, 0) + 1
    
        cv2.imshow("cukur detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




#main

if __name__ == "__main__":
    #image_process("/Users/mustafaseyyitdogan/Desktop/Yolo/YOLO_Project/test_images/4.jpg","/Users/mustafaseyyitdogan/Desktop/Yolo/YOLO_Project/runs/detect/train/weights/best.pt")
    video_proc("/Users/mustafaseyyitdogan/Desktop/Yolo/YOLO_Project/test_videos/Çukur_Keşif_Modeli_Video_Oluşturma.mp4","/Users/mustafaseyyitdogan/Desktop/Yolo/YOLO_Project/runs/detect/train/weights/best.pt")
