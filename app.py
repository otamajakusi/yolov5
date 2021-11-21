import detect

# python detect_cv2.py --weights yolov5s.pt --source data/images/bus.jpg --device cpu
detect.run(source="data/images/bus.jpg", imgsz=[640, 640], device="cpu", conf_thres=.5)
