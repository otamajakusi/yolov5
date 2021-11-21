import detect
import pprint

# python detect_cv2.py --weights yolov5s.pt --source data/images/bus.jpg --device cpu
r = detect.run(weights="runs/train/exp/weights/best.pt", source="201904300930-70861.png", imgsz=[640, 640], device="cpu", conf_thres=.5, nosave=True, max_det=20)
pprint.pprint(r)
