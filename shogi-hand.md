1. place wav2vec2 model under container_0
1. python3 detect.py --source myvideo.mp4 --weights runs/train/exp2/weights/best.pt --view-img --nosave --line-thickness 2 --audio myaudio.sr=16k.wav 2> /dev/null
