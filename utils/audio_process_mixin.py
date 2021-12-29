import numpy as np
import threading, queue

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model_name = "container_0/wav2vec2-large-xlsr-ja"
device = "cuda"
processor_name = "container_0/wav2vec2-large-xlsr-ja"

model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(processor_name)


class VAD:
    def __init__(self, confidence=0.85):
        self.confidence = confidence
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        self.model = model

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype("float32")
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def validate(self, inputs: torch.Tensor):
        with torch.no_grad():
            outs = self.model(inputs, sr=16_000)
        return outs


class AudioProcessMixin:
    WIDTH_DICT = {1: np.int8, 2: np.int16, 3: np.int32}
    vad = VAD(0.8)

    speech = []
    queue = queue.Queue()

    def process_audio(self, data, width, channels, framerate):
        # print(f"{width=},{channels=},{framerate=}")
        assert channels <= 2, f"{channel=}"
        ndata = np.frombuffer(data, np.int16)
        fdata = self.vad.int2float(ndata)
        tdata = torch.from_numpy(fdata)

        vad_outs = self.vad.validate(tdata)
        confidence = vad_outs.numpy()[0].item()
        if confidence > self.vad.confidence:
            # print(f"speeking {confidence:.2f}")
            self.speech.append(data)
        else:
            if len(self.speech):
                ndata = np.frombuffer(b"".join(self.speech), self.WIDTH_DICT[width])
                fdata = self.vad.int2float(ndata)
                out = self.predict(fdata, 16_000)
                if len(out[0]):
                    # print(out)
                    self.queue.put(out[0])

                self.speech = []

    def predict(self, data, sampling_rate):
        features = processor(
            data,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        input_values = features.input_values.to(device)
        attention_mask = features.attention_mask.to(device)
        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        decoded_results = []
        for logit in logits:
            pred_ids = torch.argmax(logit, dim=-1)
            decoded_results.append(processor.decode(pred_ids))
        return decoded_results
