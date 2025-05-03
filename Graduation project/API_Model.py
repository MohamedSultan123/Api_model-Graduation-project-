from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
import os, cv2, shutil, uuid, json
import numpy as np
import gdown

model_path = "checkpoint_epoch_1.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(model_path):
    print("Downloading model checkpoint...")
    url = "https://drive.google.com/uc?id=1dIaptYPq-1fgo0yoBoPlDsbIfs3BEqJI"
    gdown.download(url, model_path, quiet=False)

model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=3)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(device)

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")

label_map = {0: "Goal", 1: "Card", 2: "Substitution"}

app = FastAPI()

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    work_dir = f"./temp/{video_id}"
    os.makedirs(work_dir, exist_ok=True)

    try:
        video_path = f"{work_dir}/input.mp4"
        with open(video_path, "wb") as f:
            f.write(await video.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (224, 224))
            frames.append(resized)
        cap.release()

        segment_size = int(fps * 5)
        predictions = []
        output_segments = []

        for i in range(0, len(frames), segment_size):
            segment = frames[i:i+segment_size]
            if len(segment) < 16:
                continue
            indices = np.linspace(0, len(segment)-1, 16).astype(int)
            sampled_frames = [segment[idx] for idx in indices]

            inputs = feature_extractor(sampled_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)

                if confidence.item() > 0.99:
                    label = label_map[pred.item()]
                    start_time = i / fps
                    end_time = min((i + segment_size), len(frames)) / fps
                    predictions.append({
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "label": label,
                        "confidence": round(confidence.item(), 3)
                    })
                    output_segments.append(segment)

        json_path = f"{work_dir}/results.json"
        with open(json_path, "w") as jf:
            json.dump(predictions, jf, indent=2)

        if output_segments:
            out_path = f"{work_dir}/output_summary.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (224, 224))
            for seg in output_segments:
                for frame in seg:
                    out.write(frame)
            out.release()
        else:
            out_path = None

        return JSONResponse({
            "num_segments": len(output_segments),
            "results_json": json_path,
            "summary_video": out_path if out_path else "No confident predictions"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})
