import os
import cv2
import numpy as np
import onnxruntime as ort
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context

# --- Paths ---
VIDEO_PATH = 'video/input.mp4'
FRAME_DIR = 'input_frames'
PAPRIKA_DIR = 'paprika_frames'
HAYAO_DIR = 'hayao_frames'
UPSCALED_DIR = 'upscaled_frames'

PAPRIKA_MODEL = 'AnimeGANv2_Paprika.onnx'
HAYAO_MODEL = 'AnimeGANv2_Hayao.onnx'
ESRGAN_MODEL_PATH = 'realesr-general-x4v3.pth'

# --- Ensure output dirs exist ---
for d in [FRAME_DIR, PAPRIKA_DIR, HAYAO_DIR, UPSCALED_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Extract Frames from Video ---
def extract_frames():
    print("🎞️ Extracting frames from video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(FRAME_DIR, f"frame_{count:05d}.jpg"), frame)
        count += 1
    cap.release()
    print(f"✅ Extracted {count} frames.")

# --- Preprocess / Postprocess ---
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output):
    img = output.squeeze()
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# --- Stylization Functions ---
def stylize_frame_paprika(args):
    path, index = args
    frame = cv2.imread(path)
    if frame is None:
        print(f"❌ Cannot read {path}")
        return
    inp = preprocess(frame)
    session = ort.InferenceSession(PAPRIKA_MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})[0]
    stylized = postprocess(output)
    cv2.imwrite(os.path.join(PAPRIKA_DIR, f"frame_{index:05d}.png"), stylized)

def stylize_frame_hayao(args):
    path, index = args
    frame = cv2.imread(path)
    if frame is None:
        print(f"❌ Cannot read {path}")
        return
    inp = preprocess(frame)
    session = ort.InferenceSession(HAYAO_MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})[0]
    stylized = postprocess(output)
    cv2.imwrite(os.path.join(HAYAO_DIR, f"frame_{index:05d}.png"), stylized)

# --- ESRGAN Upscaling ---
def upscale_frame(args):
    input_file, output_file = args
    try:
        import torch
        from realesrgan.utils import RealESRGANer
        from basicsr.archs.srvgg_arch import SRVGGNetCompact

        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=32, upscale=4, act_type='prelu'
        )

        upsampler = RealESRGANer(
            scale=4,
            model_path=ESRGAN_MODEL_PATH,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=torch.device("cpu")
        )

        img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Failed to read: {input_file}")
            return
        output, _ = upsampler.enhance(img)
        cv2.imwrite(output_file, output)

    except Exception as e:
        print(f"❌ Error during upscaling {input_file}: {e}")

# --- Convert Upscaled Frames to Video ---
def frames_to_video(frame_dir, output_video_path, fps=24):
    print("🎞️ Converting upscaled frames to video...")
    image_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        raise FileNotFoundError("No frames found in upscaled folder.")

    first_frame = cv2.imread(os.path.join(frame_dir, image_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for filename in image_files:
        frame = cv2.imread(os.path.join(frame_dir, filename))
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()
    print(f"✅ Video saved to {output_video_path}")




# --- Generic Stylization Runner ---
def run_stylization(stylizer_func, input_dir):
    frames = sorted(glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png")))
    args = [(frame, idx) for idx, frame in enumerate(frames)]
    with get_context("spawn").Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(stylizer_func, args), total=len(args)))

# --- Main ---
if __name__ == "__main__":
    try:
        extract_frames()

        print("🎨 Stylizing with Paprika...")
        run_stylization(stylize_frame_paprika, FRAME_DIR)

        print("🖌️ Stylizing with Hayao (on Paprika)...")
        run_stylization(stylize_frame_hayao, PAPRIKA_DIR)

        print("🔍 Upscaling with Real-ESRGAN...")
        sr_args = [
            (os.path.join(HAYAO_DIR, f), os.path.join(UPSCALED_DIR, f))
            for f in os.listdir(HAYAO_DIR)
        ]
        with Pool() as pool:
            list(tqdm(pool.imap(upscale_frame, sr_args), total=len(sr_args)))
        
# --- At the End of Main ---
        frames_to_video(UPSCALED_DIR, "final_output.mp4", fps=24)    

        print("✅ All stages complete.")

    except Exception as e:
        print(f"\n❌ Top-level error: {e}")
        input("Press Enter to exit...")
