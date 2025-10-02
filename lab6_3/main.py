import torch
import torchvision
from ultralytics import YOLO
from transformers import Sam2Processor, Sam2Model
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request  # Для скачивания модели

# Debug версий
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Шаг 1: Загрузка изображений и моделей
image_path = "pizza.png"
new_object_prompt = "pink donut with a sprinkle"

# Обнаружение (YOLOv12) — с device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_yolo = YOLO("yolo12n.pt").to(device)
results = model_yolo(image_path, device=device, verbose=False)
if len(results[0].boxes) == 0:
    print("YOLO не нашёл объектов!")
    exit(1)
bbox = results[0].boxes.xyxy[0].cpu().numpy().tolist()
input_boxes = [[[bbox[0], bbox[1], bbox[2], bbox[3]]]]

# Сегментация (SAM 2)
processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(device)

image = Image.open(image_path)
inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(sam_model.device)

with torch.no_grad():
    outputs = sam_model(**inputs, multimask_output=False)

# Постобработка масок
masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

# Бинарная маска (с squeeze для 2D)
mask_3d = (masks[0].cpu().numpy() > 0.0).astype(np.uint8) * 255
mask = np.squeeze(mask_3d)

# Debug
print("Outputs pred_masks shape:", outputs.pred_masks.shape)
print("Masks shape after post_process:", masks.shape)
print("Mask shape after squeeze:", mask.shape)
print("Mask sum (non-zero pixels):", np.sum(mask > 0))
print("Original image size:", image.size)
print("Bbox:", bbox)

# Сохранение маски для дебага
Image.fromarray(mask).save("mask_debug.png")
print("Сохранено: mask_debug.png")

# Ресайз маски под модель SD (512x512)
orig_size = image.size
mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
# Расширение маски для менее строгого inpainting (dilation)
kernel = np.ones((5, 5), np.uint8)  # Размер ядра: 15x15 для умеренного расширения (можно уменьшить до 10x10 или увеличить до 20x20)
mask_expanded = cv2.dilate(mask_resized, kernel, iterations=1)  # 1 итерация — лёгкое расширение
mask_resized = mask_expanded  # Заменяем оригинальную на расширенную
print("Маска расширена для inpainting.")

# Сохранение оригинала 512x512
init_image = Image.open(image_path).resize((512, 512)).convert('RGB')

# Шаг 2: Удаление с OpenCV LaMa (скачиваем модель, если нет)
print("Загрузка LaMa в OpenCV...")
path_to_model = "lama.onnx"
if not os.path.exists(path_to_model):
    print("Скачиваем LaMa модель...")
    url = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama.onnx"
    urllib.request.urlretrieve(url, path_to_model)
    print("Модель скачана!")

try:
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path_to_model)
    sr.setModel("lama", 1)  # Scale 1 for inpainting
    init_np = np.array(init_image)
    mask_np = mask_resized
    inpainted_np = sr.inpaint(init_np, mask_np)
    inpainted_image = Image.fromarray(inpainted_np)
    print("LaMa удаление успешно!")
except Exception as e:
    print(f"Ошибка LaMa: {e}. Fallback на cv2.inpaint.")
    init_np = np.array(init_image)
    mask_np = mask_resized // 255  # Бинарная маска для cv2.inpaint
    inpainted_np = cv2.inpaint(init_np, mask_np, 3, cv2.INPAINT_TELEA)  # TELEA алгоритм
    inpainted_image = Image.fromarray(inpainted_np)

# Сохранение после удаления
inpainted_image.save("inpainted_removed_dog.jpg")
print("Сохранено: inpainted_removed_dog.jpg")

# Шаг 3: Генерация нового объекта (SD Inpainting)
dtype = torch.float16 if device == "cuda" else torch.float32
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=dtype
).to(device)
pipe.enable_model_cpu_offload()

new_object = pipe(
    prompt=new_object_prompt + "seamless integration",
    image=inpainted_image,  # Base без собаки
    mask_image=Image.fromarray(mask_resized),
    strength=0.9,
    num_inference_steps=50,
    negative_prompt="deformed, squeezed, ugly, low quality"
).images[0]

# Композитинг (масштабированный центр)
scale_x = 512 / orig_size[0]
scale_y = 512 / orig_size[1]
center_x = int((bbox[0] + (bbox[2] - bbox[0]) / 2) * scale_x)
center_y = int((bbox[1] + (bbox[3] - bbox[1]) / 2) * scale_y)
inpainted_np = np.array(inpainted_image)
new_np = np.array(new_object)
blended = cv2.seamlessClone(
    new_np, inpainted_np, mask_resized, (center_x, center_y), cv2.NORMAL_CLONE
)

# Сохранение (ресайз обратно к оригиналу)
Image.fromarray(blended).resize(orig_size).save("test_replaced3.jpg")
print("Замена завершена!")