import runpod
import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(" Qwen3-VL 모델 로드 완료")


def analyze(img: Image.Image):
    """모델 추론 → mood/style/embedding 등 처리"""

    # 이미지 설명 생성
    inputs = processor(images=img, text="Describe the mood and style of this product image.",
                       return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256
    )

    description = processor.batch_decode(output, skip_special_tokens=True)[0]

    # 임베딩 생성 (text-embedding)
    emb_inputs = processor(text=description, return_tensors="pt").to(model.device)
    with torch.no_grad():
        emb = model(**emb_inputs).last_hidden_state.mean(dim=1).cpu().tolist()[0]

    # 무드 태그 추출 (간단 rule 기반)
    mood = description[:50]  # 원하는 대로 파싱

    return {
        "embedding": emb,
        "description": description,
        "mood": mood,
        "style": []  # 나중에 규칙 넣어도 됨
    }


def handler(event):
    # event["input"] 도 지원하고, 혹시 그냥 바로 넘긴 것도 지원
    data = event.get("input", event)

    b64 = data.get("image")
    if not b64:
        return {"error": "no image"}

    try:
        image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        return {"error": f"invalid image: {e}"}

    return analyze(image)

runpod.serverless.start({"handler": handler})
