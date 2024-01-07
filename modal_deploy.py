import os
import modal
import fastapi
import fastapi.staticfiles
from modal_const import CACHE_PATH
from pydantic import BaseModel
from fastapi.responses import Response

stub = modal.Stub("bert-vits2")
web_app = fastapi.FastAPI()


def download_model_weights():
    import requests
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    model_files = ["config.json", "D_88000.pth", "DUR_88000.pth", "G_88000.pth", "WD_88000.pth"]
    os.makedirs(CACHE_PATH, exist_ok=True)
    for model_file in model_files:
        rsp = requests.get("HOSTURL" + model_file)
        rsp.raise_for_status()
        with open(os.path.join(CACHE_PATH, model_file), "wb") as f:
            for chunk in rsp.iter_content(chunk_size=1024*1024): 
                if chunk:
                    f.write(chunk)

    os.makedirs(CACHE_PATH + "/bert/deberta-v2-large-japanese-char-wwm", exist_ok=True)
    snapshot_download(
        "ku-nlp/deberta-v2-large-japanese-char-wwm",
        local_dir=CACHE_PATH + "/bert/deberta-v2-large-japanese-char-wwm",
    )
    move_cache()

    os.makedirs(CACHE_PATH + "/bert/chinese-roberta-wwm-ext-large", exist_ok=True)
    snapshot_download(
        "hfl/chinese-roberta-wwm-ext-large",
        local_dir=CACHE_PATH + "/bert/chinese-roberta-wwm-ext-large",
    )
    move_cache()

    os.makedirs(CACHE_PATH + "/bert/deberta-v3-large", exist_ok=True)
    snapshot_download(
        "microsoft/deberta-v3-large",
        local_dir=CACHE_PATH + "/bert/deberta-v3-large",
    )
    move_cache()

    os.makedirs(CACHE_PATH + "/bert/deberta-v2-large-japanese", exist_ok=True)
    snapshot_download(
        "ku-nlp/deberta-v2-large-japanese",
        local_dir=CACHE_PATH + "/bert/deberta-v2-large-japanese",
    )
    move_cache()

    os.makedirs(CACHE_PATH + "/bert/bert-base-japanese-v3", exist_ok=True)
    snapshot_download(
        "cl-tohoku/bert-base-japanese-v3",
        local_dir=CACHE_PATH + "/bert/bert-base-japanese-v3",
    )
    move_cache()

    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('cmudict')


image = (
    modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "librosa==0.9.2",
            "matplotlib",
            "numpy",
            "numba",
            "phonemizer",
            "scipy",
            "tensorboard",
            "Unidecode",
            "amfm_decompy",
            "jieba",
            "transformers",
            "pypinyin",
            "cn2an",
            "gradio==3.50.2",
            "av",
            "mecab-python3",
            "loguru",
            "unidic-lite",
            "cmudict",
            "fugashi",
            "num2words",
            "PyYAML",
            "requests",
            "pyopenjtalk-prebuilt",
            "jaconv",
            "psutil",
            "GPUtil",
            "vector_quantize_pytorch",
            "g2p_en",
            "sentencepiece",
            "pykakasi",
            "langid",
            "torch",
            "torchvision",
            "torchaudio",
        )
        .run_function(download_model_weights)
)

@stub.function(
    gpu="l4",
    image=image,
    retries=3,
    mounts=[
        modal.Mount.from_local_python_packages("config"),
        modal.Mount.from_local_python_packages("tools"),
        modal.Mount.from_local_python_packages("utils"),
        modal.Mount.from_local_python_packages("infer"),
        modal.Mount.from_local_python_packages("re_matching"),
        modal.Mount.from_local_python_packages("modal_const"),
        modal.Mount.from_local_python_packages("commons"),
        modal.Mount.from_local_python_packages("text"),
        modal.Mount.from_local_python_packages("models"),
        modal.Mount.from_local_python_packages("modules"),
        modal.Mount.from_local_python_packages("transforms"),
        modal.Mount.from_local_python_packages("attentions"),
        modal.Mount.from_local_python_packages("monotonic_align"),
        modal.Mount.from_local_python_packages("oldVersion"),
        modal.Mount.from_local_file("config.yml", CACHE_PATH + "/config.yml"),
        modal.Mount.from_local_file("bert/bert_models.json", CACHE_PATH + "/bert/bert_models.json"),
    ]
)
def speech(
    text,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    style_text,
    style_weight,
):
    import torch
    import utils
    from infer import infer, latest_version, get_net_g
    import gradio as gr
    import numpy as np
    from config import config
    import wave
    import tempfile

    net_g = None
    device = config.webui_config.device
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    # speakers = list(speaker_ids.keys())
    # languages = ["ZH", "JP", "EN", "mix", "auto"]

    def generate_audio(
        text,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        speaker,
        language,
        reference_audio,
        emotion,
        style_text,
        style_weight,
        skip_start=False,
        skip_end=False,
    ):
        slices = text.split("|")
        audio_list = []
        # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
        with torch.no_grad():
            for idx, piece in enumerate(slices):
                skip_start = idx != 0
                skip_end = idx != len(slices) - 1
                audio = infer(
                    piece,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    skip_start=skip_start,
                    skip_end=skip_end,
                    style_text=style_text,
                    style_weight=style_weight,
                )
                audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
                audio_list.append(audio16bit)
        return np.concatenate(audio_list)

    res = generate_audio(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, speaker, language, None, "Happy", style_text, style_weight)
    data = res.tobytes()
    tempfd, temppath = tempfile.mkstemp()
    with wave.open(temppath, "wb") as wav_file:
        wav_file.setparams((1, 2, 44100, 0, 'NONE', 'NONE'))
        wav_file.writeframes(data)

    ret = b""
    with open(temppath, "rb") as wav_file:
        ret = wav_file.read()

    os.close(tempfd)

    return ret

class SpeechReq(BaseModel):
    text: str
    speaker: str
    sdp_ratio: float = 0.5
    noise_scale: float = 0.6
    noise_scale_w: float = 0.9
    length_scale: float = 1.0
    language: str = "ZH"
    style_text: str = ""
    style_weight: float = 0.7

@web_app.post("/submit")
async def submit(req: SpeechReq):
    speech = modal.Function.lookup("bert-vits2", "speech")
    call = speech.spawn(
        req.text,
        req.speaker,
        req.sdp_ratio,
        req.noise_scale,
        req.noise_scale_w,
        req.length_scale,
        req.language,
        req.style_text,
        req.style_weight,
    )
    return {"call_id": call.object_id}

@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return fastapi.responses.JSONResponse(content="Still running", status_code=202)

    return Response(content=result, media_type="audio/x-wav")

@stub.function()
@modal.asgi_app()
def wrapper():
    return web_app