import os
cnhubert_base_path = "GPT_SoVITS\pretrained_models\chinese-hubert-base"
bert_path = "GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large"
os.environ["version"] = 'v2'
import re
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch,numpy as np
from pathlib import Path
import os,librosa,torch
from feature_extractor import cnhubert
cnhubert.cnhubert_base_path=cnhubert_base_path
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import os
import json
import pyopenjtalk
cwd = os.getcwd()
if os.path.exists(os.path.join(cwd,'user.dic')):
    pyopenjtalk.update_global_jtalk_with_user_dict(os.path.join(cwd, 'user.dic'))


import logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
is_half = False

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model=AutoModelForMaskedLM.from_pretrained(bert_path)
if(is_half==True):bert_model=bert_model.half().to(device)
else:bert_model=bert_model.to(device)
# bert_model=bert_model.to(device)
def get_bert_feature(text, word2ph): # Bert(不是HuBERT的特征计算)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)#####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T

loaded_sovits_model = [] # [(path, dict, model)]
loaded_gpt_model = []
ssl_model = cnhubert.get_model()
if (is_half == True):
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def load_model(sovits_path, gpt_path):
    global ssl_model
    global loaded_sovits_model
    global loaded_gpt_model
    vq_model = None
    t2s_model = None
    dict_s2 = None
    dict_s1 = None
    hps = None
    for path, dict_s2_, model in loaded_sovits_model:
        if path == sovits_path:
            vq_model = model
            dict_s2 = dict_s2_
            break
    for path, dict_s1_, model in loaded_gpt_model:
        if path == gpt_path:
            t2s_model = model
            dict_s1 = dict_s1_
            break
    
    if dict_s2 is None:
        dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]

    if dict_s1 is None:
        dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    class DictToAttrRecursive:
        def __init__(self, input_dict):
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    # 如果值是字典，递归调用构造函数
                    setattr(self, key, DictToAttrRecursive(value))
                else:
                    setattr(self, key, value)

    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"


    if not vq_model:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        if (is_half == True):
            vq_model = vq_model.half().to(device)
        else:
            vq_model = vq_model.to(device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        loaded_sovits_model.append((sovits_path, dict_s2, vq_model))
    hz = 50
    max_sec = config['data']['max_sec']
    if not t2s_model:
        t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if (is_half == True): t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(device)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        loaded_gpt_model.append((gpt_path, dict_s1, t2s_model))
    return vq_model, ssl_model, t2s_model, hps, config, hz, max_sec


def get_spepc(hps, filename):
    audio=load_audio(filename,int(hps.data.sampling_rate)) 
    audio = audio / np.max(np.abs(audio))
    audio=torch.FloatTensor(audio)
    print(torch.max(torch.abs(audio)))
    audio_norm = audio
    # audio_norm = audio / torch.max(torch.abs(audio))
    audio_norm = audio_norm.unsqueeze(0)
    print(torch.max(torch.abs(audio_norm)))
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,center=False)
    return spec

def create_tts_fn(vq_model, ssl_model, t2s_model, hps, config, hz, max_sec):
    def tts_fn(ref_wav_path, prompt_text, prompt_language, target_phone, text_language):
        t0 = ttime()
        prompt_text=prompt_text.strip()
        prompt_language=prompt_language
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)  # 派蒙
            # maxx=0.95
            # tmp_max = np.abs(wav16k).max()
            # alpha=0.5
            # wav16k = (wav16k / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * wav16k
            #在这里归一化
            #print(max(np.abs(wav16k)))
            #wav16k = wav16k / np.max(np.abs(wav16k))
            #print(max(np.abs(wav16k)))
            # 添加0.3s的静音
            wav16k = np.concatenate([wav16k, np.zeros(int(hps.data.sampling_rate * 0.3)),])
            wav16k = torch.from_numpy(wav16k)
            wav16k = wav16k.float()
            if(is_half==True):wav16k=wav16k.half().to(device)
            else:wav16k=wav16k.to(device)
            print(wav16k.shape) # 读取16k音频
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)#.float()
            print(ssl_content.shape)
            codes = vq_model.extract_latent(ssl_content)
            print(codes.shape)
            prompt_semantic = codes[0, 0]
        t1 = ttime()
        phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
        phones1=cleaned_text_to_sequence(phones1)
        #texts=text.split("\n")
        audio_opt = []
        zero_wav=np.zeros(int(hps.data.sampling_rate*0.3),dtype=np.float16 if is_half==True else np.float32)
        phones = get_phone_from_str_list(target_phone, text_language)
        for phones2 in phones:
            if(len(phones2) == 0):
                continue
            if(len(phones2) == 1 and phones2[0] == ""):
                continue
            #phones2, word2ph2, norm_text2 = clean_text(text, text_language)
            print(phones2)
            phones2 = cleaned_text_to_sequence(phones2)
            #if(prompt_language=="zh"):bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
            bert1 = torch.zeros((1024, len(phones1)),dtype=torch.float16 if is_half==True else torch.float32).to(device)
            #if(text_language=="zh"):bert2 = get_bert_feature(norm_text2, word2ph2).to(device)
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
            prompt = prompt_semantic.unsqueeze(0).to(device)
            t2 = ttime()
            idx = 0
            cnt = 0
            while idx == 0 and cnt < 2:
                with torch.no_grad():
                    # pred_semantic = t2s_model.model.infer
                    pred_semantic,idx = t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        prompt,
                        bert,
                        # prompt_phone_len=ph_offset,
                        top_k=config['inference']['top_k'],
                        early_stop_num=hz * max_sec)
                t3 = ttime()
                cnt+=1
            if idx == 0:
                return "Error: Generation failure: bad zero prediction.", None
            pred_semantic = pred_semantic[:,-idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = get_spepc(hps, ref_wav_path)#.to(device)
            if(is_half==True):refer=refer.half().to(device)
            else:refer=refer.to(device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer).detach().cpu().numpy()[0, 0]###试试重建不带上prompt部分
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return "Success", (hps.data.sampling_rate,(np.concatenate(audio_opt,0)*32768).astype(np.int16))
    return tts_fn


def get_str_list_from_phone(text, text_language):
    # raw文本过g2p得到音素列表，再转成字符串
    # 注意，这里的text是一个段落，可能包含多个句子
    # 段落间\n分割，音素间空格分割
    texts=text.split("\n")
    phone_list = []
    for text in texts:
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phone_list.append(" ".join(phones2))
    return "\n".join(phone_list)

def get_phone_from_str_list(str_list:str, language:str = 'ja'):
    # 从音素字符串中得到音素列表
    # 注意，这里的text是一个段落，可能包含多个句子
    # 段落间\n分割，音素间空格分割
    sentences = str_list.split("\n")
    phones = []
    for sentence in sentences:
        phones.append(sentence.split(" "))
    return phones

splits={"，","。","？","！",",",".","?","!","~",":","：","—","…",}#不考虑省略号
def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if (todo_text[-1] not in splits): todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while (1):
        if (i_split_head >= len_text): break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if (todo_text[i_split_head] in splits):
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def change_reference_audio(prompt_text, transcripts):
    return transcripts[prompt_text]


models = []
models_info = json.load(open("./models/models_info.json", "r", encoding="utf-8")) 



for i, info in models_info.items():
    title = info['title']
    cover = info['cover']
    gpt_weight = info['gpt_weight']
    sovits_weight = info['sovits_weight']
    example_reference = info['example_reference']
    transcripts = {}
    transcript_path = info["transcript_path"]
    path = os.path.dirname(transcript_path)
    with open(transcript_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().replace("\\", "/")
            wav,_,_, t = line.split("|")
            wav = os.path.basename(wav)
            transcripts[t] = os.path.join(os.path.join(path,"reference_audio"), wav)

    vq_model, ssl_model, t2s_model, hps, config, hz, max_sec = load_model(sovits_weight, gpt_weight)


    models.append(
        (
            i,
            title,
            cover,
            transcripts,
            example_reference,
            create_tts_fn(
                vq_model, ssl_model, t2s_model, hps, config, hz, max_sec
            )
        )
    )
with gr.Blocks() as app:
    gr.Markdown(
        "# <center> GPT-SoVITS Demo\n"
    )
    with gr.Tabs():
        for (name, title, cover, transcripts, example_reference, tts_fn) in models:
            with gr.TabItem(name):
                with gr.Row():
                    gr.Markdown(
                        '<div align="center">'
                        f'<a><strong>{title}</strong></a>'
                        '</div>')
                with gr.Row():
                    with gr.Column():
                        prompt_text = gr.Dropdown(
                            label="Transcript of the Reference Audio",
                            value=example_reference if example_reference in transcripts else list(transcripts.keys())[0],
                            choices=list(transcripts.keys())
                        )
                        inp_ref_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                            interactive=False,
                            value=transcripts[example_reference] if example_reference in transcripts else list(transcripts.values())[0]
                        )
                        transcripts_state = gr.State(value=transcripts)
                        prompt_text.change(
                            fn=change_reference_audio,
                            inputs=[prompt_text, transcripts_state],
                            outputs=[inp_ref_audio]
                        )
                        prompt_language = gr.State(value="ja")
                    with gr.Column():
                        text = gr.Textbox(label="Input Text", value="こんにちは、私はあなたのAIアシスタントです。仲良くしましょうね。")
                        text_language = gr.Dropdown(
                            label="Language",
                            choices=["ja"],
                            value="ja"
                        )
                        clean_button = gr.Button("Clean Text", variant="primary")
                        inference_button = gr.Button("Generate", variant="primary")
                        cleaned_text = gr.Textbox(label="Cleaned Text")
                        output = gr.Audio(label="Output Audio")
                        om = gr.Textbox(label="Output Message")
                        clean_button.click(
                            fn=get_str_list_from_phone,
                            inputs=[text, text_language],
                            outputs=[cleaned_text]
                        )
                        inference_button.click(
                            fn=tts_fn,
                            inputs=[inp_ref_audio, prompt_text, prompt_language, cleaned_text, text_language],
                            outputs=[om, output]
                        )

app.launch(share=True)