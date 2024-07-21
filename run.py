import json
import yaml
from pathlib import Path
import torch
from sentencepiece_tokenizer import SentencepiecesTokenizer
from wav_frontend import WavFrontend, WavFrontendOnline
from model import SenseVoiceSmall
from vad_model import FsmnVADStreaming
from load_pretrained_model import load_pretrained_model
import logging
logging.basicConfig(level=logging.INFO)


model_path = '/Users/liwei/huggingface/SenseVoiceSmall/'
yaml_path = Path(model_path) / 'config.yaml'
json_path = Path(model_path) / 'configuration.json'

vad_model_path = '/Users/liwei/huggingface/speech_fsmn_vad_zh-cn-16k-common-pytorch/'
vyaml_path = Path(vad_model_path) / 'config.yaml'
vjson_path = Path(vad_model_path) / 'configuration.json'

if __name__ == '__main__':
    with open(yaml_path, 'r') as R:
        y_conf = yaml.safe_load(R)
    with open(json_path, 'r') as R:
        j_conf = json.load(R)

    with open(vyaml_path, 'r') as R:
        vy_conf = yaml.safe_load(R)
    with open(vjson_path, 'r') as R:
        vj_conf = json.load(R)

    y_conf['tokenizer_conf']['bpemodel'] = Path(model_path) / 'chn_jpn_yue_eng_ko_spectok.bpe.model'

    tokenizer = SentencepiecesTokenizer(**y_conf['tokenizer_conf'])
    vocab_size = tokenizer.get_vocab_size()

    y_conf['frontend_conf']['cmvn_file'] = Path(model_path) / 'am.mvn'
    frontend = WavFrontend(**y_conf['frontend_conf'])

    y_conf['input_size'] = frontend.output_size()
    model = SenseVoiceSmall(**y_conf, vocab_size=vocab_size)

    load_pretrained_model(
                    model=model,
                    path=Path(model_path) / j_conf['file_path_metas']['init_param'],
                    ignore_init_mismatch=True,
                    oss_bucket=None,
                    scope_map=[],
                    excludes=None)


    device = 'cpu'
    data_type = torch.float16
    vy_conf['device'] = device
    vy_conf['frontend_conf']['cmvn_file'] = Path(vad_model_path) / 'am.mvn'
    vy_conf['frontend'] = WavFrontendOnline(**vy_conf['frontend_conf'])
    # vy_conf['model_conf']['max_single_segment_time'] = 30000
    vad_model = FsmnVADStreaming(**vy_conf)
    load_pretrained_model(
                    model=vad_model,
                    path=Path(vad_model_path) / vj_conf['file_path_metas']['init_param'],
                    ignore_init_mismatch=True,
                    oss_bucket=None,
                    scope_map=[],
                    excludes=None)
    # vy_conf['frontend'] = frontend
    res = vad_model.inference(
            data_in=['/Users/liwei/repos/SenseVoice-MLX/suibian.mp3'],
            key=['suibian'],
            **vy_conf)
    print(res)

    # model.to(data_type).to(device)
