import torch
import torch.nn as nn
import hydra
import json
import task
from .mert.feature_extractor import main as feature_extractor
from .mert.feature_extractor import MERTFeature
from .mert.audio_utils import *

def id2class(config, id):
    with open(config, 'r') as file:
        data = json.load(file)
    return data[id]

def get_model(cfg) -> task.ProberForBertUtterCLS:
    prober = eval(f'task.{cfg.task}Prober')
    return prober(cfg)

class PromptMaker(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_extractor = MERTFeature(cfg)
        self.emo = get_model(cfg.emo)
        self.genre_classification = get_model(cfg.genre_classification)
        self.instrument_classification = get_model(cfg.instrument_classification)
        self.key_detect = get_model(cfg.ket_detect)
        self.pitch_classification = get_model(cfg.pitch_classfication)
        self.vocal_technique_detection = get_model(cfg.vocal_technique_detection)
        self.all = {'emo': self.emo, 
                    'genre_classification': self.genre_classification,
                    'instrument_classification': self.instrument_classification,
                    'key_detect': self.key_detect, 
                    'pitch_classification': self.pitch_classification,
                    'vocal_technique_detection': self.vocal_technique_detection}
        self.id2classes = {'emo': cfg.id2class.emo,
                        'genre_classification': cfg.id2class.genre_classification,
                        'instrument_classification': cfg.id2class.instrument_classification,
                        'key_detect': cfg.id2class.key_detect,
                        'pitch_classification': cfg.id2class.pitch_classification,
                        'vocal_technique_detection': cfg.id2class.vocal_technique_detection}
        self.lm = ...
        
    def make_prompt(self, input_audio):
        waveform = load_audio(input_audio, target_sr=self.cfg.target_sr,
                              is_mono=self.cfg.is_mono, is_normalize=self.cfg.is_normalize,
                              crop_to_length_in_sec=self.cfg.crop_to_length_in_sec,
                              crop_randomly=self.cfg.crop_randomly, device=self.cfg.device)
        wav = self.feature_extractor.process_wav(waveform=waveform)
        wav = wav.to(self.cfg.device)
        features = feature_extractor(wav, layer='all', reduction=self.cfg.reduction)
        input_features = []
        for i in range(features.shape[1]):
            feature = features[:, i, :]
            if feature.ndim == 2:
                if self.cfg.layer == 'all':
                    feature = torch.from_numpy(feature[1:]).unsqueeze(dim=0)
                else:
                    feature = torch.from_numpy(feature[int(self.cfg.layer)]).unsqueeze(dim=0)
            else:
                raise ValueError
            input_features.append(feature)
        input_features = torch.stack(input_features, dim=0)
        input_features = input_features.unsqueeze(0)
        predictions = {}
        
        with torch.no_grad():
            for key, mod in self.all:
                pred = mod(input_features)
                pred = pred.mean(dim=0)
                class_pred = torch.argmax(pred, dim=1)
                predictions[key] = id2class(self.id2classes[key], class_pred)
    
@hydra.main(config_name="config", config_path="./configs")
def main(cfg, input_path):
    prompt_maker = PromptMaker(cfg)
    prompt_maker.make_prompt(input_path)