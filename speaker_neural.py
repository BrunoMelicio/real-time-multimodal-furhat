"""Tested Light-ASD CPU adapter; larger alternatives are excluded."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
LIGHT_SHA="9d2a6db1bf27929dfff5a460438598130bc7bd8e"

class ActiveSpeakerModel:
    def __init__(self, kind, threads=2):
        import torch
        self.torch, self.kind = torch, kind
        torch.set_num_threads(threads)
        if kind == 'light':
            from speaker_models.light.Model import ASD_Model
            self.model = ASD_Model()
            dimension = 128
            path = ROOT/'models'/'speaker_association'/'light_talkset.model'
            url = f'https://raw.githubusercontent.com/nvthach124/Light-ASD/{LIGHT_SHA}/weight/finetuning_TalkSet.model'
            print('[MODEL] Light-ASD | using cached TalkSet weights.' if path.exists() else
                  '[MODEL] Light-ASD | downloading official TalkSet weights.', flush=True)
            state = torch.hub.load_state_dict_from_url(url, model_dir=str(path.parent),
                                                       file_name=path.name, map_location='cpu', weights_only=True)
        else:
            raise ValueError('This curated build supports only Light-ASD.')
        if not isinstance(state, dict):
            raise RuntimeError('Expected an upstream state_dict checkpoint.')
        state = {k.removeprefix('module.'): v for k,v in state.items()}
        self.fc = torch.nn.Linear(dimension, 2)
        # Only training-only classifiers are omitted. All inference tensors must match.
        self.model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')}, strict=True)
        self.fc.load_state_dict({k.removeprefix('lossAV.FC.'):v for k,v in state.items() if k.startswith('lossAV.FC.')}, strict=True)
        self.model.eval()
        self.fc.eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())+sum(p.numel() for p in self.fc.parameters())
        print(f'[MODEL] {kind} ready | {self.parameters:,} parameters | CPU {threads} threads | weights {path.stat().st_size/1e6:.1f}MB', flush=True)

    def score(self, audios, videos):
        """Reuse audio/visual encodings; softmax scores are uncalibrated match evidence."""
        return self.score_frames(audios, videos).mean(axis=2)

    def score_frames(self, audios, videos):
        """Audio rows x visible-person columns x video frames, for time attribution.

        The existing score() interface still returns the same temporal average.
        These are model scores, not calibrated identity probabilities.
        """
        import python_speech_features
        torch = self.torch
        frames = min(len(video) for video in videos)
        if frames < 15:
            raise ValueError('Need at least 15 video frames')
        audio_embeddings, visual_embeddings = [], []
        with torch.inference_mode():
            for audio in audios:
                # Upstream demo computes MFCC from signed PCM16 at 16kHz.
                pcm = np.rint(np.clip(audio,-1,1)*32767).astype('int16')
                features = python_speech_features.mfcc(pcm, 16000, numcep=13, winlen=.025, winstep=.010)
                count = 4*frames
                if len(features)<count:
                    features = np.pad(features, ((0,count-len(features)),(0,0)), mode='edge')
                a = torch.from_numpy(np.asarray(features[:count],dtype='float32')).unsqueeze(0)
                audio_embeddings.append(self.model.forward_audio_frontend(a))
            for video in videos:
                v = torch.from_numpy(np.ascontiguousarray(video[:frames],dtype='float32')).unsqueeze(0)
                visual_embeddings.append(self.model.forward_visual_frontend(v))
            scores = np.zeros((len(audios),len(videos),frames))
            for row,a in enumerate(audio_embeddings):
                for column,v in enumerate(visual_embeddings):
                    if self.kind == 'talk':
                        a_pair,v_pair = self.model.forward_cross_attention(a,v)
                    else:
                        a_pair,v_pair = a,v
                    embeddings = self.model.forward_audio_visual_backend(a_pair,v_pair)
                    probabilities = torch.softmax(self.fc(embeddings),dim=-1)[:,1]
                    values = probabilities.detach().cpu().numpy()
                    if len(values) != frames:
                        raise RuntimeError(f'ASD returned {len(values)} time steps for {frames} video frames.')
                    scores[row,column] = values
        return scores
