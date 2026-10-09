"""Small interchangeable streaming VADs; model imports occur only on selection."""


class SileroVAD:
    """Stateful 16 kHz/512-sample streaming adapter for faster-whisper's v6 ONNX.

    Mirrors the bundled wrapper's 64-sample context and h/c state, but retains
    state across individual chunks instead of processing whole recordings.
    """
    frame_samples = 512

    def __init__(self, threshold=.70, *, session=None):
        import numpy as np
        if not 0 < threshold < 1:
            raise ValueError('Silero threshold must be between 0 and 1')
        self.np = np
        self.threshold = threshold
        if session is None:
            from faster_whisper.vad import get_vad_model
            session = get_vad_model().session
        self.session = session
        if {value.name for value in session.get_inputs()} != {'input','h','c'}:
            raise RuntimeError('Expected bundled Silero v6 ONNX inputs input/h/c. '
                               'Update faster-whisper using requirements_local_audio.txt.')
        self.reset_states()

    def reset_states(self):
        self.h = self.np.zeros((1,1,128),dtype='float32')
        self.c = self.np.zeros((1,1,128),dtype='float32')
        self.context = self.np.zeros((1,64),dtype='float32')
        self.probability = 0.

    def is_speech(self, pcm, sample_rate):
        if sample_rate != 16000 or len(pcm) != 1024:
            raise ValueError('Silero needs exactly 512 PCM16 samples at 16 kHz')
        audio = self.np.frombuffer(pcm,dtype='<i2').astype('float32').reshape(1,-1)/32768.
        model_input = self.np.concatenate((self.context,audio),axis=1)
        output,self.h,self.c = self.session.run(None,{'input':model_input,'h':self.h,'c':self.c})
        self.context = audio[:,-64:].copy()
        self.probability = float(output.reshape(-1)[0])
        return self.probability >= self.threshold


def make_vad(name, threshold=.70):
    if name == 'silero':
        return SileroVAD(threshold),512
    if name == 'webrtc':
        import webrtcvad
        return webrtcvad.Vad(2),320
    raise ValueError(f'Unknown VAD: {name}')
