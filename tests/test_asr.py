"""Selection and adapter contracts for local ASR; no downloads or devices."""
import argparse
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from furhat_interaction.asr import ASRConfig,add_asr_arguments,resolve_asr,create_asr


class SpeechSelection(unittest.TestCase):
    def select(self,*flags):
        parser=argparse.ArgumentParser();add_asr_arguments(parser)
        return resolve_asr(parser.parse_args(flags))

    def test_whisper_default_tiny_preset_and_legacy_option(self):
        self.assertEqual(self.select(),ASRConfig('whisper','base.en'))
        self.assertEqual(self.select('--asr=whisper-tiny'),ASRConfig('whisper','tiny.en'))
        self.assertEqual(self.select('--asr=whisper','--asr-model=tiny'),ASRConfig('whisper','tiny.en'))
        self.assertEqual(self.select('--whisper=small.en'),ASRConfig('whisper','small.en'))

    def test_whistle_and_phonon_are_independent_choices(self):
        self.assertEqual(self.select('--asr=whistle'),ASRConfig('whistle','whistle'))
        self.assertEqual(self.select('--asr=phonon'),ASRConfig('phonon','phonon-2'))
        self.assertEqual(self.select('--asr=phonon-2'),ASRConfig('phonon','phonon-2'))
        with self.assertRaises(ValueError): self.select('--asr=phonon','--whisper=base.en')

    def test_whistle_receives_audio_directly_without_files(self):
        model=Mock(weights='cached.cact')
        model.transcribe.return_value={'text':' Hello Bruno. '}
        with patch.dict(sys.modules,{'needle':SimpleNamespace(Whistle=Mock(return_value=model))}):
            backend=create_asr(ASRConfig('whistle','whistle'))
            audio=[0.,.1]
            self.assertEqual(backend.decode(audio),'Hello Bruno.')
            model.transcribe.assert_called_once_with(audio,language='en')
            backend.close()

    def test_phonon_uses_persistent_array_api(self):
        model=Mock();model.transcribe_array.return_value=(' Hello Bruno. ',.01,.5)
        load=Mock(return_value=model)
        with patch.dict(sys.modules,{'fermion':SimpleNamespace(load_speech=load)}):
            backend=create_asr(ASRConfig('phonon','phonon-2'))
            self.assertEqual(backend.decode([0.,.1]),'Hello Bruno.')
            load.assert_called_once_with('phonon-2')
            backend.decode([0.,.2])
            self.assertEqual(load.call_count,1)
            backend.close()


if __name__=='__main__': unittest.main()
