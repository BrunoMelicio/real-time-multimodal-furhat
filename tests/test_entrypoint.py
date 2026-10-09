"""Entrypoint routing checks; no hardware/model owners are started."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
from furhat_interaction.app import main


class EntryPoint(unittest.TestCase):
    def test_multi_forwards_only_scene_arguments(self):
        with patch('furhat_interaction.multi_person.runtime.run',return_value=0) as run:
            self.assertEqual(main(['--people=2','--part=3','--mic=iphone','--names','Bruno','Maria','--winner','Maria']),0)
        run.assert_called_once_with(3,['--mic=iphone','--names','Bruno','Maria','--winner','Maria'])

    def test_single_restores_arguments_after_error(self):
        before=sys.argv
        def fail():
            self.assertEqual(sys.argv[1:],['--mic=builtin'])
            raise RuntimeError('fake scene error')
        with patch('furhat_interaction.app.importlib.import_module',return_value=SimpleNamespace(main=fail)) as load:
            with self.assertRaises(RuntimeError): main(['--people=1','--part=2','--mic=builtin'])
        self.assertIs(sys.argv,before)
        load.assert_called_once_with('furhat_interaction.single_person.game')

    def test_named_multi_scenarios(self):
        for part, scenario in enumerate(('introduction', 'game', 'reflection'), 1):
            with self.subTest(scenario=scenario), patch('furhat_interaction.multi_person.runtime.run',return_value=0) as run:
                self.assertEqual(main(['--people=2', '--scenario='+scenario, '--mic=builtin']),0)
                run.assert_called_once_with(part, ['--mic=builtin'])

    def test_named_single_scenarios(self):
        for scenario in ('introduction', 'game', 'reflection'):
            with self.subTest(scenario=scenario), patch('furhat_interaction.app.importlib.import_module',return_value=SimpleNamespace(main=lambda:0)) as load:
                self.assertEqual(main(['--people=1', '--scenario='+scenario]),0)
                load.assert_called_once_with('furhat_interaction.single_person.'+scenario)

    def test_paths_follow_repository_root(self):
        from pathlib import Path
        from furhat_interaction.paths import ROOT
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertTrue((ROOT/'models/manifest.json').is_file())

    def test_preflight_routes_without_loading_a_scene(self):
        with patch('furhat_interaction.app.check',return_value=0) as check,patch('furhat_interaction.app.importlib.import_module') as load:
            self.assertEqual(main(['--check']),0)
        check.assert_called_once_with();load.assert_not_called()


if __name__=='__main__': unittest.main()
