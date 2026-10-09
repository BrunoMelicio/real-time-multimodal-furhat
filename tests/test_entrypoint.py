"""Entrypoint routing checks; no hardware/model owners are started."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
from furhat_interaction.app import main


class EntryPoint(unittest.TestCase):
    def test_multi_forwards_only_scene_arguments(self):
        with patch('conference_ready_multi.runtime.run',return_value=0) as run:
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
        load.assert_called_once_with('conference_ready_2')

    def test_preflight_routes_without_loading_a_scene(self):
        with patch('furhat_interaction.app.check',return_value=0) as check,patch('furhat_interaction.app.importlib.import_module') as load:
            self.assertEqual(main(['--check']),0)
        check.assert_called_once_with();load.assert_not_called()


if __name__=='__main__': unittest.main()
