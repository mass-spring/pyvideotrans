"""
Tests for the CLI speaker-diarization options (--diarize / --enable_diariz, --num-speakers, --speaker-type).

All mock-based: no model, audio or GPU is used.
Background: TransCreate.diariz() (vtv) does nothing unless trk.do_diarize is True and the GUI is the only place
that set it, so `cli.py --task vtv --enable_diariz` never produced speaker.json. The CLI now sets it.
"""

import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cli
from cli import SPEAKER_TYPES, TEXT_DB, build_parser, resolve_diariz_args, vtv_fun, stt_fun


def parse(tmp_path, *argv, task="stt"):
    f = tmp_path / "demo.mp4"
    f.touch()
    base = ["--task", task, "--name", str(f)]
    if task == "vtv":
        base += ["--source_language_code", "ja", "--target_language_code", "en"]
    parser = build_parser()
    return parser, parser.parse_args(base + list(argv))


# ---------------------------------------------------------------------------
# parser: upstream defaults unchanged, new options present
# ---------------------------------------------------------------------------
class TestParser:
    def test_defaults_unchanged(self, tmp_path):
        _, args = parse(tmp_path)
        assert args.enable_diariz is False
        assert args.nums_diariz == 0
        assert args.num_speakers is None
        assert args.speaker_type is None
        assert args.voice_profile_dir is None

    def test_diarize_is_alias_of_enable_diariz(self, tmp_path):
        _, a = parse(tmp_path, "--diarize")
        _, b = parse(tmp_path, "--enable_diariz")
        assert a.enable_diariz is True and b.enable_diariz is True

    def test_legacy_nums_diariz_still_parsed(self, tmp_path):
        _, args = parse(tmp_path, "--enable_diariz", "--nums_diariz", "3")
        assert args.enable_diariz is True and args.nums_diariz == 3

    def test_num_speakers_and_speaker_type(self, tmp_path):
        _, args = parse(tmp_path, "--num-speakers", "4", "--speaker-type", "pyannote")
        assert args.num_speakers == 4 and args.speaker_type == "pyannote"

    def test_speaker_type_choices_match_gui_settings(self):
        assert SPEAKER_TYPES == ["built", "ali_CAM", "pyannote", "enrolled"]
        # the GUI's advanced settings offer the same list
        text = Path(__file__).resolve().parents[1].joinpath("videotrans/ui/advset_keys.py").read_text(encoding="utf-8")
        assert "'speaker_type': ['built', 'ali_CAM', 'pyannote', 'enrolled']" in text.replace('"', "'")

    def test_speaker_type_invalid_choice(self, tmp_path):
        with pytest.raises(SystemExit):
            parse(tmp_path, "--speaker-type", "nope")

    def test_new_text_db_entries_are_bilingual_and_consistent(self):
        for key in ("help_num_speakers", "help_speaker_type", "err_num_speakers_neg", "err_num_speakers_conflict",
                    "err_diariz_hf_token", "warn_num_speakers_one", "warn_speaker_type_ignored",
                    "warn_vtv_diariz_needs_dubbing", "help_voice_profile_dir", "err_enrolled_profiles"):
            assert key in TEXT_DB and "zh" in TEXT_DB[key] and "en" in TEXT_DB[key]
            assert TEXT_DB[key]["zh"].count("{}") == TEXT_DB[key]["en"].count("{}")


# ---------------------------------------------------------------------------
# resolve_diariz_args
# ---------------------------------------------------------------------------
class TestResolveDiarizArgs:
    def test_nothing_given_leaves_args_untouched(self, tmp_path):
        parser, args = parse(tmp_path)
        resolve_diariz_args(args, parser)
        assert (args.enable_diariz, args.nums_diariz) == (False, 0)

    @pytest.mark.parametrize("n,expected_idx", [(2, 1), (3, 2), (4, 3), (10, 9)])
    def test_num_speakers_maps_to_gui_index(self, tmp_path, n, expected_idx):
        parser, args = parse(tmp_path, "--num-speakers", str(n))
        resolve_diariz_args(args, parser)
        assert args.enable_diariz is True
        assert args.nums_diariz == expected_idx
        # TransCreate / SpeechToText turn the index back into the number of speakers
        max_speakers = max(args.nums_diariz, 0) + 1
        assert max_speakers == n

    def test_num_speakers_zero_means_unlimited(self, tmp_path):
        parser, args = parse(tmp_path, "--num-speakers", "0")
        resolve_diariz_args(args, parser)
        assert args.enable_diariz is True and args.nums_diariz == 0

    def test_num_speakers_one_skips_diarization(self, tmp_path, capsys):
        parser, args = parse(tmp_path, "--diarize", "--num-speakers", "1")
        resolve_diariz_args(args, parser)
        assert args.enable_diariz is False
        err = capsys.readouterr().err
        assert "--num-speakers 1" in err

    def test_negative_num_speakers_is_an_error(self, tmp_path):
        parser, args = parse(tmp_path, "--num-speakers", "-2")
        with pytest.raises(SystemExit):
            resolve_diariz_args(args, parser)

    def test_num_speakers_and_nums_diariz_conflict(self, tmp_path):
        parser, args = parse(tmp_path, "--num-speakers", "4", "--nums_diariz", "2")
        with pytest.raises(SystemExit):
            resolve_diariz_args(args, parser)

    def test_pyannote_without_token_fails_loudly(self, tmp_path):
        parser, args = parse(tmp_path, "--diarize", "--speaker-type", "pyannote")
        with pytest.raises(SystemExit):
            resolve_diariz_args(args, parser, hf_token="")
        parser, args = parse(tmp_path, "--diarize", "--speaker-type", "pyannote")
        resolve_diariz_args(args, parser, hf_token="hf_xxx")  # token present: fine

    def test_pyannote_from_settings_without_token_fails(self, tmp_path):
        parser, args = parse(tmp_path, "--diarize")
        with pytest.raises(SystemExit):
            resolve_diariz_args(args, parser, hf_token="", default_speaker_type="pyannote")

    def test_built_and_ali_cam_need_no_token(self, tmp_path):
        for st in ("built", "ali_CAM"):
            parser, args = parse(tmp_path, "--diarize", "--speaker-type", st)
            resolve_diariz_args(args, parser, hf_token="")

    @staticmethod
    def _profile_dir(root):
        folder = root / "profiles" / "Aさん"
        (folder / "refs").mkdir(parents=True)
        (folder / "refs" / "a.wav").write_bytes(b"RIFF")
        (folder / "profile.json").write_text('{"refs": [{"wav": "refs/a.wav"}]}', encoding="utf-8")
        return str(root / "profiles")

    def test_enrolled_without_profiles_fails_loudly_in_japanese(self, tmp_path, capsys):
        parser, args = parse(tmp_path, "--diarize", "--speaker-type", "enrolled")
        with pytest.raises(SystemExit) as e:
            resolve_diariz_args(args, parser, voice_profile_dir="")
        assert e.value.code == 2
        err = capsys.readouterr().err
        assert "--speaker-type enrolled" in err and "声の登録フォルダが指定されていません" in err

    def test_enrolled_from_settings_default_also_needs_profiles(self, tmp_path):
        parser, args = parse(tmp_path, "--diarize")
        with pytest.raises(SystemExit):
            resolve_diariz_args(args, parser, default_speaker_type="enrolled", voice_profile_dir=str(tmp_path / "missing"))

    def test_enrolled_folder_without_any_profile(self, tmp_path, capsys):
        parser, args = parse(tmp_path, "--diarize", "--speaker-type", "enrolled", "--voice-profile-dir", str(tmp_path))
        with pytest.raises(SystemExit):
            resolve_diariz_args(args, parser)
        assert "1人も見つかりません" in capsys.readouterr().err

    def test_enrolled_with_profiles_from_settings_or_option_is_fine(self, tmp_path):
        d = self._profile_dir(tmp_path)
        parser, args = parse(tmp_path, "--diarize", "--speaker-type", "enrolled")
        resolve_diariz_args(args, parser, voice_profile_dir=d)  # cfg.json value
        parser, args = parse(tmp_path, "--diarize", "--speaker-type", "enrolled", "--voice-profile-dir", d)
        resolve_diariz_args(args, parser, voice_profile_dir="")  # option wins over an empty setting
        assert args.enable_diariz is True

    def test_enrolled_not_checked_when_diarization_is_off(self, tmp_path):
        parser, args = parse(tmp_path, "--speaker-type", "enrolled")
        resolve_diariz_args(args, parser, voice_profile_dir="")  # only the note, no error

    def test_enrolled_needs_no_hf_token(self, tmp_path):
        d = self._profile_dir(tmp_path)
        parser, args = parse(tmp_path, "--diarize", "--speaker-type", "enrolled")
        resolve_diariz_args(args, parser, hf_token="", voice_profile_dir=d)

    def test_token_not_checked_when_diarization_is_off(self, tmp_path):
        parser, args = parse(tmp_path, "--speaker-type", "pyannote")
        resolve_diariz_args(args, parser, hf_token="")  # only a note, no error

    def test_speaker_type_without_diarize_prints_note(self, tmp_path, capsys):
        parser, args = parse(tmp_path, "--speaker-type", "built")
        resolve_diariz_args(args, parser)
        assert "--diarize" in capsys.readouterr().err

    def test_vtv_without_voice_role_prints_note(self, tmp_path, capsys):
        parser, args = parse(tmp_path, "--diarize", task="vtv")
        resolve_diariz_args(args, parser)
        assert "--task stt" in capsys.readouterr().err
        parser, args = parse(tmp_path, "--diarize", "--voice_role", "en-US-GuyNeural", task="vtv")
        resolve_diariz_args(args, parser)
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("task", ["tts", "sts"])
    def test_other_tasks_are_ignored(self, tmp_path, task):
        args = Namespace(task=task, num_speakers=4, nums_diariz=0, enable_diariz=False, speaker_type="pyannote", voice_role=None)
        resolve_diariz_args(args, build_parser(), hf_token="")
        assert args.enable_diariz is False and args.nums_diariz == 0

    def test_params_reach_the_task_config(self, tmp_path):
        """--num-speakers 4 --diarize ends up as enable_diariz/nums_diariz in the stt and vtv parameter dicts."""
        for task in ("stt", "vtv"):
            parser, args = parse(tmp_path, "--num-speakers", "4", task=task)
            resolve_diariz_args(args, parser)
            args.cuda = False
            params = cli.build_stt_params(args) if task == "stt" else cli.build_vtv_params(args)
            assert params["enable_diariz"] is True and params["nums_diariz"] == 3


# ---------------------------------------------------------------------------
# vtv_fun: do_diarize must be True before diariz() is called
# ---------------------------------------------------------------------------
def _fake_task_modules(trk):
    trans_create = types.ModuleType("videotrans.task.trans_create")
    trans_create.TransCreate = MagicMock(return_value=trk)
    taskcfg = types.ModuleType("videotrans.task.taskcfg")
    taskcfg.TaskCfgVTT = MagicMock()
    config = types.ModuleType("videotrans.configure.config")
    config.app_cfg = MagicMock()
    return {
        "videotrans.task.trans_create": trans_create,
        "videotrans.task.taskcfg": taskcfg,
        "videotrans.configure.config": config,
    }


class TestVtvFun:
    def test_do_diarize_is_set_before_diariz_runs(self):
        order = []
        trk = MagicMock()
        trk.diariz.side_effect = lambda: order.append(("diariz", trk.do_diarize))
        for name in ("prepare", "recogn", "trans", "dubbing", "align", "recogn2pass", "assembling", "task_done"):
            getattr(trk, name).side_effect = (lambda n=name: order.append((n, None)))
        with patch.dict(sys.modules, _fake_task_modules(trk)):
            vtv_fun({"name": "x.mp4", "enable_diariz": True})
        assert ("diariz", True) in order
        names = [n for n, _ in order]
        assert names == ["prepare", "recogn", "diariz", "trans", "dubbing", "align", "recogn2pass", "assembling", "task_done"]

    def test_task_cfg_receives_diarization_params(self):
        trk = MagicMock()
        mods = _fake_task_modules(trk)
        with patch.dict(sys.modules, mods):
            vtv_fun({"name": "x.mp4", "enable_diariz": True, "nums_diariz": 3})
        mods["videotrans.task.taskcfg"].TaskCfgVTT.assert_called_once_with(name="x.mp4", enable_diariz=True, nums_diariz=3)


class TestSttFun:
    def test_diariz_runs_between_recogn_and_task_done(self):
        order = []
        trk = MagicMock()
        for name in ("prepare", "recogn", "diariz", "task_done"):
            getattr(trk, name).side_effect = (lambda n=name: order.append(n))
        mod = types.ModuleType("videotrans.task.speech2text")
        mod.SpeechToText = MagicMock(return_value=trk)
        cfgmod = types.ModuleType("videotrans.task.taskcfg")
        cfgmod.TaskCfgSTT = MagicMock()
        with patch.dict(sys.modules, {"videotrans.task.speech2text": mod, "videotrans.task.taskcfg": cfgmod}):
            stt_fun({"name": "x.mp4", "enable_diariz": True})
        assert order == ["prepare", "recogn", "diariz", "task_done"]


# ---------------------------------------------------------------------------
# The gate that made the flag necessary (real mixin, dummy host)
# ---------------------------------------------------------------------------
class TestDiarizMixinGate:
    def _host(self, tmp_path, do_diarize, enable_diariz=True, should_dubbing=True, max_speakers=0):
        from videotrans.task._stage_diariz import DiarizMixin

        class Host(DiarizMixin):
            calls = 0

            def _exit(self):
                return False

            def _diariz_common(self, *a, **kw):
                Host.calls += 1

        h = Host()
        h.cfg = types.SimpleNamespace(enable_diariz=enable_diariz, cache_folder=str(tmp_path))
        h.should_dubbing = should_dubbing
        h.do_diarize = do_diarize
        h.max_speakers = max_speakers
        return h, Host

    def test_skipped_when_do_diarize_false(self, tmp_path):
        h, Host = self._host(tmp_path, do_diarize=False)
        h.diariz()
        assert Host.calls == 0

    def test_runs_when_do_diarize_true(self, tmp_path):
        h, Host = self._host(tmp_path, do_diarize=True)
        h.diariz()
        assert Host.calls == 1

    def test_not_run_when_disabled_or_single_speaker_or_no_dubbing(self, tmp_path):
        for kw in ({"enable_diariz": False}, {"max_speakers": 1}, {"should_dubbing": False}):
            h, Host = self._host(tmp_path, do_diarize=True, **kw)
            h.diariz()
            assert Host.calls == 0, kw

    def test_not_run_when_speaker_json_exists(self, tmp_path):
        (tmp_path / "speaker.json").write_text("[]")
        h, Host = self._host(tmp_path, do_diarize=True)
        h.diariz()
        assert Host.calls == 0


# ---------------------------------------------------------------------------
# main(): wiring (everything heavy is mocked)
# ---------------------------------------------------------------------------
class TestMainWiring:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        from videotrans.configure import config

        monkeypatch.setattr(config, "init_run", lambda: None)
        # restore global state touched by main()
        monkeypatch.setattr(config.app_cfg, "exit_soft", config.app_cfg.exit_soft, raising=False)
        monkeypatch.setattr(config.app_cfg, "exec_mode", getattr(config.app_cfg, "exec_mode", "gui"), raising=False)
        monkeypatch.setattr(config.settings, "speaker_type", "built", raising=False)
        monkeypatch.setattr(config.settings, "hf_token", "", raising=False)
        gpus = types.ModuleType("videotrans.util.gpus")
        gpus.getset_gpu = lambda: None
        monkeypatch.setitem(sys.modules, "videotrans.util.gpus", gpus)
        monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
        monkeypatch.setattr(cli, "build_common_params", lambda args, output_dir=None: {"name": args.name, "target_dir": str(tmp_path)})
        seen = {}

        def make(name):
            def fn(params):
                seen["task"] = name
                seen["params"] = params
                seen["speaker_type"] = config.settings.speaker_type
            return fn

        for n in ("stt", "vtv"):
            monkeypatch.setattr(cli, f"{n}_fun", make(n))
        f = tmp_path / "demo.mp4"
        f.touch()
        return types.SimpleNamespace(config=config, seen=seen, file=str(f), monkeypatch=monkeypatch)

    def run(self, env, *argv):
        env.monkeypatch.setattr(sys, "argv", ["cli.py", *argv])
        return cli.main()

    def test_upstream_defaults(self, env):
        assert self.run(env, "--task", "stt", "--name", env.file) == 0
        assert env.seen["params"]["enable_diariz"] is False and env.seen["params"]["nums_diariz"] == 0
        assert env.seen["speaker_type"] == "built"

    def test_diarize_with_num_speakers(self, env):
        assert self.run(env, "--task", "stt", "--name", env.file, "--diarize", "--num-speakers", "4", "--speaker-type", "ali_CAM") == 0
        p = env.seen["params"]
        assert p["enable_diariz"] is True and p["nums_diariz"] == 3
        assert env.seen["speaker_type"] == "ali_CAM"                    # applied while the task runs ...
        assert env.config.settings.speaker_type == "built"              # ... and restored afterwards (never saved)

    def test_speaker_type_restored_even_when_the_task_fails(self, env):
        def boom(params):
            raise RuntimeError("boom")
        env.monkeypatch.setattr(cli, "stt_fun", boom)
        assert self.run(env, "--task", "stt", "--name", env.file, "--diarize", "--speaker-type", "ali_CAM") == 1
        assert env.config.settings.speaker_type == "built"

    def test_voice_profile_dir_option_applies_to_this_run_only(self, env, tmp_path):
        config = env.config
        env.monkeypatch.setattr(config.settings, "voice_profile_dir", "/original", raising=False)
        d = tmp_path / "profiles" / "Aさん"
        (d / "refs").mkdir(parents=True)
        (d / "refs" / "a.wav").write_bytes(b"RIFF")
        (d / "profile.json").write_text('{"refs": [{"wav": "refs/a.wav"}]}', encoding="utf-8")
        seen = {}
        env.monkeypatch.setattr(cli, "stt_fun", lambda params: seen.update(profile_dir=config.settings.voice_profile_dir,
                                                                           speaker_type=config.settings.speaker_type))
        assert self.run(env, "--task", "stt", "--name", env.file, "--diarize", "--speaker-type", "enrolled",
                        "--voice-profile-dir", str(tmp_path / "profiles")) == 0
        assert seen["speaker_type"] == "enrolled"
        assert Path(seen["profile_dir"]) == tmp_path / "profiles"        # applied while the task runs ...
        assert config.settings.voice_profile_dir == "/original"          # ... and restored afterwards (never saved)
        assert config.settings.speaker_type == "built"

    def test_enrolled_without_profiles_exits_with_usage_error(self, env):
        env.monkeypatch.setattr(env.config.settings, "voice_profile_dir", "", raising=False)
        with pytest.raises(SystemExit) as e:
            self.run(env, "--task", "stt", "--name", env.file, "--diarize", "--speaker-type", "enrolled")
        assert e.value.code == 2
        assert "task" not in env.seen          # the task never started

    def test_vtv_gets_diarization_params(self, env):
        assert self.run(env, "--task", "vtv", "--name", env.file, "--source_language_code", "ja", "--target_language_code", "en",
                        "--voice_role", "en-US-GuyNeural", "--num-speakers", "3") == 0
        p = env.seen["params"]
        assert env.seen["task"] == "vtv" and p["enable_diariz"] is True and p["nums_diariz"] == 2

    def test_pyannote_without_token_exits_with_usage_error(self, env):
        with pytest.raises(SystemExit) as e:
            self.run(env, "--task", "stt", "--name", env.file, "--diarize", "--speaker-type", "pyannote")
        assert e.value.code == 2
        assert "task" not in env.seen          # the task never started
