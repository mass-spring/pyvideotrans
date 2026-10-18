from dataclasses import dataclass
from typing import List, Dict, Union

from gradio_client import handle_file

from videotrans.configure.config import params, settings
from videotrans.tts._gradio import GradioBase
from videotrans.util.emotion_tts import voxcpm_control_instruction
from videotrans.util.help_misc import vail_file


@dataclass(repr=False)
class VoxCPMTTS(GradioBase):
    def __post_init__(self):
        self.ainame = "voxcpm"
        super().__post_init__()

    def _run(self, data_item: Union[Dict, List, None], idx: int = -1) -> Union[str, None]:
        if vail_file(data_item['filename']):return
        kwargs = {
            "do_normalize": False,
            "denoise": False,
            "api_name": '/generate'
        }
        _version = params.get('voxcpmtts_version', 'v2')
        text=data_item['text'].strip()
        ref_wav,ref_text=self.get_ref_wav(data_item)
        # emotion of this line as a VoxCPM2 control instruction ('' = none = upstream). The VoxCPM app ignores the instruction
        # while "use_prompt_text" (transcript-guided cloning) is on, so an emotional line is cloned from the reference clip only
        control = voxcpm_control_instruction(data_item, settings.get('voxcpm_emotion_priority', 'emotion'))
        use_prompt_text = True if (ref_text and not control) else False
        if _version == 'v2':
            kwargs['text'] = text
            kwargs['control_instruction'] = control
            kwargs['use_prompt_text'] = use_prompt_text
            kwargs['ref_wav'] = handle_file(ref_wav)
            kwargs['dit_steps'] = 10
            kwargs["cfg_value"] = 2.0
            #kwargs["seed_value"]=42
            kwargs["prompt_text_value"] = ref_text
        elif _version == 'hf':
            kwargs['text_input'] = text
            kwargs['control_instruction'] = control
            kwargs['use_prompt_text'] = use_prompt_text
            kwargs['reference_wav_path_input'] = handle_file(ref_wav)
            kwargs["cfg_value_input"] = 2
            kwargs["prompt_text_input"] = ref_text
        else:
            kwargs['text_input'] = text
            kwargs['prompt_wav_path_input'] = handle_file(ref_wav)
            kwargs['inference_timesteps_input'] = 10
            kwargs["cfg_value_input"] = 2
            kwargs["prompt_text_input"] = ref_text
        return self._send(kwargs,data_item)
