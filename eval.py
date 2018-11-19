'''
eval.py
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
'''
import os
import numpy as np
import util.audio as audio
import util.util as ut
from warnings import warn
from config import config


def eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker, _frontend):
    # harded coded
    texts = [
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There's a way to measure the acute emotional intelligence that has never gone out of style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "Generative adversarial network or variational auto-encoder.",
        "Please call Stella.",
        "Some have accepted this as a miracle without any physical explanation.",
    ]
    import synthesis
    synthesis._frontend = _frontend

    eval_output_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare model for evaluation
    model_eval = ut.build_model().to(device)
    model_eval.load_state_dict(model.state_dict())

    # hard coded
    speaker_ids = [0, 1, 10] if ismultispeaker else [None]
    for speaker_id in speaker_ids:
        speaker_str = "multispeaker{}".format(speaker_id) if speaker_id is not None else "single"

        for idx, text in enumerate(texts):
            signal, alignment, _, mel = synthesis.tts(
                model_eval, text, p=0, speaker_id=speaker_id, fast=True)
            signal /= np.max(np.abs(signal))

            # Alignment
            path = os.path.join(eval_output_dir, "step{:09d}_text{}_{}_alignment.png".format(
                global_step, idx, speaker_str))
            ut.save_alignment(path, alignment, global_step)
            tag = "eval_averaged_alignment_{}_{}".format(idx, speaker_str)
            # writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255), global_step)

            # Mel
            # writer.add_image("(Eval) Predicted mel spectrogram text{}_{}".format(idx, speaker_str),
                             # prepare_spec_image(mel), global_step)

            # Audio
            path = os.path.join(eval_output_dir, "step{:09d}_text{}_{}_predicted.wav".format(
                global_step, idx, speaker_str))
            audio.save_wav(signal, path)

            try:
                writer.add_audio("(Eval) Predicted audio signal {}_{}".format(idx, speaker_str),
                                 signal, global_step, sample_rate=config.sample_rate)
            except Exception as e:
                warn(str(e))
                pass
