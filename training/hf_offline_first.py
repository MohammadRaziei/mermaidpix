"""
hf_offline_first.py

Fixes a reported bug: `transformers.from_pretrained(...)`, even for a
checkpoint that's already fully downloaded and cached locally, still lets
`huggingface_hub` make a network round-trip (an ETag/metadata check)
*every time it's called* -- unless told not to. On a flaky connection,
that surfaces as the process repeatedly "wanting to reconnect to
HuggingFace" for something it already has on disk and never actually
needed to redownload.

`from_pretrained_offline_first` tries the local cache FIRST (zero network
calls, `local_files_only=True`) and only reaches out to the network if
that fails because the checkpoint genuinely isn't cached yet -- so the
very first run (nothing cached) still downloads normally, but every run
after that never touches the network for this checkpoint again.
"""
from __future__ import annotations


def from_pretrained_offline_first(model_or_processor_cls, checkpoint: str, **kwargs):
    """model_or_processor_cls: any transformers class with a
    `.from_pretrained` classmethod (VisionEncoderDecoderModel,
    AutoImageProcessor, etc.) -- this isn't specific to one of them.
    """
    try:
        return model_or_processor_cls.from_pretrained(checkpoint, local_files_only=True, **kwargs)
    except OSError:
        # Genuinely not cached yet (first run ever, or a different machine/
        # fresh venv) -- this IS a real, one-time, legitimate download.
        return model_or_processor_cls.from_pretrained(checkpoint, local_files_only=False, **kwargs)
