import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from transformers import T5EncoderModel, T5Tokenizer


class LightDreamModel:
    _log = logging.getLogger("LightDreamModel")

    def __init__(
        self,
        text_encoder: "T5Encoder",
        unet: "LightDreamUnetTorchScriptModel",
    ):
        """
        Constructor.
        :param text_encoder: The T5 XXL text encoder.
        :param unet: The UNet (as a TorchScript model).
        """

        self.text_encoder = text_encoder
        self.unet = unet

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "LightDreamModel":
        """
        Load weights from a snapshot.
        :param path: The path to the snapshot.
        :param device: The device to load the model to.
        """

        path = Path(path)
        model = cls(
            text_encoder=cls._load_text_encoder(path),
            unet=cls._load_unet(path),
        )

        if device is not None:
            model.to(device)

        return model

    def to(self, device: str) -> "LightDreamModel":
        """
        Move model to device.
        :param device: Device to move model to.
        """
        self.text_encoder.to(device)
        self.unet.to(device)
        return self

    @classmethod
    def _load_text_encoder(cls, path: Path) -> "T5Encoder":
        start_time = time.time()
        text_encoder = T5Encoder(
            model_path=path / "google-t5-v1_1-xxl",
            dtype=torch.float32,
        )
        cls._log.debug(f"Loaded the text encoder in {time.time() - start_time:.2f} seconds")
        return text_encoder

    @classmethod
    def _load_unet(cls, path: Path) -> "LightDreamUnetTorchScriptModel":
        start_time = time.time()
        unet_path = str(path / "unet" / "model.ts")
        unet = LightDreamUnetTorchScriptModel(unet_path)
        cls._log.debug(f"Loaded the UNet in {time.time() - start_time:.2f} seconds")
        return unet


class T5Encoder:
    def __init__(
        self,
        model_path: Union[str, Path],
        max_length: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self._model_path = Path(model_path)
        self._max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._load_model()

    @property
    def embedding_dim(self) -> int:
        return self._encoder.shared.embedding_dim

    @property
    def max_sequence_length(self) -> int:
        return self._max_length

    def _load_model(self) -> None:
        tokenizer_path = Path(self._model_path) / "tokenizer"
        self._tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

        encoder_path = Path(self._model_path) / "encoder"
        self._encoder = T5EncoderModel.from_pretrained(encoder_path)

        if self.dtype == torch.bfloat16:
            self._encoder.to(self.dtype)
        elif self.dtype == torch.half:
            self._encoder.half()

        self._encoder.eval()

    def to(self, device: str) -> "T5Encoder":
        """
        Move model to device.
        :param device: Device to move model to.
        """
        self._encoder.to(device=device)
        return self

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        """
        Tokenize and encode text.
        Return a padded sequence for each text and the length of each sequence before padding.
        """
        encoded = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding="max_length",  # We use 'max_length' padding to avoid recompilation of the graph on TPUs.
            max_length=self._max_length,
            truncation=True,
        )

        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        encoded_text = self._encode(input_ids=input_ids, attention_mask=attention_mask)

        sequence_lengths = attention_mask.sum(dim=1, keepdim=False)
        attention_mask = attention_mask.to(torch.bool)
        inverse_attention_mask = ~attention_mask
        encoded_text = encoded_text.masked_fill(inverse_attention_mask.unsqueeze(dim=2), 0.0)

        return encoded_text, sequence_lengths

    @staticmethod
    @torch.no_grad()
    def unpad(padded_sequence: Tensor, sequence_lengths: Tensor) -> List[np.ndarray]:
        """
        Removes padding from a batch of sequences.
        Returns a list of numpy arrays, each holds a sequence with a dynamic length.
        """
        padded_sequence = padded_sequence.to(torch.float32).detach().cpu().numpy()
        sequence_lengths = sequence_lengths.detach().cpu().numpy()

        results = []
        for idx, sequence_length in enumerate(sequence_lengths):
            results.append(padded_sequence[idx][0:sequence_length])
        return results

    def _encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_text = output.last_hidden_state
        return encoded_text


class LightDreamUnetTorchScriptModel:
    def __init__(self, path: Union[str, Path]):
        self._model = torch.jit.load(path, map_location="cpu")

    def to(self, device: str) -> "LightDreamUnetTorchScriptModel":
        self._model.to(device)
        return self

    def __call__(
        self,
        x: Tensor,
        timestep: Tensor,
        text_encoder_seq: Tensor,
        text_encoder_last: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        return self._model(
            x,
            timestep,
            text_encoder_seq,
            text_encoder_last,
            attention_mask,
        )
