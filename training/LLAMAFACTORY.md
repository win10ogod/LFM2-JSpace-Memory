# LlamaFactory integration used for this SFT

The Trainer, preprocessing processor, and optimizer remain native. These are the integration points in the user-maintained LlamaFactory fork. Apply them against a compatible checkout; this document is not a claim that an arbitrary upstream version accepts the custom model unchanged.

## model/loader.py

```python
        else:
            if (type(config) in AutoModelForImageTextToText._model_mapping.keys()
                    or "AutoModelForImageTextToText" in getattr(config,"auto_map",{})):  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text for qwen omni
                load_class = AutoModelForTextToWaveform
```

```python
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
    # Architecture-owned tensor adapters supplement ordinary PEFT linear LoRA.
    # Data loading, batching and the Trainer remain the standard SFT path.
    if is_trainable and hasattr(model,"load_sft_memory_adapters"):
        if not model_args.adapter_name_or_path:
            raise ValueError("LFM2 memory SFT requires its complete saved LoRA initialization.")
        model.load_sft_memory_adapters(model_args.adapter_name_or_path[-1])

```

## train/sft/workflow.py

```python
    )

    if training_args.do_train and hasattr(model,"native_sft_callback"):
        trainer.add_callback(model.native_sft_callback(training_args.resume_from_checkpoint))

    # Training
    if training_args.do_train:
```

## model/model_utils/checkpointing.py

```python
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning_rank0("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            if getattr(model.config,"model_type",None)=="lfm2_titans":
                # This architecture restores each row's physical FFN memory
                # during recomputation. Replacing its method loses that state.
                model_args.use_reentrant_gc=False
            else:
                gradient_checkpointing_enable = partial(
                    _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
                )
                model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=_get_gradient_checkpointing_kwargs(model_args)
            )
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info_rank0("Gradient checkpointing enabled.")

```

## data/processor/supervised.py

```python
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            retained_images = examples["_images"][i]
            finalize = getattr(self.template.mm_plugin, "finalize_truncated_sample", None)
            if finalize is not None:
                input_ids, labels, retained_images = finalize(input_ids, labels, retained_images or [], self.tokenizer)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(retained_images)
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs
```

## data/collator.py

```python
        fake_input_ids = []
        has_dummy_image = False
        if (
            self.template.mm_plugin.image_token is not None
            and sum(batch_imglens) == 0
            and sum(batch_vidlens) == 0
            and not is_moss_vl  # MOSS-VL builds one native zero-valued dummy per text-only sample in its plugin.
            and model_type != "lfm2_titans"  # A fabricated image must not enter persistent memory.
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
```

## data/mm_plugin.py

Use the native HF image expansion and align an incomplete image span after native truncation. Add `from ..extras.logging import get_logger` and `logger = get_logger(__name__)` if absent.

```python
class LFMVLPlugin(BasePlugin):
    r"""Plugin for LFM2.5-VL vision-language models.

    LFM2.5-VL uses dynamic image token counts based on image resolution.
    The image processor returns spatial_shapes tensor with [height, width] grid dimensions.
    Token count per image = (spatial_h * spatial_w) / (downsample_factor^2)
    """

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            native = image_processor(images, return_tensors="pt")
            for key in ("image_rows", "image_cols", "image_sizes"):
                native.pop(key, None)
            mm_inputs.update(native)
        return mm_inputs

    def finalize_truncated_sample(self, input_ids, labels, images, tokenizer):
        """Keep only complete native image spans after ordinary cutoff truncation."""
        cached = getattr(self, "_lfm_image_token_cache", None)
        if cached is None or cached[0] != id(tokenizer):
            vocab = tokenizer.get_vocab()
            start, end = vocab["<|image_start|>"], vocab["<|image_end|>"]
            image_tokens = {value for token, value in vocab.items()
                            if token == "<image>" or token.startswith(("<|image_", "<|img_"))}
            cached = (id(tokenizer), start, end, image_tokens)
            self._lfm_image_token_cache = cached
        _, start, end, image_tokens = cached
        opened = None
        complete = 0
        remove = set()
        partial_seen = False
        for i, token in enumerate(input_ids):
            if opened is not None and token not in image_tokens:
                remove.update(range(opened, i)); opened = None; partial_seen = True
            if token == start:
                if opened is not None:
                    raise ValueError("Nested native image span")
                opened = i
            elif token == end:
                if opened is None or partial_seen:
                    raise ValueError("Truncated images are not a retained source prefix")
                complete += 1; opened = None
        if opened is not None:
            remove.update(range(opened, len(input_ids)))
        if complete > len(images):
            raise ValueError("More complete image spans than image sources")
        if images and not complete:
            raise ValueError("The first complete image does not fit cutoff_len; adjust the native input budget")
        if remove or complete != len(images):
            logger.warning_rank0(f"Native LFM cutoff: retained {complete}/{len(images)} complete images; removed {len(remove)} incomplete image tokens.")
        return ([v for i, v in enumerate(input_ids) if i not in remove],
                [v for i, v in enumerate(labels) if i not in remove], images[:complete])

    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if not self.expand_mm_tokens or not images:
            return messages
        loaded = self._regularize_images(
            images, image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
            image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
        )["images"]
        native = processor.image_processor(loaded, return_tensors="pt")
        grouped = []
        offset = 0
        for message in messages:
            count = message["content"].count(IMAGE_PLACEHOLDER)
            grouped.append(loaded[offset:offset + count])
            offset += count
        expanded = processor.expand_text_with_placeholders(
            [message["content"] for message in messages], grouped,
            image_rows=native["image_rows"], image_cols=native["image_cols"],
            image_sizes=native["image_sizes"], use_image_special_tokens=True,
        )
        for message, text in zip(messages, expanded):
            message["content"] = text
        return messages
```
