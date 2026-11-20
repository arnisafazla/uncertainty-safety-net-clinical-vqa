
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Dict, Any, List
from abc import ABC
import transformers
from transformers import GenerationConfig

print("Visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

class BaseModelHandler(ABC):
    """Base class for all model handlers"""
    
    def __init__(self, model_config: dict):
        self.model_id = model_config.get("model_id", None)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.max_new_tokens = model_config.get("max_new_tokens", 64)

        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        self.num_beams = model_config.get("num_beams", 1)

        self.save_full_distributions = model_config.get("save_full_distributions", False)

        self.gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            num_beams=self.num_beams
        )

    def load_model(self, model_path: str, processor_class, model_class, model_precision=torch.bfloat16):
        """Load the model, tokenizer, and processor"""
        processor_class = getattr(transformers, processor_class, None)
        model_class = getattr(transformers, model_class, None)

        self.processor = processor_class.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map='auto'
        )
        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=model_precision,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.gen_config.pad_token_id = self.tokenizer.pad_token_id

    def format_inputs(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": batch_item["prompt"]},
                    {"type": "image"},
                ],
            }
        ]
        
        return {
            **batch_item,
            "messages": messages,
            "image": Image.open(batch_item["image_path"])
        }

    def prepare_inputs(self, batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        assert self.processor and self.tokenizer and self.model
        
        prompts_batch = [self.processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True) for item in batch]        
        images_batch = [item["image"] for item in batch]

        inputs = self.processor(
            images=images_batch,
            text=prompts_batch,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        return inputs, prompts_batch

    def run_batch_inference(self, inputs) -> List[Dict[str,Any]]:
        assert self.processor and self.tokenizer and self.model

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                generation_config=self.gen_config
            )

        return outputs

    def process_outputs(self, outputs, inputs, prompts, batch, return_input_logits: bool) -> List[Dict[str,Any]]:
        assert self.processor and self.tokenizer and self.model
        results = []
        input_ids = inputs["input_ids"]  # (batch_size, seq_len)
        
        for b_idx, item in enumerate(batch):
            input_len = input_ids[b_idx].shape[0]
            generated_tokens = outputs.sequences[b_idx, input_len:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logprobs = []
            full_dists = []
            for j, logits in enumerate(outputs.logits):
                if j >= generated_tokens.shape[0]:  # Avoid out-of-bounds if gen shorter than scores
                    break
                token_id = outputs.sequences[b_idx, input_len + j].item()
                probs = torch.nn.functional.softmax(logits[b_idx], dim=-1).detach().cpu()
                logprob = torch.log(probs[token_id]).item()
                token_str = self.tokenizer.decode([token_id])
                logprobs.append((token_str, logprob))

                if self.save_full_distributions:
                    full_dists.append(probs)

            prompt_logprobs = []
            if return_input_logits:
                # --- Compute logprobs for input prompt tokens ---
                with torch.no_grad():
                    prompt_outputs = self.model(input_ids[b_idx].unsqueeze(0))
                    prompt_logits = prompt_outputs.logits.squeeze(0)

                for j in range(1, input_len):  # skip first token (no previous context)
                    probs = torch.nn.functional.softmax(prompt_logits[j-1], dim=-1)
                    token_id = input_ids[b_idx][j].item()
                    logprob = torch.log(probs[token_id]).item()
                    token_str = self.tokenizer.decode([token_id])
                    prompt_logprobs.append((token_str, logprob))

            results.append({
                "item": item,
                "prompt": prompts[b_idx],
                "model_response": generated_text,
                "logprobs": logprobs,
                "input_logprobs": prompt_logprobs,
                "full_distributions": torch.stack(full_dists, dim=0) if self.save_full_distributions else None
            })
        
        return results

    def generate_response_single_input(self, item: Dict) -> Dict[str,Any]:
        assert self.processor and self.tokenizer and self.model

        prompt = self.processor.apply_chat_template(
            item["messages"],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        if "image" in item:
            image = item["image"]

            inputs = self.processor(
                images=[image],
                text=[prompt],
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

        inputs = self.processor(
            text=[prompt],
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return {"model_response": response, "item": item}
    
    def format_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of items for this model"""
        return [self.format_inputs(item) for item in batch]
    
    def print_device_map(self) -> None:
        """Get the device map for the model"""
        print("HuggingFace device map:", getattr(self.model, "hf_device_map", "No device map found"))

class ImageTextToTextModelHandler(BaseModelHandler):
    """Handler for LlavaNext models"""

class MedGemmaModelHandler(BaseModelHandler):
    """Handler for MedGemma models"""

    def format_inputs(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        image = Image.open(batch_item["image_path"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": batch_item["prompt"]},
                    {"type": "image", "image":image},
                ],
            }
        ]

        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        return {
            **batch_item,
            "messages": messages,
            "image": [image],
            "prompt_text": prompt_text
        }

    def prepare_inputs(self, batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        assert self.processor and self.tokenizer and self.model
        
        prompts_batch = [item["prompt_text"] for item in batch]
        images_batch = [item["image"] for item in batch]

        # Now collate everything in one go
        inputs = self.processor(
            text=prompts_batch,
            images=images_batch,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        return inputs, prompts_batch

    def process_outputs(self, outputs, inputs, prompts, batch, return_input_logits: bool) -> List[Dict[str,Any]]:
        assert self.processor and self.tokenizer and self.model
        input_ids = inputs["input_ids"]  # (batch_size, seq_len)

        results = []
        for b_idx, item in enumerate(batch):  # batch_data is your [(prompt, image_url), ...]
            # --- Prompt length for this item ---
            input_len = input_ids[b_idx].shape[0]

            # --- Slice generated tokens ---
            generated_tokens = outputs.sequences[b_idx, input_len:]

            # --- Decode ---
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

            # --- Compute logprobs for each generated token ---
            logprobs = []
            for j, logits in enumerate(outputs.logits):
                if j >= generated_tokens.shape[0]:
                    break
                token_id = outputs.sequences[b_idx, input_len + j].item()
                probs = torch.nn.functional.softmax(logits[b_idx], dim=-1)
                logprob = torch.log(probs[token_id]).item()
                token_str = self.processor.tokenizer.decode([token_id])
                logprobs.append((token_str, logprob))

            if logprobs[-1][0] == "[":
                print("Warning: Last token is '[', early stopping.")
                results.append({
                    "item": item,
                    "prompt": item["prompt_text"],  # original text prompt
                    "model_response": generated_text,
                    "logprobs": logprobs,
                    "input_logprobs": None
                })
                return results

            results.append({
                "item": item,
                "prompt": item["prompt_text"],  # original text prompt
                "model_response": generated_text,
                "logprobs": logprobs,
                "input_logprobs": None
            })
        
        return results

class MolmoModelHandler(BaseModelHandler):
    """Model handler for the AllenAI Molmo VLM"""
    def load_model(self, model_path: str, processor_class, model_class, model_precision=torch.bfloat16):
        super().load_model(model_path, processor_class, model_class, model_precision=model_precision)
        assert self.processor and self.tokenizer and self.model
        self.gen_config.stop_sequences = ["<|endoftext|>"]
        self.gen_config.eos_token_id = self.tokenizer.eos_token_id
        self.gen_config.pad_token_id = self.tokenizer.pad_token_id

    def format_inputs(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Molmo takes plain text + image directly.
        Expect batch_item to contain:
        {
            "prompt": "Describe this image.",
            "image_path": "/path/to/image.jpg"
        }
        """
        return {
            **batch_item,
            "image": Image.open(batch_item["image_path"]).convert("RGB"),
            "text": batch_item["prompt"]
        }

    def prepare_inputs(self, batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        assert self.processor and self.tokenizer and self.model
                
        def format_batch(
            processor,
            texts: List[str],
            images_list: List[List[Image.Image]]
        ) -> Dict[str, torch.Tensor]:
            """
            Process in batch.
            
            Args:
                processor: The original processor.
                texts: List of text inputs
                images_list: List of lists containing PIL images.
                
            Returns:
                Dict with padded input_ids, images, image_input_idx, image_masks.
            """
            batch_size = len(texts)
            tokens_list = []
            for text in texts:
                tokens = processor.tokenizer.encode(" " + text, add_special_tokens=False)
                tokens_list.append(tokens)
            images_arrays_list = []
            image_idxs_list = []
            for images in images_list:
                if images:
                    image_arrays = []
                    for image in images:
                        if isinstance(image, Image.Image):
                            image = image.convert("RGB")
                            image = ImageOps.exif_transpose(image)
                            image_arrays.append(np.array(image))
                        else:
                            assert len(image.shape) == 3 and image.shape[-1] == 3
                            image_arrays.append(image.astype(np.uint8))
                    images_arrays_list.append(image_arrays)
                    image_idx = [-1] * len(image_arrays)
                    image_idxs_list.append(image_idx)
                else:
                    images_arrays_list.append(None)
                    image_idxs_list.append(None)
            images_kwargs = {
                "max_crops": 12,
                "overlap_margins": [4, 4],
                "base_image_input_size": [336, 336],
                "image_token_length_w": 12,
                "image_token_length_h": 12,
                "image_patch_size": 14,
                "image_padding_mask": True,
            }
            outputs_list = []
            for i in range(batch_size):
                tokens = tokens_list[i]
                images = images_arrays_list[i]
                image_idx = image_idxs_list[i]
                out = processor.image_processor.multimodal_preprocess(
                    images=images,
                    image_idx=image_idx,
                    tokens=np.asarray(tokens).astype(np.int32),
                    sequence_length=1536,
                    image_patch_token_id=processor.special_token_ids["<im_patch>"],
                    image_col_token_id=processor.special_token_ids["<im_col>"],
                    image_start_token_id=processor.special_token_ids["<im_start>"],
                    image_end_token_id=processor.special_token_ids["<im_end>"],
                    **images_kwargs,
                )
                outputs_list.append(out)

            batch_outputs = {}
            for key in outputs_list[0].keys():
                tensors = [torch.from_numpy(out[key]) for out in outputs_list]
                batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=-1
                )
            bos = processor.tokenizer.bos_token_id or processor.tokenizer.eos_token_id
            batch_outputs["input_ids"] = torch.nn.functional.pad(
                batch_outputs["input_ids"], (1, 0), value=bos
            )
            if "image_input_idx" in batch_outputs:
                image_input_idx = batch_outputs["image_input_idx"]
                batch_outputs["image_input_idx"] = torch.where(
                    image_input_idx < 0, image_input_idx, image_input_idx + 1
                )
            return batch_outputs

        # Process all images + texts
        images_batch = [[item["image"]] for item in batch]
        prompts_batch = [item["text"] for item in batch]
        texts = ["User: " + prompt + " Assistant:" for prompt in prompts_batch]
        inputs = format_batch(self.processor, texts, images_batch)

        # Move to device and add batch dimension
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs, texts

    def run_batch_inference(self, inputs) -> List[Dict[str,Any]]:
        assert self.processor and self.tokenizer and self.model

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            outputs = self.model.generate_from_batch(
                inputs,
                self.gen_config,
                tokenizer=self.processor.tokenizer,
            )

        return outputs

class Qwen2_5VLModelHandler(BaseModelHandler):
    """Handler for Qwen2.5-VL Models"""

    def format_inputs(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": batch_item["image_path"]},
                    {"type": "text", "text": batch_item["prompt"]},
                    
                ],
            }
        ]
        
        return {
            **batch_item,
            "messages": messages,
            "image": Image.open(batch_item["image_path"])
        }

    def prepare_inputs(self, batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        assert self.processor and self.tokenizer and self.model

        from qwen_vl_utils import process_vision_info

        prompts_batch = [self.processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True) for item in batch]        
        images_batch = [process_vision_info(item["messages"])[0][0] for item in batch]

        inputs = self.processor(
            images=images_batch,
            text=prompts_batch,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        return inputs, prompts_batch

class InferenceManager:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.return_input_logits = model_config.get("return_input_logits", False)
        handler_class = globals()[model_config["handler_class"]]
        self.handler = handler_class(self.model_config)
        
    def load_model(self, model_path: str):
        """Load a specific model"""
        processor_class = self.model_config.get("processor_class", None)
        model_class = self.model_config.get("model_class", None)

        self.handler.load_model(model_path, processor_class, model_class)
        self.handler.print_device_map()
        
    def run_inference(self, batch: List[Dict[str, Any]]) -> List[Dict[str,Any]]:
        """Run inference on a batch for a specific model"""
        formatted_batch: List[Dict[str, Any]] = self.handler.format_batch(batch)
        inputs, prompts = self.handler.prepare_inputs(formatted_batch)
        outputs = self.handler.run_batch_inference(inputs)
        return self.handler.process_outputs(outputs, inputs, prompts, formatted_batch, return_input_logits=self.return_input_logits)