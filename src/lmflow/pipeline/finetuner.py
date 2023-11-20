#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset.
"""

import copy
import logging
import os
import sys
import time
import datasets
import math
import transformers
import torch
import evaluate
from collections import defaultdict
from itertools import chain
from transformers import (
    Trainer,
    default_data_collator,
    set_seed,
)
from copy import deepcopy
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
import wandb
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_tuner import BaseTuner
from lmflow.pipeline.utils.lion_lamb import Lion_lamb
from lmflow.pipeline.utils.lion import Lion
from lmflow.pipeline.utils.adamw import AdamW
from lmflow.pipeline.utils.adagrad import Adagrad
from lmflow.pipeline.utils.sgd import SGD

from lmflow.pipeline.utils.peft_trainer import PeftTrainer, PeftSavingCallback
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


logger = logging.getLogger(__name__)


class Finetuner(BaseTuner):
    """
    Initializes the `Finetuner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_args : FinetunerArguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.

    """
    def __init__(self, model_args, data_args, finetuner_args, *args, **kwargs):

        self.model_args = model_args
        self.data_args = data_args
        self.finetuner_args = finetuner_args

        # Sending telemetry. Tracking the example usage helps us better
        # allocate resources to maintain them. The information sent is the one
        # passed as arguments along with your Python/PyTorch versions.
        send_example_telemetry("run_clm", model_args, data_args)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = finetuner_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {finetuner_args.local_rank},"
            f" device: {finetuner_args.device},"
            f" n_gpu: {finetuner_args.n_gpu},"
            f"distributed training: {bool(finetuner_args.local_rank != -1)},"
            f" 16-bits training: {finetuner_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {finetuner_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(finetuner_args.output_dir) and finetuner_args.do_train and not finetuner_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(finetuner_args.output_dir)
            if last_checkpoint is None and len(os.listdir(finetuner_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({finetuner_args.output_dir}) already"
                    " exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and finetuner_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at"
                    f" {last_checkpoint}. To avoid this behavior, change"
                    " the `--output_dir` or add `--overwrite_output_dir` to"
                    " train from scratch."
                )
        self.last_checkpoint = last_checkpoint

        # Set seed before initializing model.
        set_seed(finetuner_args.seed)


    def group_text(self, tokenized_datasets, model_max_length):
        """
        Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
        a dictionary.
        """
        data_args = self.data_args
        finetuner_args = self.finetuner_args

        if data_args.block_size is None:
            block_size = model_max_length
            if block_size > 1024:
                logger.warning(
	    			"The chosen tokenizer supports a `model_max_length` that is"
	    			" longer than the default `block_size` value"
	    			" of 1024. If you would like to use a longer `block_size`"
	    			" up to `tokenizer.model_max_length` you can override this "
	    			" default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if data_args.block_size > model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger"
	    			f" than the maximum length for the model"
                    f"({model_max_length})."
                    f" Using block_size={model_max_length}."
                )
            block_size = min(data_args.block_size, model_max_length)

        # Main data processing function that will concatenate all texts from
        # our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model
            # supported it instead of this drop, you can customize this part to
            # your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts
        # together, so group_texts throws away a remainder for each of those
        # groups of 1,000 texts. You can adjust that batch_size here but a
        # higher value might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation
        # of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        with finetuner_args.main_process_first(desc="grouping texts together"):
            group_batch_size = data_args.group_texts_batch_size
            if data_args.disable_group_texts:
                group_batch_size = 1
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                )

        return lm_datasets


    def tune(self, model, dataset, transform_dataset_in_place=True):
        """
        Perform tuning for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform tuning.

        dataset:
            dataset to train model.

        """
        model_args = self.model_args
        data_args = self.data_args
        finetuner_args = self.finetuner_args
        if not transform_dataset_in_place:
            dataset = copy.deepcopy(dataset)

        # Tokenization and text grouping must be done in the main process
        with finetuner_args.main_process_first(desc="dataset map tokenization"):
            tokenized_dataset = model.tokenize(dataset)
            lm_dataset = self.group_text(
                tokenized_dataset,
                model_max_length=model.get_max_length(),
            )

        train_dataset = lm_dataset.get_backend_dataset()

        if finetuner_args.do_eval:
            eval_dataset_args = deepcopy(data_args)
            eval_dataset_args.dataset_path = finetuner_args.eval_dataset_path
            eval_dataset = Dataset(eval_dataset_args)
            with finetuner_args.main_process_first(desc="dataset map tokenization"):
                tokenized_dataset = model.tokenize(eval_dataset)
                lm_dataset = self.group_text(
                    tokenized_dataset,
                    model_max_length=model.get_max_length(),
                )
            eval_dataset = lm_dataset.get_backend_dataset()


            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy")

            def compute_metrics(eval_preds):
                # import pdb; pdb.set_trace()
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)

        

        if finetuner_args.do_train:
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        # Initialize our Trainer
        training_args = finetuner_args
        training_args.block_size = data_args.block_size

        if model_args.use_lora:
            FinetuningTrainer = PeftTrainer
            trainer_callbacks = [PeftSavingCallback]
        else:
            FinetuningTrainer = Trainer
            trainer_callbacks = []

        if os.environ.get('LOCAL_RANK', '0') == '0':
            wandb.init(project="optimal_pretain",name=training_args.run_name)


        def get_layer_from_name(name):
            """Extract layer name from parameter name."""
            components = name.split('.')
            # breakpoint()
            if len(components) >= 3:
                # return ".".join(components[0:3]) # gptj 6b per layer, eg layer0, layer1
                # return ".".join(components[2:5]) # lora gptj 6b per layer, eg layer0, layer1


                # return ".".join(components[1:3]) # gpt2 ft per layer, eg layer0, layer1
                return ".".join(components[3:5]) # gpt2 lora per layer, eg layer0, layer1
                # return ".".join(components[1:4]) # gpt2 per layer, eg layer0.ln_1,layer0.attn, , layer1.mlp
                # return ".".join(components[1:5]) # gpt2 per layer, eg layer0.ln_1.weight,layer0.attn, , layer1.mlp

                # return ".".join(components[0:3]) # gpt2-xl ft per layer, eg layer0, layer1
                # return ".".join(components[2:5]) # gpt2-xl lora per layer, eg layer0, layer1
                # return ".".join(components[2:6]) # gpt2-xl lora per middle layer, eg layer0.ln_1,layer0.attn, , layer1.mlp
                # return ".".join(components[2:7]) # gpt2-xl lora per middle layer, eg layer0.ln_1.weight,layer0.attn.c_attn, , layer1.mlp.weight

                # return ".".join(components[1:3]) # llama 2 per layer, eg layer0, layer1
                # return ".".join(components[1:4]) # llama 2 ft, eg layer0.self_attn, layer1.mlp
                # return ".".join(components[3:6]) # llama 2 continue lora, eg layer0.self_attn, layer1.mlp 
                # return ".".join(components[3:5]) # llama 2 continue lora, eg layer0, layer1

                
                # return ".".join(components[1:4]) # llama 2, eg layer0.self_attn.k,layer0.self_attn.v, layer1.mlp.up_proj
                
                # return ".".join(components[2:4]) # phi 1.5 lora  per layer layers.17
                # return ".".join(components[2:5]) # phi 1.5 lora  per layer layers.17.mlp
                # return ".".join(components[0:2]) # phi 1.5 ft  per layer layers.17.mlp


                # return ".".join(components[:])

            else:
                return components[1]

        def get_detail_layer_from_name(name):
            """Extract layer name from parameter name."""
            components = name.split('.')
            # breakpoint()
            if len(components) >= 3:
                return ".".join(components[:])

            else:
                return components[1]
            
        def get_layerwise_params(model):
            layerwise_params = defaultdict(int)
            
            for name, param in model.named_parameters():
                # Only count parameters that have gradients
                # if '.h.0.attn.c_attn.weight' in name:
                #     breakpoint()
                if param.requires_grad:
                    layer_name = get_layer_from_name(name)
                    layerwise_params[layer_name] += torch.numel(param)
            
            return layerwise_params

        # breakpoint()
        layerwise_counts = get_layerwise_params(model.get_backend_model())

        

        # wandb report loss/time, loss/tokens, tokens/time
        class LossTimeCallback(TrainerCallback):
            def __init__(self):
                super().__init__()
                self.timestamps = []
                self.losses = []
                self.portion_step = 0  # Add this line to store portion_step

            def on_train_begin(self, args, state, control, **kwargs):
                self.start_time = time.time()
            
            def on_step_end(self, args, state, control, **kwargs):
                world_size = os.environ.get('WORLD_SIZE', 1)
                total_batch_size = int(args.per_device_train_batch_size) * int(world_size) * int(args.gradient_accumulation_steps)
                if state.is_local_process_zero:
                    model = kwargs.get("model")  # Assuming model is passed in **kwargs
                    if model :
                        layerwise_params = defaultdict(list)

                        for name, param in model.named_parameters():
                            layer_name = get_detail_layer_from_name(name)
                            layerwise_params[layer_name].append(param)
                            
                        for layer, params in layerwise_params.items():
                            # Concatenate all tensors for the current layer
                            all_params = torch.cat([p.view(-1) for p in params])
                            norm = torch.norm(all_params).item()
                            # print(f"Layer {layer}: Weight Norm {norm}")
                    if len(state.log_history)>0:
                        elapsed_time = time.time() - self.start_time
                        elapsed_tokens = total_batch_size * int(args.block_size) * int(state.global_step)
                        continue_global_tokens = total_batch_size * int(args.block_size) * int(self.portion_step)

                        current_loss = state.log_history[-1].get("loss")
                        current_lr = state.log_history[-1].get("learning_rate")

                        wandb.log({"loss": current_loss, "time": elapsed_time})
                        wandb.log({"loss": current_loss, "tokens": elapsed_tokens})
                        wandb.log({"time": elapsed_time, "tokens": elapsed_tokens})
                        wandb.log({"loss": current_loss,"continue_global_step": self.portion_step}) 
                        wandb.log({"lr": current_lr,"continue_global_step": self.portion_step})  
                        wandb.log({"loss":current_loss, "continue_global_tokens":continue_global_tokens})

        loss_time_callback = LossTimeCallback()
        trainer_callbacks.append(loss_time_callback)

        class ConloraTrainer(FinetuningTrainer):
            def get_global_step(self):
                return self.state.global_step

            def create_scheduler(self, num_training_steps:int, optimizer=None):
                num_portions = training_args.num_portions  # 总的份数
                selected_portion = training_args.selected_portion  # 选择的份数，从1开始
                lr_min = 0.0  # 你的最小学习率
                lr_max = training_args.learning_rate  # 你的最大学习率
                warmup_proportion = training_args.warmup_ratio  # 预热阶段占每一份的比例

                steps_per_portion = num_training_steps 
                warmup_steps = math.ceil(steps_per_portion * warmup_proportion)
                def lr_lambda(current_step):
                    current_step = self.get_global_step()
                    portion_step = current_step % steps_per_portion  + steps_per_portion * (selected_portion - 1)
                    loss_time_callback.portion_step = portion_step

                    warmup_start_step = steps_per_portion * (selected_portion - 1) + warmup_steps
                    lr_max_portion = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * warmup_start_step / ((steps_per_portion - warmup_steps)*num_portions)))
                    if current_step  < warmup_steps:
                        # 预热阶段
                        lr = lr_max_portion * current_step / warmup_steps
                    else:
                        # 余弦退火阶段
                        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (portion_step - warmup_steps) / ((steps_per_portion - warmup_steps)*num_portions)))
                    if os.environ.get('LOCAL_RANK', '0') == '0':
                        print("\ncurrent step", portion_step)
                        print('\nsteps_per_portion',steps_per_portion)
                        print('\nwarmup_start_step',warmup_steps)
                    return lr / training_args.learning_rate

                self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

            def setup_training(self, num_training_steps:int):
                self.create_scheduler(num_training_steps)

        class OptimizerConloraTrainer(FinetuningTrainer):
            def get_global_step(self):
                return self.state.global_step

            def create_optimizer(self):
                if training_args.optimizer_name == "LionLamb":
                    self.optimizer = Lion_lamb(model.get_backend_model().parameters(), layer_shape=layerwise_counts, betas=[0.95,0.98], lr=training_args.learning_rate, weight_decay=training_args.weight_decay, min_x=training_args.min_x, max_x=training_args.max_x)
                elif training_args.optimizer_name == "Lion":
                    self.optimizer = Lion(model.get_backend_model().parameters(), betas=[0.95,0.98], lr=training_args.learning_rate, weight_decay=training_args.weight_decay,layer_shape=layerwise_counts)
                elif training_args.optimizer_name == "Adagrad":
                    self.optimizer = Adagrad(model.get_backend_model().parameters(), lr=training_args.learning_rate, layer_shape=layerwise_counts)
                elif training_args.optimizer_name == "SGD":
                    self.optimizer = SGD(model.get_backend_model().parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay, layer_shape=layerwise_counts)
                elif training_args.optimizer_name == "Adamw":
                    self.optimizer = AdamW(model.get_backend_model().parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay,layer_shape=layerwise_counts)
                else:
                    raise ValueError(f"Unknown optimizer name: {training_args.optimizer_name}")
            
            def create_scheduler(self, num_training_steps:int, optimizer=None):
                num_portions = training_args.num_portions  # 总的份数
                selected_portion = training_args.selected_portion  # 选择的份数，从1开始
                lr_min = 0.0  # 你的最小学习率
                lr_max = training_args.learning_rate  # 你的最大学习率
                warmup_proportion = training_args.warmup_ratio  # 预热阶段占每一份的比例

                steps_per_portion = num_training_steps 
                warmup_steps = math.ceil(steps_per_portion * warmup_proportion)
                def lr_lambda(current_step):
                    current_step = self.get_global_step()
                    portion_step = current_step % steps_per_portion  + steps_per_portion * (selected_portion - 1)
                    loss_time_callback.portion_step = portion_step

                    warmup_start_step = steps_per_portion * (selected_portion - 1) + warmup_steps
                    lr_max_portion = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * warmup_start_step / ((steps_per_portion - warmup_steps)*num_portions)))
                    if current_step  < warmup_steps:
                        # 预热阶段
                        lr = lr_max_portion * current_step / warmup_steps
                    else:
                        # 余弦退火阶段
                        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (portion_step - warmup_steps) / ((steps_per_portion - warmup_steps)*num_portions)))
                        # lr = lr_max
                    if os.environ.get('LOCAL_RANK', '0') == '0':
                        print("\ncurrent step", portion_step)
                        print('\nsteps_per_portion',steps_per_portion)
                        print('\nwarmup_start_step',warmup_steps)
                    return lr / training_args.learning_rate

                self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

            def setup_training(self, num_training_steps:int):
                self.create_optimizer()
                self.create_scheduler(num_training_steps)

        # FinetuningTrainer / OptimizerConloraTrainer
        trainer = OptimizerConloraTrainer(
                model=model.get_backend_model(),
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=model.get_tokenizer(),
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=default_data_collator,
                compute_metrics=compute_metrics if training_args.do_eval else None,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
                callbacks=trainer_callbacks
            )
        # if training_args.optimizer_name == "Adamw":
        #     trainer = ConloraTrainer(
        #         model=model.get_backend_model(),
        #         args=training_args,
        #         train_dataset=train_dataset if training_args.do_train else None,
        #         eval_dataset=eval_dataset if training_args.do_eval else None,
        #         tokenizer=model.get_tokenizer(),
        #         # Data collator will default to DataCollatorWithPadding, so we change it.
        #         data_collator=default_data_collator,
        #         compute_metrics=compute_metrics if training_args.do_eval else None,
        #         preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        #         callbacks=trainer_callbacks
        #     )
        # else:
        #     trainer = OptimizerConloraTrainer(
        #         model=model.get_backend_model(),
        #         args=training_args,
        #         train_dataset=train_dataset if training_args.do_train else None,
        #         eval_dataset=eval_dataset if training_args.do_eval else None,
        #         tokenizer=model.get_tokenizer(),
        #         # Data collator will default to DataCollatorWithPadding, so we change it.
        #         data_collator=default_data_collator,
        #         compute_metrics=compute_metrics if training_args.do_eval else None,
        #         preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        #         callbacks=trainer_callbacks
        #     )

        # Training
        if training_args.do_train:
            checkpoint = None
            last_checkpoint = self.last_checkpoint
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            print(trainer.optimizer.optimizer)
            if not model_args.use_lora:
                trainer.save_model()  # Saves the tokenizer too for easy upload
            else:
                if model_args.save_aggregated_lora:
                    model.merge_lora_weights()
                model.save(finetuner_args.output_dir,model_args.save_aggregated_lora)

            metrics = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
    
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

        return model
