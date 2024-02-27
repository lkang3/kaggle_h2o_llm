import argparse
import copy

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset
from sklearn.preprocessing import normalize
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer
from utilities import DataCollatorForMultipleChoice
from utilities import DataConfig
from utilities import MyAwesomeClassifier
from utilities import TokenizationConfig
from utilities import TrainConfig
from utilities import get_multichoice_context
from utilities import get_partition_indices_tvh
from utilities import my_awesome_trainer
from utilities import predict
from utilities import prepare_test_data_with_tta
from utilities import process_text_with_augmentations
from utilities import process_text_with_random_drop
from utilities import process_text_with_shuffle_sentences
from utilities import prompt_eng_answer_and_context
from utilities import tokenization_preprocess

random_state = np.random.RandomState(123)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    args = arg_parser.parse_args()
    config_path = args.config_path

    # load config file
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    tokenizer_config = TokenizationConfig.from_config(config["tokenizer"])
    data_config = DataConfig.from_config(config["data"])
    model_config = config["model"]
    train_config = TrainConfig.from_config(config["training"])
    random_seed = train_config.random_seed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.model_name)
    tokenizer.add_tokens(
        new_tokens=[
            tokenizer_config.special_token_answer_start,
            tokenizer_config.special_token_answer_end,
            tokenizer_config.special_token_context_start,
            tokenizer_config.special_token_context_end,
            tokenizer_config.special_token_context_sep,
        ],
        special_tokens=True,
    )

    preprocess_args_for_train = {
        "tokenizer": tokenizer,
        "tokenizer_config": tokenizer_config,
        "data_config": data_config,
        "prompt_engineer_func": prompt_eng_answer_and_context,
    }
    preprocess_args_for_inference = copy.deepcopy(preprocess_args_for_train)
    preprocess_args_for_inference["is_for_inference"] = True
    data_collator = DataCollatorForMultipleChoice(
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        data_config=data_config,
    )

    # data
    data_file_config = {
        "train": data_config.train_data_path,
        "test": data_config.test_data_path,
    }
    dataset_train = pd.read_csv(data_file_config["train"])
    multi_choices_context = get_multichoice_context(
        df=dataset_train,
        question_colname=data_config.question_field,
        answer_col_name=data_config.answer_field,
        output_col_name="Context",
        special_token_start=tokenizer_config.special_token_context_start,
        special_token_end=tokenizer_config.special_token_context_end,
        random_state=random_state,
        random_drop_context=True,
        random_shuffle_context=True,
    )
    dataset_train = dataset_train.merge(
        multi_choices_context, on=data_config.question_field
    )
    dataset_test = pd.read_csv(data_file_config["test"])
    multi_choices_context = get_multichoice_context(
        df=dataset_test,
        question_colname=data_config.question_field,
        answer_col_name=data_config.answer_field,
        output_col_name="Context",
        special_token_start=tokenizer_config.special_token_context_start,
        special_token_end=tokenizer_config.special_token_context_end,
        random_state=random_state,
        random_drop_context=False,
        random_shuffle_context=False,
    )
    dataset_test = dataset_test.merge(
        multi_choices_context, on=data_config.question_field
    )

    # prep training data
    partition_schema = train_config.partition_tvh
    partition_indices = get_partition_indices_tvh(
        data_size=dataset_train.shape[0],
        train_pct=partition_schema.train_pct,
        train_label=partition_schema.train_label,
        validation_pct=partition_schema.validation_pct,
        validation_label=partition_schema.validation_label,
        random_seed=random_seed,
    )
    dataset_train["partition"] = partition_indices
    ds_train = dataset_train[dataset_train["partition"] == partition_schema.train_label]
    ds_train_augmented = process_text_with_augmentations(
        input_data=ds_train,
        text_processors=[
            process_text_with_random_drop,
            process_text_with_shuffle_sentences,
        ],
        data_col_names=[data_config.question_field, data_config.answer_field],
        random_state=random_state,
        ignore_index=True,
        shuffle_rows=True,
    )
    ds_train = pd.concat([ds_train, ds_train_augmented], ignore_index=True)
    ds_train = Dataset.from_pandas(ds_train)
    ds_train.cleanup_cache_files()
    ds_eval = dataset_train[
        dataset_train["partition"] == partition_schema.validation_label
    ]
    ds_eval = Dataset.from_pandas(ds_eval)
    ds_eval.cleanup_cache_files()
    # tokenization
    ds_train = ds_train.map(
        function=tokenization_preprocess,
        batched=True,
        fn_kwargs=preprocess_args_for_train,
        remove_columns=ds_train.column_names,
    )
    ds_eval = ds_eval.map(
        function=tokenization_preprocess,
        batched=True,
        fn_kwargs=preprocess_args_for_train,
        remove_columns=ds_eval.column_names,
    )
    train_data_loader = DataLoader(
        ds_train,
        batch_size=train_config.train_batch_size,
        collate_fn=data_collator,
        generator=torch_gen,
        shuffle=True,
    )
    eval_data_loader = DataLoader(
        ds_eval,
        batch_size=train_config.eval_batch_size,
        collate_fn=data_collator,
        generator=torch_gen,
        shuffle=True,
    )
    # pred with TTA
    tta_dataset = prepare_test_data_with_tta(
        input_data=dataset_test,
        text_processors=[
            process_text_with_random_drop,
            process_text_with_shuffle_sentences,
        ],
        data_col_names=[data_config.question_field, data_config.answer_field],
        random_state=random_state,
    )

    # train
    class_count = model_config["class_count"]
    model = MyAwesomeClassifier(
        model_name=model_config["model_name"],
        class_count=int(class_count),
        tokenizer_length=len(tokenizer),
        pretrained_model_config={"num_labels": class_count},
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate)
    learning_rate_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=600,
        eta_min=train_config.min_learning_rate,
    )
    accelerator = Accelerator()
    my_awesome_trainer(
        device=device,
        train_config=train_config,
        model=model,
        train_data_loader=train_data_loader,
        eval_data_loader=eval_data_loader,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        num_epoch=train_config.num_epoch,
        num_gradient_update_batch=train_config.num_gradient_update_batch,
        loss_func=CrossEntropyLoss(),
        accelerator=accelerator,
    )

    # TTA
    tta_dataset_indices = np.tile(np.arange(len(dataset_test)), len(tta_dataset))
    tta_preds = np.zeros((len(tta_dataset_indices), class_count))
    for idx, dataset_test in enumerate(tta_dataset):
        # prep testing data
        ds_test = Dataset.from_pandas(dataset_test)
        ds_test.cleanup_cache_files()
        ds_test = ds_test.map(
            function=tokenization_preprocess,
            batched=True,
            fn_kwargs=preprocess_args_for_inference,
            remove_columns=ds_test.column_names,
        )
        test_data_loader = DataLoader(
            ds_test,
            batch_size=train_config.holdout_batch_size,
            collate_fn=data_collator,
            generator=torch_gen,
        )
        preds, _ = predict(
            device=device,
            model=model,
            data_loader=test_data_loader,
        )
        tta_preds_start_row = idx * len(preds)
        tta_preds_end_row = (idx + 1) * len(preds)
        tta_preds[tta_preds_start_row:tta_preds_end_row, :] = preds

    tta_preds_col_names = np.arange(class_count)
    tta_preds_col_names.append("pred_id")
    tta_preds = pd.DataFrame(
        np.concatenate(
            [tta_preds, tta_dataset_indices.reshape(len(tta_dataset_indices), -1)],
            axis=-1,
        ),
        columns=tta_preds_col_names,
    )
    tta_preds.set_index("pred_id", inplace=True)
    tta_preds = tta_preds.groupby("pred_id").apply(
        lambda x: np.average(x.values, axis=0)
    )
    preds = np.stack(tta_preds.values)
    preds = normalize(preds, axis=1, norm="l1")
    num_preds = preds.shape[0]
    index = np.arange(num_preds)
    column_names = [f"target_{i}" for i in np.arange(class_count)]
    column_names.insert(0, "id")
    data = np.concatenate([index.reshape(num_preds, -1), preds], axis=-1)
    outputs = pd.DataFrame(data, columns=column_names)
    outputs.to_csv("/kaggle/working/submission.csv", index=False)
