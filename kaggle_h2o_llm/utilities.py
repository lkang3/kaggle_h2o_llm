import gc
import logging
import math
from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import Parameter
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BatchEncoding
from transformers import EvalPrediction
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.WARNING)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)


@dataclass
class DataConfig:
    question_field: str
    answer_field: str
    target_field: str
    context_field: str
    train_data_path: str
    test_data_path: str

    @staticmethod
    def from_config(data_config: Dict):
        return DataConfig(
            data_config["schema"]["question_field"],
            data_config["schema"]["answer_field"],
            data_config["schema"]["target_field"],
            data_config["schema"]["context_field"],
            data_config["train"]["path"],
            data_config["test"]["path"],
        )


@dataclass
class PartitionSchemaCV:
    folds: int

    @staticmethod
    def from_config(data_config: Dict):
        partition = data_config["partition"]
        return PartitionSchemaCV(int(partition["cv"]))


@dataclass
class PartitionSchemaTVH:
    train_pct: float
    train_label: int
    validation_pct: float
    validation_label: int
    holdout_pct: float
    holdout_label: int

    @staticmethod
    def from_config(data_config: Dict):
        partition = data_config["partition"]["tvh"]
        return PartitionSchemaTVH(
            float(partition["train"]["pct"]),
            int(partition["train"]["label"]),
            float(partition["validation"]["pct"]),
            int(partition["validation"]["label"]),
            float(partition.get("holdout", {}).get("pct", 0.0)),
            float(partition.get("holdout", {}).get("label", -1)),
        )


@dataclass
class TokenizationConfig:
    model_name: str
    max_length: int
    label_name: str
    context_mask_field: str
    answer_mask_field: str
    tokenization_output_fields: set[str]
    special_token_answer_start: str
    special_token_answer_end: str
    special_token_context_start: str
    special_token_context_end: str
    special_token_context_sep: str

    @staticmethod
    def from_config(data_config: Dict):
        return TokenizationConfig(
            model_name=data_config["model_name"],
            max_length=int(data_config["max_length"]),
            label_name=data_config["label_name"],
            tokenization_output_fields=set(
                data_config["tokenization_output_fields"].split(",")
            ),
            context_mask_field=data_config["context_mask_field"],
            answer_mask_field=data_config["answer_mask_field"],
            special_token_answer_start=data_config["special_token_answer_start"],
            special_token_answer_end=data_config["special_token_answer_end"],
            special_token_context_start=data_config["special_token_context_start"],
            special_token_context_end=data_config["special_token_context_end"],
            special_token_context_sep=data_config["special_token_context_sep"],
        )


@dataclass
class ModelConfig:
    model_name: str
    num_classes: int

    @staticmethod
    def from_config(data_config: Dict):
        return ModelConfig(
            model_name=data_config["model_name"],
            num_classes=data_config.get("num_classes", 0),
        )


@dataclass
class TrainConfig:
    random_seed: int
    model_output_path: str
    enable_grad_scale: bool
    partition_cv: PartitionSchemaCV
    partition_tvh: PartitionSchemaTVH
    num_epoch: int
    num_gradient_update_batch: int
    train_batch_size: int
    eval_batch_size: int
    holdout_batch_size: int
    learning_rate: float
    min_learning_rate: float

    @staticmethod
    def from_config(data_config: Dict):
        return TrainConfig(
            random_seed=int(data_config["random_seed"]),
            model_output_path=data_config["model_output_path"],
            enable_grad_scale=bool(data_config["enable_grad_scale"]),
            partition_cv=PartitionSchemaCV.from_config(data_config),
            partition_tvh=PartitionSchemaTVH.from_config(data_config),
            num_epoch=int(data_config["num_epoch"]),
            num_gradient_update_batch=int(data_config["num_gradient_update_batch"]),
            train_batch_size=int(data_config["train_batch_size"]),
            eval_batch_size=int(data_config["eval_batch_size"]),
            holdout_batch_size=int(data_config["holdout_batch_size"]),
            learning_rate=float(data_config["learning_rate"]),
            min_learning_rate=float(data_config["min_learning_rate"]),
        )


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    tokenizer_config: TokenizationConfig
    data_config: DataConfig

    def __call__(self, records: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(records)
        outputs = {}

        # add labels (if available)
        label_name = self.tokenizer_config.label_name
        if label_name in records[0]:
            outputs["labels"] = torch.tensor(
                [record.pop(label_name) for record in records], dtype=int
            )

        # convert tokenization outputs to tensor
        input_fields = self.tokenizer_config.tokenization_output_fields
        for field_name in input_fields:
            data = torch.zeros(
                (batch_size, self.tokenizer_config.max_length), dtype=int
            )
            for i in range(batch_size):
                data[i, :] = torch.tensor(records[i][field_name])
            outputs[field_name] = data

        return outputs


class MyAwesomePooler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.p = Parameter(torch.ones(1) * 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).expand(x.shape)
        x = (x.clip(min=self.eps) * mask).pow(self.p).sum(1)
        ret = x / mask.sum(1).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class MyAwesomeClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        class_count: int,
        tokenizer_length: int,
        pretrained_model_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, **pretrained_model_config)
        self.model.resize_token_embeddings(tokenizer_length)
        self.drop = torch.nn.Dropout(p=0.2)
        self.pool = MyAwesomePooler()
        self.header = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=class_count,
            bias=True,
        )

    def forward(self, input_ids, attention_mask, token_type_ids, answer_attention_mask):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        output = self.pool(output.last_hidden_state, answer_attention_mask)
        output = self.drop(output)
        return self.header(output)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def process_text_random_drop(
    input_text: str,
    drop_rate: float,
    random_state: np.random.RandomState,
) -> str:
    seperator = " "
    input_text = str(input_text).split(seperator)
    num_of_chars = len(input_text)
    if num_of_chars == 1:
        return input_text[0]
    char_indices = list(range(len(input_text)))
    random_state.shuffle(char_indices)
    char_indices_to_remove = char_indices[: max(int(drop_rate * num_of_chars), 1)]
    output_text = [
        char
        for char_idx, char in enumerate(input_text)
        if char_idx not in char_indices_to_remove
    ]
    return seperator.join(output_text)


def process_text_shuffle_sentences(
    input_text: str,
    random_state: np.random.RandomState,
) -> str:
    sentences = sent_tokenize(str(input_text))
    if len(sentences) < 2:
        return input_text
    sentence_indices = np.arange(len(sentences))
    random_state.shuffle(sentence_indices)
    return " ".join([str(sentences[idx]) for idx in sentence_indices])


def process_text_shuffle_answers(
    text_list: pd.Series,
    random_state: np.random.RandomState,
    seperator: Optional[str] = "[A_SEP]",
) -> List[str]:
    text_list = [str(text).strip() for text in text_list]
    random_state.shuffle(text_list)
    return seperator.join(text_list)


def process_text_add_context_with_multi_choices(
    input_data: pd.DataFrame,
    groupby_col_name: str,
    input_col_name: str,
    output_col_name: str,
    random_state: np.random.RandomState,
) -> pd.DataFrame:
    data = input_data.groupby(groupby_col_name)
    input_data[output_col_name] = data[input_col_name].transform(
        process_text_shuffle_answers,
        random_state=random_state,
    )
    return input_data


def process_text_with_random_drop(
    input_data: pd.DataFrame,
    data_col_names: List[str],
    random_state: np.random.RandomState,
) -> pd.DataFrame:
    input_data.loc[:, data_col_names] = input_data.loc[:, data_col_names].applymap(
        process_text_random_drop,
        drop_rate=0.1,
        random_state=random_state,
    )
    return input_data


def process_text_with_shuffle_sentences(
    input_data: pd.DataFrame,
    data_col_names: List[str],
    random_state: np.random.RandomState,
) -> pd.DataFrame:
    input_data.loc[:, data_col_names] = input_data.loc[:, data_col_names].applymap(
        process_text_shuffle_sentences,
        random_state=random_state,
    )
    return input_data


def process_text_with_augmentations(
    input_data: pd.DataFrame,
    text_processors: List[Callable],
    data_col_names: List[str],
    random_state: np.random.RandomState,
    ignore_index: Optional[bool] = True,
    shuffle_rows: Optional[bool] = False,
) -> pd.DataFrame:
    data = [
        text_processor(input_data.copy(), data_col_names, random_state)
        for text_processor in text_processors
    ]
    data = pd.concat(data, ignore_index=ignore_index)
    if shuffle_rows:
        return data.sample(
            len(data), random_state=random_state, ignore_index=ignore_index
        )
    return data


def prepare_test_data_with_tta(
    input_data: pd.DataFrame,
    text_processors: List[Callable],
    data_col_names: List[str],
    random_state: np.random.RandomState,
) -> List[pd.DataFrame]:
    return [
        text_processor(input_data.copy(), data_col_names, random_state)
        for text_processor in text_processors
    ]


def partition_data(
    data: Dataset,
    train_pct: float,
    validation_pct: float,
    holdout_pct: Optional[float] = 0.0,
    random_seed: Optional[int] = 123,
) -> Tuple[Dataset, Dataset, Dataset]:
    if train_pct + validation_pct + holdout_pct > 1.0:
        raise ValueError("Partition pct is invalid")

    rs = np.random.RandomState(random_seed)
    data_size = data.shape[0]
    partition_index = np.arange(data_size)
    rs.shuffle(partition_index)

    train_size = math.floor(train_pct * data_size)
    validation_size = math.floor(validation_pct * data_size)
    train_index = partition_index[:train_size]
    validation_index = partition_index[train_size : train_size + validation_size]
    if holdout_pct:
        holdout_index = partition_index[train_size + validation_size :]
    else:
        holdout_index = None

    train_ds = Dataset.from_dict(data[train_index])
    validation_ds = Dataset.from_dict(data[validation_index])
    holdout_ds = (
        Dataset.from_dict(data[holdout_index]) if holdout_index is not None else None
    )
    return train_ds, validation_ds, holdout_ds


def prompt_eng_answer_only(
    data,
    data_config: DataConfig,
) -> Union[str, List[str]]:
    answer = data[data_config.answer_field]
    if not isinstance(data, list):
        return f"{answer}"
    return answer


def prompt_eng_answer_and_context(
    data,
    data_config: DataConfig,
) -> Union[str, List[str]]:
    answer = data[data_config.answer_field]
    context = data[data_config.context_field]
    if not isinstance(data, list):
        return f"[A_START]{answer}[A_END][SEP]{context}"
    return [
        f"[A_START]{_answer}[A_END][SEP]{_context}"
        for _answer, _context in zip(answer, context)
    ]


def get_multichoice_context(
    df: pd.DataFrame,
    question_colname: str,
    answer_col_name: str,
    output_col_name: str,
    special_token_start: Optional[str] = "",
    special_token_end: Optional[str] = "",
    special_token_sep: Optional[str] = "",
    random_state: Optional[np.random.RandomState] = None,
    random_drop_context: Optional[bool] = False,
    random_shuffle_context: Optional[bool] = False,
    reset_index: Optional[bool] = False,
) -> pd.DataFrame:
    random_state = random_state or np.random.RandomState(123)

    def _apply_shuffle_choices(data: pd.Series) -> pd.Series:
        """Randomly shuffle choices"""
        return data.sample(frac=1, random_state=random_state)

    def _apply_drop_choices(data: pd.Series, drop_num: Optional[int] = 1) -> pd.Series:
        """Randomly drop drop_num choice(s)"""
        data_values = data.values
        len_values = len(data_values)
        if len_values > drop_num:
            data_indices = random_state.choice(
                np.arange(len_values), len_values - drop_num
            )
            return pd.Series(data_values[data_indices])
        return data

    def _apply_on_choices(data: pd.DataFrame) -> str:
        if random_shuffle_context:
            data = _apply_shuffle_choices(data)
        if random_drop_context:
            data = _apply_drop_choices(data)

        return (
            special_token_start + (data + special_token_sep).sum() + special_token_end
        )

    groupby_df = df.groupby(question_colname)
    output = groupby_df.apply(
        lambda x: _apply_on_choices(x[answer_col_name].astype(str))
    )
    # rename for merging with DataFrame down the road
    output.rename(output_col_name, inplace=True)
    if reset_index:
        output.reset_index(inplace=True)
    return output


def tokenization_preprocess(
    record,
    tokenizer: AutoTokenizer,
    tokenizer_config: TokenizationConfig,
    data_config: DataConfig,
    prompt_engineer_func: Callable,
    is_for_inference: Optional[bool] = False,
) -> BatchEncoding:
    # extract question and prompt
    question = record[data_config.question_field]
    prompt = (
        prompt_engineer_func(record, data_config)
        if prompt_engineer_func is not None
        else None
    )

    # perform tokenization on question and prompt
    processed_record = tokenizer(
        question,
        prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer_config.max_length,
    )

    # add customized masking based on tokenization outputs
    input_ids = np.array(processed_record.data["input_ids"])
    output_mask = np.zeros(input_ids.shape, dtype=int)
    for row_idx in range(len(input_ids)):
        end_tokens = np.where(
            np.array(input_ids[row_idx, :])
            == tokenizer.convert_tokens_to_ids(
                tokenizer_config.special_token_answer_end
            )
        )[0]
        if len(end_tokens):
            output_mask[row_idx, : end_tokens[0] + 1] = 1
    processed_record[tokenizer_config.answer_mask_field] = output_mask.tolist()

    # add target if applied
    if not is_for_inference:
        target_value = (
            record[data_config.target_field] if is_for_inference is not None else None
        )
        processed_record[tokenizer_config.label_name] = target_value

    return processed_record


def one_hot_encode_label(labels: List[int], num_of_labels: int) -> List[int]:
    encoded_labels = np.zeros((len(labels), num_of_labels))
    for idx, label in enumerate(labels):
        encoded_labels[idx, label] = 1
    return encoded_labels


def compute_metric(eval_preds: EvalPrediction, num_of_labels: int = 10) -> Dict:
    predictions, labels = eval_preds
    predictions = softmax(predictions, axis=-1)
    return {
        "log_loss": log_loss(one_hot_encode_label(labels, num_of_labels), predictions)
    }


def compute_metrics_logloss(num_of_labels: int) -> Callable:
    def _compute_metric(eval_preds: EvalPrediction) -> Dict:
        predictions, labels = eval_preds
        predictions = softmax(predictions, axis=-1)
        return {
            "log_loss": log_loss(
                one_hot_encode_label(labels, num_of_labels), predictions
            )
        }

    return _compute_metric


def get_partition_indices_group_based(
    data_size: int,
    num_of_folds: int,
    group_col: np.ndarray,
) -> np.ndarray:
    partition_index = np.zeros(data_size)
    partition_index[:] = -1
    group_k_folds = GroupKFold(n_splits=num_of_folds)
    for fold_idx, (_, test_index) in enumerate(
        group_k_folds.split(np.ones(data_size), groups=group_col)
    ):
        partition_index[test_index] = fold_idx
    return partition_index


def get_partition_indices_tvh(
    data_size: int,
    train_pct: float,
    train_label: int,
    validation_pct: float,
    validation_label: int,
    holdout_pct: Optional[float] = 0.0,
    holdout_label: Optional[int] = -1,
    random_seed: Optional[int] = 123,
) -> np.ndarray:
    if train_pct <= 0.0:
        raise ValueError("Training % <= 0.0.")
    if validation_pct <= 0.0:
        raise ValueError("Validation % <= 0.0.")
    # create TVH partition indices
    indices = np.arange(data_size)
    num_train = int(data_size * train_pct)
    num_validation = int(data_size * validation_pct)
    num_holdout = int(data_size * holdout_pct)
    indices[:num_train] = train_label
    indices[num_train : num_train + num_validation] = validation_label
    if num_holdout:
        indices[-num_holdout:] = holdout_label
    # shuffle
    rs = np.random.RandomState(random_seed)
    rs.shuffle(indices)
    return indices


def get_model_inputs(
    device: torch.device,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # SMELL
    input_fields = set(
        ["input_ids", "token_type_ids", "attention_mask", "answer_attention_mask"]
    )
    return {k: v.to(device) for k, v in batch.items() if k in input_fields}


def get_model_inputs_without_accelerator(
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # SMELL
    input_fields = set(
        ["input_ids", "token_type_ids", "attention_mask", "answer_attention_mask"]
    )
    return {k: v for k, v in batch.items() if k in input_fields}


def get_model_targets(
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    target_name: Optional[str] = "labels",
) -> Optional[torch.Tensor]:
    if target_name in batch:
        return batch[target_name].to(device)
    return None


def get_model_targets_without_accelerator(
    batch: Dict[str, torch.Tensor],
    target_name: Optional[str] = "labels",
) -> Optional[torch.Tensor]:
    if target_name in batch:
        return batch[target_name]
    return None


def convert_model_output_to_ndarray(
    model_outputs: SequenceClassifierOutput,
) -> np.ndarray:
    model_outputs = (
        model_outputs
        if isinstance(model_outputs, torch.Tensor)
        else model_outputs.logits
    )
    return model_outputs.softmax(dim=1).detach().cpu().numpy()


def get_pred_and_loss(
    device: torch.device,
    model: Module,
    inputs: Dict[str, torch.Tensor],
    loss_func: Callable,
    target: Optional[torch.Tensor] = None,
    enable_autocast: Optional[bool] = False,
) -> Tuple[SequenceClassifierOutput, torch.Tensor]:
    with autocast(device_type=device.type, enabled=enable_autocast):
        outputs = model(**inputs)
        outputs = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        loss = loss_func(outputs, target) if target is not None else None
        return outputs, loss


def save_pytorch_model(model: Module, output_path: str):
    torch.save(model.state_dict(), output_path)


def my_awesome_trainer(
    device: torch.device,
    train_config: TrainConfig,
    model: Module,
    train_data_loader: DataLoader,
    eval_data_loader: DataLoader,
    optimizer: Optimizer,
    learning_rate_scheduler: LRScheduler,
    num_epoch: int,
    num_gradient_update_batch: int,
    loss_func: Callable,
    accelerator: Optional[Accelerator] = None,
):
    # accelerator prep
    to_accelerate = accelerator is not None
    if to_accelerate:
        train_data_loader, eval_data_loader, model, optimizer = accelerator.prepare(
            train_data_loader,
            eval_data_loader,
            model,
            optimizer,
        )

    best_score = np.inf
    train_loss_history = []
    eval_loss_history = []

    progress_epoch = tqdm(range(num_epoch), total=num_epoch)
    for epoch_index in progress_epoch:
        progress_epoch.set_description(f"Epoch: {epoch_index}")
        epoch_train_loss = train_one_epoch(
            device,
            train_config,
            model,
            train_data_loader,
            optimizer,
            learning_rate_scheduler,
            num_gradient_update_batch,
            loss_func,
            accelerator,
        )
        train_loss_history.append(epoch_train_loss)

        epoch_eval_loss = evaluate(device, model, eval_data_loader, loss_func)
        eval_loss_history.append(epoch_eval_loss)

        if np.isfinite(epoch_eval_loss) and epoch_eval_loss < best_score:
            best_score = epoch_eval_loss
            save_pytorch_model(model, output_path=train_config.model_output_path)

        print(f"Echo: {epoch_index} Loss {epoch_eval_loss}")


@torch.no_grad()
def evaluate(
    device: torch.device,
    model: Module,
    data_loader: DataLoader,
    loss_func: Callable,
) -> float:
    model.eval()
    eval_loss = []

    for batch in data_loader:
        inputs = get_model_inputs(device, batch)
        targets = get_model_targets(device, batch)
        _, loss = get_pred_and_loss(device, model, inputs, loss_func, targets)
        eval_loss.append(loss.item())

    return np.mean(eval_loss)


@torch.no_grad()
def predict(
    device: torch.device,
    model: Module,
    data_loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()
    preds = []

    with autocast(device_type=device.type):
        for batch in data_loader:
            inputs = get_model_inputs(device, batch)
            outputs = model(**inputs)
            outputs = convert_model_output_to_ndarray(outputs)
            preds.append(outputs)

    preds = np.concatenate(preds)
    labels = np.argmax(preds, axis=-1)
    return preds, labels


def train_one_epoch(
    device: torch.device,
    train_config: TrainConfig,
    model: Module,
    train_data_loader: DataLoader,
    optimizer: object,
    learning_rate_scheduler: LRScheduler,
    num_gradient_update_batch: int,
    loss_func: Callable,
    accelerator: Optional[Accelerator] = None,
) -> float:
    to_accelerate = accelerator is not None
    scaler = GradScaler(enabled=train_config.enable_grad_scale)
    batch_loss = []
    num_of_batch = len(train_data_loader)

    for batch_idx, batch in enumerate(train_data_loader):
        inputs = (
            get_model_inputs(device, batch)
            if not to_accelerate
            else get_model_inputs_without_accelerator(batch)
        )
        targets = (
            get_model_targets(device, batch)
            if not to_accelerate
            else get_model_targets_without_accelerator(batch)
        )
        _, loss = get_pred_and_loss(device, model, inputs, loss_func, targets, True)
        batch_loss.append(loss.item())
        # accumulate gradients (scaled)
        if to_accelerate:
            accelerator.backward(scaler.scale(loss))
        else:
            scaler.scale(loss).backward()
        # weight update
        if (
            batch_idx + 1
        ) % num_gradient_update_batch == 0 or batch_idx == num_of_batch - 1:
            # unscale gradients
            scaler.step(optimizer)
            # update weights
            scaler.update()
            # reset the gradients
            optimizer.zero_grad()
            # adjust learning rate
            learning_rate_scheduler.step()

    torch.cuda.empty_cache()
    gc.collect()

    return np.mean(batch_loss)
