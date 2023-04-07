import logging
import os
from typing import List, Optional, Union
from transformers.data import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    PreTrainedTokenizer,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors
)
logger = logging.getLogger(__name__)

def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    text=None
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    if text == 'text_a':
        batch_encoding = tokenizer.batch_encode_plus(
            [example.text_a for example in examples],
            max_length=max_length, pad_to_max_length=True, return_token_type_ids=True
        )
    elif text == 'text_b':
        batch_encoding = tokenizer.batch_encode_plus(
            [example.text_b for example in examples],
            max_length=max_length, pad_to_max_length=True, return_token_type_ids=True
        )
    else:
        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples],
            max_length=max_length, pad_to_max_length=True, return_token_type_ids=True
        )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # for i, example in enumerate(examples[:5]):
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("features: %s" % features[i])

    return features

class SentenceDataProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return NotImplementedError

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# monkey-patch all glue classes to have test examples
def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

for task in glue_processors:
    processor = glue_processors[task]
    processor.get_test_examples = get_test_examples

# Other datasets
class AGNewsProcessor(SentenceDataProcessor):
    def get_labels(self):
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        return labels

class IMDBProcessor(SentenceDataProcessor):
    def get_labels(self):
        labels = ["pos", "neg"]
        return labels

class PubMedProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""""
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_dev_examples(self, data_dir):
        """See base class."""""
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_labels(self):
        labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
        return labels



class CFEVERProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["evidence"].numpy().decode("utf-8"),
            tensor_dict["claim"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[2]
            text_b = line[1]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

processors = glue_processors.copy()
processors.update(
    {"pubmed":PubMedProcessor, "agnews":AGNewsProcessor, "imdb":IMDBProcessor, "cfever":CFEVERProcessor, "scifact":CFEVERProcessor, "scifact_oracle":CFEVERProcessor}
)
output_modes = glue_output_modes
output_modes.update(
    {"pubmed":"classification", "agnews":"classification", "imdb":"classification", "cfever":"classification", "scifact":"classification", "scifact_oracle":"classification"}
)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["pubmed", "agnews", "imdb","sst-2"]:
        return {"f1":f1_score(y_true=labels, y_pred=preds, average="micro")}
    elif task_name in ["cfever"]:
        return {"f1":f1_score(y_true=labels, y_pred=preds, average="macro"), "acc":accuracy_score(y_true=labels, y_pred=preds)}
    elif task_name in glue_processors:
        return glue_compute_metrics(task_name, preds, labels)
    else:
        raise NotImplementedError

