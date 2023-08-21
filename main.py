import os 
os.environ["WANDB_DISABLED"] = 'true'

import torch 

from transformers import Trainer
from models.sasrec import SASRecConfig, SASRecModel

class TrainDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 5
    
    def __getitem__(self, index):
        return {'x': [1] * 10, 'mask': [1]* 10, 'labels': [index] * 10}
    
class ValidDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 5
    
    def __getitem__(self, index):
        return {'x': [1] * 10, 'mask': [1]* 10, 'labels': [index] * 10}
    
class TrainerForSequentialRecommendation(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if return_outputs:
            outputs_dict = {
                'last_logits': outputs.logits[torch.arange(len(outputs.logits)), inputs['mask'].sum(dim=-1)-1]
            }
            return outputs.loss, outputs_dict
        else:
            return outputs.loss

if __name__ == '__main__':
    # dataset
    train_dataset = TrainDataset()
    valid_dataset = ValidDataset()
    test_dataset = ValidDataset()

    # model
    config = SASRecConfig(n_items=10, n_layers=2, n_heads=2, hidden_size=64, max_len=50, dropout=0.2)
    model = SASRecModel(config)

    # evaluate
    def compute_metrics(eval_predictions):
        # https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/src/transformers/trainer.py#L2647
        # our model output should be in dict format to avoid this bug
        # also check this https://discuss.huggingface.co/t/evalprediction-returning-one-less-prediction-than-label-id-for-each-batch/6958/6
        # solution: use `SASRecModelOutput(loss=None, logits=logits) ` to fit `logits = outputs[1:]`

        predictions, label_ids = eval_predictions
        pred_label = predictions[label_ids[:, -1:]]
        return {'f1': 0.75}
        
    # trainer
    trainer = TrainerForSequentialRecommendation(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    # train
    trainer.train()

    # hyperparameter search
    # check more details in https://huggingface.co/docs/transformers/hpo_train

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
        }

    def model_init(trial):
        config = SASRecConfig(n_items=10, n_layers=2, n_heads=2, hidden_size=64, max_len=50, dropout=0.2)
        return SASRecModel(config)
        
    trainer = TrainerForSequentialRecommendation(
        model=None,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=20,
    )