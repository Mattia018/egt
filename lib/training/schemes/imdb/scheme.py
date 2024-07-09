import torch
import torch.nn.functional as F

from lib.training.training import cached_property
from ..egt_mol_training import EGT_MOL_Training

from lib.models.imdb.model import EGT_IMDB
from lib.data.imdb.data import IMDBStructuralSVDGraphDataset


class IMDB_Training(EGT_MOL_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name='imdb',
            dataset_path='cache_data/IMDB',
            evaluation_type='prediction',
            predict_on=['test'],
            state_file=None,
        )
        return config_dict

    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, IMDBStructuralSVDGraphDataset

    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, EGT_IMDB

    def calculate_bce_loss(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        return F.binary_cross_entropy_with_logits(outputs, targets)

    def calculate_loss(self, outputs, inputs):
        return self.calculate_bce_loss(outputs, inputs['target'])

    @cached_property
    def evaluator(self):
        from torch_geometric.datasets import IMDB
        evaluator = Evaluator(name="ogbg-molhiv")  # Adjust as needed for IMDB
        return evaluator

    def prediction_step(self, batch):
        return dict(
            predictions=torch.sigmoid(self.model(batch)),
            targets=batch['target'],
        )

    def evaluate_predictions(self, predictions):
        input_dict = {"y_true": predictions['targets'],
                      "y_pred": predictions['predictions']}
        results = self.evaluator.eval(input_dict)

        xent = self.calculate_bce_loss(torch.from_numpy(predictions['predictions']),
                                       torch.from_numpy(predictions['targets'])).item()
        results['xent'] = xent

        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results

    def evaluate_on(self, dataset_name, dataset, predictions):
        print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions)
        return results


SCHEME = IMDB_Training