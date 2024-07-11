import torch
import torch.nn.functional as F
from lib.training.training import cached_property
from ..egt_mol_training import EGT_MOL_Training

from lib.models.dblp.model import EGT_DBLP
from lib.data.dblp.data import DBLPStructuralSVDGraphDataset

class DBLP_Training(EGT_MOL_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name='dblp',
            dataset_path='cache_data/DBLP',
            evaluation_type='prediction',
            predict_on=['test'],
            state_file=None,
        )
        return config_dict

    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, DBLPStructuralSVDGraphDataset

    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, EGT_DBLP

    def calculate_ce_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def calculate_loss(self, outputs, inputs):
        return self.calculate_ce_loss(outputs, inputs['target'])

    @cached_property
    def evaluator(self):
        from torch_geometric.datasets import DBLP
        # You might need to create a custom evaluator for DBLP
        evaluator = CustomDBLPEvaluator()  # This needs to be implemented
        return evaluator

    def prediction_step(self, batch):
        return dict(
            predictions=self.model(batch),
            targets=batch['target'],
        )

    def evaluate_predictions(self, predictions):
        input_dict = {"y_true": predictions['targets'],
                      "y_pred": predictions['predictions'].argmax(dim=1)}
        results = self.evaluator.eval(input_dict)

        ce_loss = self.calculate_ce_loss(torch.from_numpy(predictions['predictions']),
                                         torch.from_numpy(predictions['targets'])).item()
        results['ce_loss'] = ce_loss

        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results

    def evaluate_on(self, dataset_name, dataset, predictions):
        print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions)
        return results

SCHEME = DBLP_Training