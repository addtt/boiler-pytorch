import argparse
import os
import warnings

from boilr.eval import BaseOfflineEvaluator
from experiments import MnistExperiment


class Evaluator(BaseOfflineEvaluator):

    def run(self):

        experiment = self._experiment
        model = experiment.model
        model.eval()

        # Run evaluation and print results
        results = experiment.test_procedure(
            iw_samples=self._eval_args.ll_samples)
        print("Eval results:\n{}".format(results))

        # Save samples
        fname = os.path.join(self._img_folder, "samples.png")
        experiment.save_samples(fname, n=8)

        # Save input and reconstructions
        x, y = next(iter(experiment.dataloaders.test))
        fname = os.path.join(self._img_folder, "reconstructions.png")
        experiment.save_input_and_recons(x, fname, n=8)

    def _parse_args(self):

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_required_args(parser)

        parser.add_argument('--ll',
                            action='store_true',
                            help="estimate log likelihood with importance-"
                            "weighted bound")
        parser.add_argument('--ll-samples',
                            type=int,
                            default=100,
                            dest='ll_samples',
                            metavar='N',
                            help="number of importance-weighted samples for "
                            "log likelihood estimation")
        parser.add_argument('--ps',
                            type=int,
                            default=1,
                            dest='prior_samples',
                            metavar='N',
                            help="number of batches of samples from prior")

        args = parser.parse_args()
        if args.test_batch_size == -1:
            args.test_batch_size = None
        if not args.ll:
            args.ll_samples = 1
        if args.load_step is not None:
            warnings.warn(
                "Loading weights from specific training step is not supported "
                "for now. The model will be loaded from the last checkpoint.")
        return args


def main():
    evaluator = Evaluator(MnistExperiment)
    evaluator()


if __name__ == "__main__":
    main()