import argparse
import os
import warnings

from boilr.eval import BaseOfflineEvaluator
from experiments import MnistExperiment


class Evaluator(BaseOfflineEvaluator):

    # TODO this could mostly be made into a generic VAE evaluator

    def run(self):

        experiment = self._experiment
        experiment.model.eval()

        # Run evaluation and print results
        results = experiment.test_procedure(iw_samples=self.args.ll_samples)
        print("Eval results:\n{}".format(results))

        # Save samples
        fname = os.path.join(self._img_folder, "samples.png")
        experiment.generate_and_save_samples(fname, nrows=8)

        # Save input and reconstructions
        x, y = next(iter(experiment.dataloaders.test))
        fname = os.path.join(self._img_folder, "reconstructions.png")
        experiment.generate_and_save_reconstructions(x, fname, nrows=8)

    # @classmethod
    # def _define_args_defaults(cls) -> dict:
    #     defaults = super(Evaluator, cls)._define_args_defaults()
    #     return defaults

    def _add_args(self, parser: argparse.ArgumentParser) -> None:

        super(Evaluator, self)._add_args(parser)

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

    @classmethod
    def _check_args(cls, args: argparse.Namespace) -> argparse.Namespace:
        args = super(Evaluator, cls)._check_args(args)

        if not args.ll:
            args.ll_samples = 1
        if args.load_step is not None:
            warnings.warn(
                "Loading weights from specific training step is not supported "
                "for now. The model will be loaded from the last checkpoint.")
        return args

def main():
    evaluator = Evaluator(experiment_class=MnistExperiment)
    evaluator()


if __name__ == "__main__":
    main()
