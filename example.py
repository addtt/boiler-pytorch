from boiler import Trainer
from experiments import MnistExperiment


def main():
    experiment = MnistExperiment()
    trainer = Trainer(experiment)
    trainer.run()

if __name__ == "__main__":
    main()
