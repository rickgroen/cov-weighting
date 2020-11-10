import time
import torch

# Project imports
from utils import *
from options import TrainOptions
from data_loaders import create_dataloader
from methods import create_method


class Trainer:

    """
        Trains any of the model methods in methods/ using any of the losses
        in losses/.
        Initialize the trainer using a set of parsed arguments from options/.
    """

    def __init__(self, args):
        self.args = args

        # Retrieve train and validation data loaders.
        self.loader = create_dataloader(self.args, 'train')
        self.val_loader = create_dataloader(self.args, 'val')
        num_iterations_per_epoch = len(self.loader)
        setattr(args, 'num_iterations', num_iterations_per_epoch)

        # Initialize a model.
        self.model = create_method(args, self.loader)

        # We keep track of the aggregated losses per epoch in a dict. For now the pre-training
        # train loss is set to zero.
        self.best_val_loss = float('Inf')
        self.validate(-1)
        pre_validation_update(self.model.losses[-1]['val'])

    def train(self):
        """ Main function for training any of the methods, given an input parse.
        """
        for epoch in range(self.args.epochs):
            self.model.update_learning_rate(epoch, self.args.learning_rate)

            c_time = time.time()
            self.model.to_train()

            # Run a single training epoch.
            self.model.run_epoch(epoch)

            # Perform a validation pass each epoch.
            self.validate(epoch)

            # Print an update of training, val losses.
            print_epoch_update(epoch, time.time() - c_time, self.model.losses)

            # Make a checkpoint, so training can be resumed.
            running_val_loss = self.model.losses[epoch]['val']
            is_best = running_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = running_val_loss
                self.model.save_network('best')
        print('Finished Training. Best validation loss:\t{:.4f}'.format(self.best_val_loss))

        # Save the model of the final epoch. If another model was better, also save it separately as best.
        self.model.save_network('final')
        self.model.save_losses()

    def validate(self, epoch):
        self.model.to_eval()

        val_loss = 0.0
        for data in self.val_loader:
            # Get the losses for the model for this epoch.
            self.model.set_input(data)
            self.model.forward()
            iter_loss = self.model.get_untrained_loss()
            val_loss += iter_loss
        # Compute the loss over this validation set.
        val_loss /= len(self.val_loader)
        # Store the running loss for the validation images.
        self.model.store_val_loss(val_loss, epoch)

    def verify_data(self):
        """ Verifies whether all data has been downloaded and correctly put in the data directory.
        """
        check_if_all_images_are_present('eigen', self.args.data_dir)
        check_if_all_images_are_present('cityscapes', self.args.data_dir)


def main():
    parser = TrainOptions()
    args = parser.parse()
    args.mode = 'train'

    # Print CUDA version.
    print("Running code using CUDA {}".format(torch.version.cuda))
    gpu_id = int(args.device[-1])
    torch.cuda.set_device(gpu_id)
    print('Training on device cuda:{}'.format(gpu_id))

    trainer = Trainer(args)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'verify-data':
        trainer.verify_data()


if __name__ == '__main__':
    main()
    print("YOU ARE TERMINATED!")
