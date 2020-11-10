import torch
import cv2
from tqdm import tqdm

# Project imports
from utils import *
from options import TestOptions
from data_loaders import create_dataloader
from methods import BaseMethod


class Tester:

    """
        Tests the performance of models in saved_models//
        Initialize the trainer using a set of parsed arguments from options/.
    """

    def __init__(self, args):
        self.args = args
        self.args.mode = 'test'

        if not args.dataset == 'kitti':
            raise ValueError("We test on the KITTI test set, also for models trained on CityScapes.")

        # Retrieve a test data loader.
        self.loader = create_dataloader(self.args, 'test')

        # Initialize a model.
        self.model = BaseMethod(self.args, self.loader)
        which_model = 'final' if args.load_final else 'best'
        self.model.load_network(which_model)
        self.model.to_eval()

        self.name = args.model_name

        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.do_post_processing = args.postprocessing
        self.num_samples = len(self.loader)

    def test(self):
        """ Loops over the data in the test loader, computes the disparities and passes the disparity
            to a error function.
        """
        # Record the errors in arrays.
        rms = np.zeros(self.num_samples, np.float32)
        log_rms = np.zeros(self.num_samples, np.float32)
        abs_rel = np.zeros(self.num_samples, np.float32)
        sq_rel = np.zeros(self.num_samples, np.float32)
        a1 = np.zeros(self.num_samples, np.float32)
        a2 = np.zeros(self.num_samples, np.float32)
        a3 = np.zeros(self.num_samples, np.float32)

        with torch.no_grad():
            for i, data in tqdm(enumerate(self.loader)):
                # Retrieve the left and mirrored-left images, the ground truth depth and camera parameters.
                images = data['left_image'].squeeze()
                depth = data['gt_depth'].squeeze().numpy().astype(np.float32)
                f, b = [ele.numpy() for ele in data['camera']]

                # Do a forward pass
                disps = self.model.predict(images)
                disps = disps[0][:, 0, :, :]

                if self.do_post_processing:
                    predicted_disparity = post_process_disparity(disps.cpu().numpy())
                else:
                    disp = disps.unsqueeze(1)
                    predicted_disparity = disp[0].squeeze().cpu().numpy()

                # Resize the disparity to the size of the depth.
                predicted_disparity = cv2.resize(predicted_disparity, (depth.shape[1], depth.shape[0]),
                                                 interpolation=cv2.INTER_LINEAR)
                predicted_disparity = predicted_disparity * predicted_disparity.shape[1]

                # Now pass the predicted disparity and the ground truth depth to an evaluate function.
                # Additionally pass the focal length and baseline, to convert the disparity to depth.
                error_metrics = self.evaluate_kitti(predicted_disparity, depth, f, b)
                abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = error_metrics

        # Finally, print the errors.
        result_name_string = "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}"
        result_name_string = result_name_string.format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                                                       'a1', 'a2', 'a3')
        print(result_name_string)

        result_metric_string = "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}"
        result_metric_string = result_metric_string.format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),
                                                           a1.mean(), a2.mean(), a3.mean())
        print(result_metric_string)

    def evaluate_kitti(self, disparity, gt_depth, focal_length, baseline):
        """ Computes 7 measures of error given a predicted disparity and a ground truth depth.
            To convert the disparity to a depth, we also need a focal length and a baseline.
        """
        depth_pred = (baseline * focal_length) / disparity
        depth_pred[np.isinf(depth_pred)] = 0

        depth_pred[depth_pred < self.min_depth] = self.min_depth
        depth_pred[depth_pred > self.max_depth] = self.max_depth
        mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
        return self.compute_errors(gt_depth[mask], depth_pred[mask])

    @staticmethod
    def compute_errors(gt, pred):
        # Sometimes, pred = 0. Thus we need to increase it by a small bit to prevent zero division.
        pred[pred < 1e-6] = 1e-6

        # Compute the distances within 3 thresholds.
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        # Compute RMSE.
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        # Compute relative distances.
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def main():
    parser = TestOptions()
    args = parser.parse()
    args.mode = 'test'
    print("Running code using CUDA {}".format(torch.version.cuda))

    tester = Tester(args)
    tester.test()


if __name__ == '__main__':
    main()
    print("YOU ARE TERMINATED!")
