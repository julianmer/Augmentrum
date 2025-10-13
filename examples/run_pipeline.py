####################################################################################################
#                                         run_pipeline.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Testing the pipline functionality. The script loads data, initializes the pipeline      #
#          with desired modules, and runs the pipeline on the data.                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse

# own
from augmentrum.dataset.brainbeats import BrainBeatsData
from augmentrum.dataset.fmrsinpain import fMRSinPainData


#***************************************************#
#   parse command line arguments and build config   #
#***************************************************#
def parse_args():
    """
    Parse command line arguments and build a configuration dictionary.

    Returns:
        config (dict): Configuration dictionary with dataset, paths, and other parameters.
    """
    parser = argparse.ArgumentParser(description="Run fMRS pipeline on in-vivo data")

    parser.add_argument('--data_dir', type=str,
                        default='../data/fMRSinPain/SUBJECTS/RAW/',
                        help='Path to the dataset directory')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')

    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of data to use for validation')

    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Fraction of data to use for testing')

    parser.add_argument('--n_coils', nargs=2, type=int, default=(1, None),
                        help='Range of coils to use (start, end (exclusive))')

    parser.add_argument('--n_averages', nargs=2, type=int, default=(1, None),
                        help='Range of averages to use (start, end (exclusive))')

    parser.add_argument('--cache_det', action='store_true', help='Enable cache_det')
    parser.add_argument('--no_cache_det', dest='cache_det', action='store_false', help='Disable cache_det')
    parser.set_defaults(cache_det=False)

    args = parser.parse_args()

    # Build configuration dictionary
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'val_frac': args.val_frac,
        'test_frac': args.test_frac,
        'n_coils': tuple(args.n_coils),
        'n_averages': tuple(args.n_averages),
        'pipelines': None,  # optional, can be extended later
        'cache_det': args.cache_det,
        'sampling_mode': None,
        'to_tensor': False,
        'perturber_args': {
            'amp_mean': 0.0,
            'amp_var_low': 0.0, 'amp_var_high': 0.0,
            'phase_low': 0.0, 'phase_high': 0.0,
            'freq_low': 0.0, 'freq_high': 0.0,
            'misalign': False
        }
    }
    return config


#**********#
#   main   #
#**********#
def main():
    """
    Main function to run the pipeline on the selected dataset.
    """
    config = parse_args()

    # load dataset
    print(f"Loading dataset from {config['data_dir']}...")
    if 'fmrsinpain' in config['data_dir'].lower():
        dataset = fMRSinPainData(**config)
    elif 'brainbeats' in config['data_dir'].lower():
        dataset = BrainBeatsData(**config)

    # example
    train_loader = dataset.train_dataloader()
    x, x_ref = next(train_loader)
    import matplotlib.pyplot as plt
    for elem in x:
        elem.plot()
        plt.show()

    # next batch
    x, x_ref = next(train_loader)
    for elem in x:
        elem.plot()
        plt.show()

    # validation (deterministic sampling)
    val_loader = dataset.val_dataloader()
    x, x_ref = next(iter(val_loader))

    for elem in x:
        elem.plot()
        plt.show()

    print("Pipeline run completed.")


#***************************#
#   run from command line   #
#***************************#
if __name__ == "__main__":
    # This script can be run directly from the command line.
    #
    # Example usage(s):
    # python -m examples.run_pipeline --data_dir ../data/fMRSinPain/SUBJECTS/RAW/

    main()
