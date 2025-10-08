####################################################################################################
#                                      pipeline.py                                                 #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Defines AugmentationPipeline, a modular class to chain multiple augmentations           #
#          in sequence, similar to torchvision.transforms.Compose.                                 #
#                                                                                                  #
####################################################################################################


#**************************************************************************************************#
#                                    Class AugmentationPipeline                                    #
#**************************************************************************************************#
#                                                                                                  #
# Abstract base class for chaining multiple augmentations in sequence.                             #
#                                                                                                  #
#**************************************************************************************************#
class AugmentationPipeline:
    """
    Chains multiple augmentation steps in sequence.
    """

    def __init__(self, steps):
        """
        Initializes the pipeline with a list of augmentation steps.

        Args:
            steps (list): List of augmentation step instances.
        """
        self.steps = steps

    def __call__(self, data, water=None, **kwargs):
        """
        Applies the augmentation steps in sequence to the data.

        Args:
            data: Input MRS data (NiftiMRS object).
            water: Optional water reference data (NiftiMRS object).
            **kwargs: Additional arguments passed to each step.
        """
        for step in self.steps:
            data, water = step(data, water, **kwargs)
        return data, water