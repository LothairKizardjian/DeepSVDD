# DeepSVDD
Implementation of the papaer http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf

# TODO :

- generate adversarial examples for each class individually (check the method used in the paper)
	--> if needed train a CNN for the mnist dataset then use it to generate adversarial examples
	(put everything in adversarial.py as a class) (CNN for mnist done, need to save weights once
	trained in dir 'models')
- create model for the problem (try resnet if pc can handle it)
- reproduce loss function of the paper : minimizing hypersphere in the feature space with regularizer on the weight.
