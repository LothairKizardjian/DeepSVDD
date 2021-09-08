import ast

class ConfigClass(object):
    def __init__(self, configFile):

        ### paths
        paths = configFile['paths']
        self.src_dir = paths['src_dir']
        self.model_dir = paths['model_dir']

        ### model
        model = configFile['model']
        self.input_shape = ast.literal_eval(model['input_shape'])
        self.output_channels = int(model['output_channels'])
        self.filt_size = int(model['filt_size'])
        self.kernel_size = ast.literal_eval(model['kernel_size'])
        self.pool_size = ast.literal_eval(model['pool_size'])
        self.activation = model['activation']
        self.dropout = float(model['dropout'])
        self.epochs = int(model['epochs'])
        self.batch_size = int(model['batch_size'])
        self.lr = float(model['lr'])
        self.mnist_cnn_weights = model['mnist_cnn_weights']
        self.load_model = model.getboolean('load_model')

        ### DATA
        data = configFile['data']
        self.normal = int(data['normal'])
        self.outliers = int(data['outliers'])
        self.max_samples = int(data['max_samples'])

        ### SVDD
        svdd = configFile['SVDD']
        self.lambda_regularizer = float(svdd['lambda_regularizer'])
