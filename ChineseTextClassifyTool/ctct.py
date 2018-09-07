from model_site import *

class ctct(object):
    """ctct is short for chinese text classification tool
       Tt is the framework for chinese text classification 
    
    """
    def __init__(self, model,config,data,*args, **kwargs):
        if model.isnull():
            print("Please input a value for model!")
            raise ValueError
        self.model = model

        if config.isnull():
            if model == 'fasttext':
                pass 
            if model == 'TextCNN':
                config = TCNNConfig()
            if model == 'TextRNN':
                config = TRNNConfig()
        self.config = config


      

    def cutText(self):
        pass

    def dropWord(self):
        pass

    def modeTrain(self):
        pass

    def predict(self):
        pass
