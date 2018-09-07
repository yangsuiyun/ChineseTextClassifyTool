from ChineseTextClassifyTool import model,train
import ChineseTextClassifyTool.model_site as config


tc = config.TCNNConfig()
tc.batch_size = 20
model,acc = model.TextCNN(tc)
train.train