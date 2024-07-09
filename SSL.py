import tensorflow as tf
from ssl_models import MultimediaSSLModel
from data_loader import MultimediaDataset


dataset = MultimediaDataset('path_to_dataset')
train_data, val_data, test_data = dataset.split_train_val_test()


ssl_model = MultimediaSSLModel()
ssl_model.compile(optimizer='adam', loss='contrastive_loss')
ssl_model.fit(train_data, epochs=10, validation_data=val_data)


metrics = ssl_model.evaluate(test_data)
print("Evaluation Metrics:", metrics)