from keras import Sequential
from keras.engine.saving import model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, Dropout, LSTM, RepeatVector, GlobalMaxPooling1D, \
    Concatenate, UpSampling2D, UpSampling1D, concatenate
from keras.models import Model
from lionets.load_dataset import Load_Dataset
from sklearn.metrics import accuracy_score,classification_report
#X,y,class_names = Load_Dataset.load_hate_speech()
X,y,class_names = Load_Dataset.load_smsspam()

X_train, X_test ,y_train ,y_test = train_test_split(X,y, random_state=70, stratify=y, test_size=0.33)
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

vec = TfidfVectorizer(max_features=5000)
vec.fit(X_train_copy)
X_train_copy = vec.transform(X_train_copy)
X_test_copy = vec.transform(X_test_copy)

input_dim = len(vec.get_feature_names())
autoencoder_input = Input(shape=(input_dim,))
autoencoder_x = Dense(800, activation='relu')(autoencoder_input)
autoencoder_x = Dropout(0.2)(autoencoder_x)
autoencoder_x = Dense(600, activation='relu')(autoencoder_x)
autoencoder_x = Dense(400, activation='relu')(autoencoder_x)
autoencoder_x = Dense(800, activation='relu')(autoencoder_x)
autoencoder_output = Dense(input_dim, activation='sigmoid')(autoencoder_x)
autoencoder = Model(autoencoder_input, autoencoder_output)

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
print(autoencoder.summary())
autoencoder.fit(X_train_copy, X_train_copy, epochs=100, batch_size=64, shuffle=True,
                validation_data=(X_test_copy, X_test_copy),verbose=2)

input_text = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[1](input_text)
encoder_layer = autoencoder.layers[2](encoder_layer)
encoder_layer = autoencoder.layers[3](encoder_layer)
encoder_layer = autoencoder.layers[4](encoder_layer)
encoder = Model(input_text, encoder_layer)

encoded_train = encoder.predict(X_train_copy)
decoded_train = autoencoder.predict(X_train_copy)

encoded_test = encoder.predict(X_test_copy)
decoded_test = autoencoder.predict(X_test_copy)

new_x_train = encoded_train
new_x_test = encoded_test

input_text = Input(shape=(400,))
predictions = Dense(1, activation='sigmoid')(input_text)

predictor = Model(input_text,predictions)
predictor.compile(optimizer=Adam(),loss=["binary_crossentropy"],metrics=['accuracy'])
print(predictor.summary())
predictor.fit([new_x_train], [y_train], validation_data=(new_x_test,y_test), epochs=20, verbose=2)  # starts training
y_preds = predictor.predict(new_x_test)
y_pred = []
for i in y_preds:
    if i>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

input_text = Input(shape=(400,))
decoder_layer = autoencoder.layers[5](input_text)
decoder_layer = autoencoder.layers[6](decoder_layer)
decoder = Model(input_text, decoder_layer)
decoder.summary()
decoded_train = decoder.predict(encoded_train)
decoded_test = decoder.predict(encoded_test)

from lionets.LioNets import LioNet
lionet = LioNet(predictor,autoencoder,decoder,encoder, vec.get_feature_names())
print(X_test[4])
lionet.explain_instance(X_test_copy[4])
lionet.print_neighbourhood_labels_distribution()
#lionet.accuracy