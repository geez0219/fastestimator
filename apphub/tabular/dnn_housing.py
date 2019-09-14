# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
import tempfile
import fastestimator as fe
from fastestimator.network.loss import MeanSquaredError
from fastestimator.network.model import FEModel, ModelOp
from fastestimator.estimator.trace import ModelSaver


def create_dnn():
    model = tf.keras.Sequential()
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="linear"))
    return model


def get_estimator(epochs=50, batch_size=32, model_dir=tempfile.mkdtemp()):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.boston_housing.load_data()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_eval = scaler.transform(x_eval)
    data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data)

    #prepare model
    model = FEModel(model_def=create_dnn, model_name="dnn", optimizer="adam")
    network = fe.Network(
        ops=[ModelOp(inputs="x", model=model, outputs="y_pred"), MeanSquaredError(y_true="y", y_pred="y_pred")])

    #create estimator
    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=epochs, log_steps=10, traces=ModelSaver(model_name="dnn", save_dir=model_dir, save_best=True))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
