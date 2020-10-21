# %%
## Same code tou can find in the onnx tutorial: http://onnx.ai/sklearn-onnx/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# %%
### Training of a model
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)
print(len(X_test))

# %%
# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(clr, initial_types=initial_type)
with open("random_forest_iris.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Conversion has been successful!")

#%% 
# Load the model
import onnxruntime as rt
sess = rt.InferenceSession("random_forest_iris.onnx")
print("input name='{}' and shape={}".format(
    sess.get_inputs()[0].name, sess.get_inputs()[0].shape))
print("output name='{}' and shape={}".format(
    sess.get_outputs()[0].name, sess.get_outputs()[0].shape))

# %% Compute the predictions
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
print(confusion_matrix(y_test, pred_onx))

# %%
### Benchmark ####
## Sklearn
prob_sklearn = clr.predict(X_test)
## Onnx
prob_name = sess.get_outputs()[1].name
prob_rt = sess.run([prob_name], {input_name: X_test.astype(np.float32)})[0]

## timeit
from timeit import Timer

def speed(inst, number=10, repeat=20):
    timer = Timer(inst, globals=globals())
    raw = np.array(timer.repeat(repeat, number=number))
    ave = raw.sum() / len(raw) / number
    mi, ma = raw.min() / number, raw.max() / number
    print("Average %1.3g min=%1.3g max=%1.3g" % (ave, mi, ma))
    return ave

print("Execution time for clr.predict")
ave_sci = speed("clr.predict(X_test)")

print("Execution time for ONNX Runtime")
ave_onnx = speed("sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]")

# %%

# Plotting Performances
names = ['std_inference', 'onnxruntime_inference']
values = [ave_sci * 10e2, ave_onnx * 10e2]
fig  = plt.figure(figsize=(9,10))
plt.yticks(np.arange(0, 1, 0.05))
plt.xlabel('Inference Engines', fontsize='large',  fontweight='bold')
plt.ylabel('Time [ms]', fontsize='large',  fontweight='bold')
plt.title('Random Forest average inference performance (Iris Test set)', fontsize='large', fontweight='bold')
plt.bar(names, values)
plt.show()

# %%
