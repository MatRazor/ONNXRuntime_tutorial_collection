# %%
## Most part of the code taken from https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb

import os
import requests
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from transformers.data.processors.squad import SquadV1Processor
from transformers import squad_convert_examples_to_features
import torch
import torch.nn as nn
import onnxruntime
import matplotlib.pyplot as plt
from timeit import Timer
import numpy as np

def load_bert():
    # The following code is adapted from HuggingFace transformers
    # https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
    config = config_class.from_pretrained(model_name_or_path, cache_dir = cache_dir)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case = True, cache_dir = cache_dir)
    model = model_class.from_pretrained(model_name_or_path,
                                        from_tf = False,
                                        config = config,
                                        cache_dir = cache_dir)
    # load some examples
    processor = SquadV1Processor()
    examples = processor.get_dev_examples(None, filename=predict_file)

    # Convert examples to features
    features, dataset = squad_convert_examples_to_features( 
                examples=examples[:total_samples], # convert just enough examples for this notebook
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=False,
                return_dataset='pt'
            )
    return model, features, dataset


# %%
def speed(inst, number=10, repeat=20):
    timer = Timer(inst, globals=globals())
    raw = np.array(timer.repeat(repeat, number=number))
    ave = raw.sum() / len(raw) / number
    mi, ma = raw.min() / number, raw.max() / number
    print("Average %1.3g min=%1.3g max=%1.3g" % (ave, mi, ma))
    return ave

# %%
if __name__ == '__main__':

    # Create a cache directory to store pretrained model.
    cache_dir = os.path.join(".", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Download Stanford Question Answering Dataset (SQuAD) dataset (BERT trained on it)
    predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    file_name = "dev-v1.1.json"
    predict_file = os.path.join(cache_dir, file_name)
    if not os.path.exists(predict_file):
        print("Start downloading predict file.")
        r = requests.get(predict_file_url)
        with open(predict_file, 'wb') as f:
            f.write(r.content)
        print("Predict file downloaded.")

# %%
    # Bert Base Code for the Demo
    model_name_or_path = "bert-base-cased"
    max_seq_length = 128
    doc_stride = 128
    max_query_length = 64

    # Total samples to inference. It shall be large enough to get stable latency measurement.
    total_samples = 100

    # Load BERT PyTorch
    model, features, dataset = load_bert()
    output_dir = os.path.join(".", "onnx_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    export_model_path = os.path.join(output_dir, 'bert-base-cased-squad.onnx')

# %%
    data = dataset[0]

    inputs = {
        'input_ids' : data[0].reshape(1, max_seq_length),
        'attention_mask' : data[1].reshape(1, max_seq_length),
        'token_type_ids' : data[2].reshape(1, max_seq_length)
    }

    model.eval()

    # dynamic elements
    symbolic_names = {0 : 'batch_size', 1 : 'max_seq_length'}
    torch.onnx.export(model, 
                    args = tuple(inputs.values()),
                    f = export_model_path,
                    opset_version = 11,
                    do_constant_folding = True,
                    input_names = ['input_ids',
                                    'input_mask',
                                    'segment_ids'],
                    output_names = ['start',
                                    'end'],
                    dynamic_axes = {'input_ids' : symbolic_names,
                                    'input_mask' : symbolic_names,
                                    'segment_ids' : symbolic_names,
                                    'start' : symbolic_names,
                                    'end' : symbolic_names}                   
                                    )
    print('Model exported successfully in:', export_model_path)

#%%  
    print("Starting Pytorch...")
    #  torch model
    torch_avg_time = []
    with torch.no_grad():
        for i in range(total_samples):
            data = dataset[i]
            inputs = {
            'input_ids' : data[0].reshape(1, max_seq_length),
            'attention_mask' : data[1].reshape(1, max_seq_length),
            'token_type_ids' : data[2].reshape(1, max_seq_length)
            }
            ave_torch = speed("model(**inputs)")
            torch_avg_time.append(ave_torch)

    # ONNXRuntime
    print("Starting ONNX...")
    # Create a session
    session = onnxruntime.InferenceSession(export_model_path)
    
# %%
    # Inference through Onnxruntime
    onnxruntime_avg_time = []
    for i in range(total_samples):
        data = dataset[i]
        ort_inputs = {
            'input_ids' : data[0].reshape(1, max_seq_length).numpy(),
            'input_mask' : data[1].reshape(1, max_seq_length).numpy(),
            'segment_ids' : data[2].reshape(1, max_seq_length).numpy()
            }
        ave_onnx = speed("session.run(None, ort_inputs)")
        onnxruntime_avg_time.append(ave_onnx)
    
# %%
    torch_avg_final = sum(torch_avg_time) / len(torch_avg_time)
    print("Execution time for PyTorch")
    print(torch_avg_final)

    onnx_avg_final = sum(onnxruntime_avg_time) / len(onnxruntime_avg_time)
    print("Execution time for ONNX Runtime")
    print(onnx_avg_final)

# %%
    # Plotting Performances
    names = ['std_inference', 'onnxruntime_inference']
    values = [torch_avg_final * 10e2, onnx_avg_final * 10e2]
    fig  = plt.figure(figsize=(9,10))
    plt.yticks(np.arange(0, 170, 5))
    plt.xlabel('Inference Engines', fontsize='large',  fontweight='bold')
    plt.ylabel('Time [ms]', fontsize='large',  fontweight='bold')
    plt.title('BERT average inference performance (SQuAD set)', fontsize='large', fontweight='bold')
    plt.bar(names, values)
    plt.show()
# %%
