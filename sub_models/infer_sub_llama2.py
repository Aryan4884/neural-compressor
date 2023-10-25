import onnx
# from onnx.checker import check_model
from onnx.external_data_helper import load_external_data_for_model
# sub_model_name = "./llama2_with_out_with_input_less_middle_sub_model_32.onnx"
# onnx_model = onnx.load(sub_model_name, load_external_data=False)
# initalizer_data_dir = "/mnt/disk4/modelHub/llama-2-7-onnx/"
# load_external_data_for_model(onnx_model, initalizer_data_dir)



import onnxruntime as ort

# ort.InferenceSession(onnx_model.SerializeToString() , providers=ort.get_available_providers())
# import pdb; pdb.set_trace()
# from neural_compressor.model import Model
# inc_model = Model(onnx_model)
# inc_model.topological_sort()
# sub_model = inc_model._model

# # check_result = check_model(sub_model)
import gc

import numpy as np
def get_input_sample_npz():
    sample_input_path = "/home/st_liu/workspace/projects/transformers/demo/llama_sample.npz"
    input_sample = np.load(sample_input_path)
    result = {}
    for key in input_sample.keys():
        result[key] = input_sample[key]
    return result


input_sample = get_input_sample_npz()

def optimize_model(model):
    from neural_compressor.model import Model
    inc_model = Model(model)
    inc_model.topological_sort()
    return inc_model._model

from onnxruntime.quantization.onnx_model import ONNXModel



def record_time(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to run.")
        return result
    return wrapper

@record_time
def create_session(model):
    serialized_model_start = time.time()
    serialized_model = model.SerializeToString()
    serialized_model_finished = time.time()
    print(f"took {serialized_model_finished- serialized_model_start:.6f} seconds to serialize the model in RAM.")
    session = ort.InferenceSession(path_or_bytes=serialized_model,providers=ort.get_available_providers())
    return session

@record_time
def infer_model(model, input_sample, output_dict):
    session = create_session(model)
    output_names = [out.name for out in model.graph.output]
    model_required_inputs_name_lst = [n.name for n in model.graph.input]
    # the model inputs come from two ways:
    #   the output of previous sub_model
    #   the global input
    input_feed = {}
    for input_name in model_required_inputs_name_lst:
        if input_name in input_sample:
            input_feed[input_name] = input_sample[input_name]
        elif input_name in output_dict:
            input_feed[input_name] = output_dict[input_name]
        else:
            raise Exception(f"input_name {input_name} is not in input_sample and output_dict")
    # import pdb; pdb.set_trace()
    print(f"*** Set the model inputs as {input_feed.keys()}")
    output = session.run(output_names=output_names, input_feed=input_feed)
    del session
    gc.collect()
    return output, output_names
import time

output_dict = {}
model_idx_lst = range(32, -1, -1)
for model_idx in model_idx_lst:
    sub_model = f"llama2_with_out_with_input_less_middle_sub_model_{model_idx}.onnx"
    print(f"************ Start do inference with {model_idx} graph {sub_model}")
# for sub_model in ["with_out_with_input_less_middle_sub_model_12.onnx", "with_out_with_input_less_middle_sub_model_11.onnx"]:
    #model = onnx.load(sub_model)
    load_start = time.time()
    model = onnx.load(sub_model, load_external_data=False)
    initalizer_data_dir = "/mnt/disk4/modelHub/llama-2-7-onnx/"
    load_external_data_for_model(model, initalizer_data_dir)
    ort_model = ONNXModel(model)
    ort_model.topological_sort()
    load_finished = time.time()
    print(f"Function took {load_finished- load_start:.6f} seconds to load the model from disk.")

    # # mdoel = optimize_model(model)
    # # check_model(model)
    # session = ort.InferenceSession(path_or_bytes=model.SerializeToString(),providers=ort.get_available_providers())
    # output_names = [out.name for out in model.graph.output]
    # model_required_inputs_name_lst = [n.name for n in model.graph.input]
    # # the model inputs come from two ways:
    # #   the output of previous sub_model
    # #   the global input
    # input_feed = {}
    # for input_name in model_required_inputs_name_lst:
    #     if input_name in input_sample:
    #         input_feed[input_name] = input_sample[input_name]
    #     elif input_name in output_dict:
    #         input_feed[input_name] = output_dict[input_name]
    #     else:
    #         raise Exception(f"input_name {input_name} is not in input_sample and output_dict")
    # # import pdb; pdb.set_trace()
    # print(f"*** Set the model inputs as {input_feed.keys()}")
    # output = session.run(output_names=output_names, input_feed=input_feed)
    output, output_names = infer_model(model, input_sample, output_dict)
    output_dict = {name: value for name, value in zip(output_names, output)}
    input_sample.update(output_dict)
    # import pdb; pdb.set_trace()
    print(f"output name {[ (k, v.shape) for k, v in output_dict.items() ]}")

    input = output
import pdb; pdb.set_trace()
    