import os,sys,re,time,shutil
from collections import defaultdict
import numpy as np
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2, prediction_service_pb2_grpc
from tensorflow.compat.v1 import make_tensor_proto

class TFServingClient():
    def __init__(self, grpc_point='127.0.0.1:8500', model_name='model', signature_name='serving_default'):
        channel = grpc.insecure_channel(grpc_point)
        self.prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.model_name = model_name
        self.signature_name = signature_name 
        self.current_request_uid = 0
        self.request_queue = defaultdict(dict)
        self.response_queue = defaultdict(dict)

    #批量预测
    def predict(self, inputs, version=None):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        if version: request.model_spec.version.value = version
        for name, data in inputs.items():
            request.inputs[name].CopyFrom(make_tensor_proto(data, shape=list(data.shape), dtype=data.dtype))
        response = self.prediction_service_stub.Predict(request, timeout=1000.0)
        results = {key:tf.make_ndarray(tensor_proto) for key,tensor_proto in response.outputs.items()}
        return results

    #主工作循环
    async def main_loop(self, batch_size=100):
        while True:
            await asyncio.sleep(1)
            uid_list=[]
            inputs_dict = defaultdict([])
            for i in range(batch_size):
                try:
                    uid, inputs = self.request_queue.popitem()
                    uid_list.append(uid)
                    for name, data in inputs:
                        inputs_dict[name].append(data[np.newaxis,:])
                except:
                    break
            for name, data_list in inputs_dict.items():
                inputs_dict[name] = np.concatenate(data_list)

            results = self.predict(inputs=inputs_dict)
            for name, outputs in results:
                for i in range(outputs.shape[0]):
                    uid = uid_list[i]
                    self.response_queue[uid][name]=outputs[i]
                    
    #异步预测
    async def apredict(self, inputs):
          uid = self.current_request_uid
          self.current_request_uid += 1
          self.request_queue[uid] = inputs
          for i in range(10):
              await asyncio.sleep(1)
              if uid in self.response_queue:
                  res = self.response_queue[uid]
                  return res

if __name__=='__main__':
    client = TFServingClient(model_name='only_attention') 
    import time
    t0=time.time()
    inputs={'wid':np.zeros([10,1000],dtype=np.int32), 'T':np.array([0.1],dtype=np.float32)}
    res = client.predict(inputs=inputs)
    t1=time.time()
    print('---------use time: ', t1-t0)
    tasks = [client.apredict(inputs=inputs)] 
    loop.run_until_complete(client.main_loop()) 
