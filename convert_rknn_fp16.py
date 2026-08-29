from rknn.api import RKNN

OUT = 'yolov8n_fp16.rknn'
rk = RKNN()
rk.config(mean_values=[[0, 0, 0]],
          std_values=[[255, 255, 255]],
          target_platform='RK3566')
rk.load_onnx('yolov8n.onnx')
# Float16 model: no quantization, no calibration set needed. Runs accurately on the NPU.
rk.build(do_quantization=False)
rk.export_rknn(OUT)
print('EXPORTED', OUT)
