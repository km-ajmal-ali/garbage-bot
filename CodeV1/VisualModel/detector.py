import numpy as np
from hailo_platform import HEF, Device, VDevice, HailoStreamInterface, InferVStreams

class HailoDetector:
    def __init__(self, hef_path):
        self.device = VDevice()
        self.hef = HEF(hef_path)
        self.network_group = self.device.configure(self.hef)[0]
        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

    def detect(self, frame):
        # Pre-process frame to match model input (e.g., 640x640)
        # In a real setup, use Hailo's optimized buffer management
        with InferVStreams(self.network_group, self.input_vstream_info, self.output_vstream_info) as infer_pipeline:
            input_data = {self.input_vstream_info[0].name: np.expand_dims(frame, axis=0)}
            with self.network_group.activate_context():
                output_data = infer_pipeline.infer(input_data)
        
        # Return bounding boxes [x, y, w, h, confidence]
        return output_data