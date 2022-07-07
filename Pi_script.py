#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:57:17 2022

@author: harry
"""

# import the opencv library
import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
import time
def visual_test(model, model_name, device, frame):
        # read class_indict
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(frame)
    img.to(device)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    #img = torch.quantize_per_tensor(img, 0.1, 0, torch.quint8)    
    #torch.backends.quantized.engine = 'fbgemm'
    #quantization_config = torch.quantization.get_default_qconfig('fbgemm')
    #model.qconfig = quantization_config
    #torch.quantization.prepare_qat(model, inplace=True)
    #model = torch.quantization.convert(model, inplace=True)
    times = []
    model.eval()
    with torch.no_grad():
        since = time.time()
        predictions = model(img)
        res = time.time() - since
        #Yprint('{} Time:{}s'.format(i, res))
        times.append(res)
        answer = predictions.data.cpu().numpy().argmax()
        category_index = {0:"Anchovy", 1:"Cod", 2:"Grey Gurnard", 3:"Haddock", 4: "Herring", 5: "Mackerel", 6:"Norway Pout", 7:"Red Mullet", 8:"Saithe", 9:"Sardine"}       
        print(category_index[answer])

    return times    
#torch.backends.quantized.engine = "qnnpack"
model = torch.jit.load('./AlexNet.pt', map_location="cpu:0")
print(model)
model.eval()
prev_frame_time = 0
fps_arr = []
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps_arr.append(fps)
    
    image_np = np.array(frame)

    input_tensor = torch.tensor(np.expand_dims(image_np, 0), dtype=torch.float32)
    times = visual_test(model, model_name="norm", device="cpu", frame=frame)
    print(times)
    fps = str(fps)
    print("FPS = " + fps)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()