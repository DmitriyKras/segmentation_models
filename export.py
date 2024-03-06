import argparse
import os
import torch
import onnxruntime
import cv2
import numpy as np
from onnxsim import simplify
import onnx
# import models
from models.effunet import EfficientUnet
from models.bisenet import BiSeNet
from models.liteseg_mobilenet import LiteSegMobileNetV2
from models.liteseg_shufflenet import LiteSegShuffleNet
from models.esnet import ESNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', 
                        help="""Segmentation model name. Available models: effunet, bisenet, 
                        liteseg_mobilenet, liteseg_shufflenet, esnet""")
    parser.add_argument('--img', type=int, default=512, 
                        help='export image size (pixels)')
    parser.add_argument('--weights', type=str, default='weights/effunet/best.pt',
                        help='path to weight file to validate')
    parser.add_argument('--onnx-file', type=str, default='onnx/effunet/model.onnx',
                        help='path of exported onnx file')
    parser.add_argument('--checking', type=bool, default=True,
                        help='whether to check exported model with test image and video')
    parser.add_argument('--opset', type=int, default=11,
                        help='desired onnx opset version')
    args = parser.parse_args()

    # create folders if not exist
    if not os.path.exists('onnx'):
        os.makedirs('onnx')
    if not os.path.exists(f'onnx/{args.model}'):
        os.makedirs(f'onnx/{args.model}')
    
    ### GET MODEL ###

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
    if torch.cuda.is_available():
        print('[INFO] Using CUDA device')
    else:
        print('[INFO] CUDA is not detected. Using CPU')
    print('[INFO] Loading model...')
    if args.model == 'effunet':
        model = EfficientUnet().to(device)
    elif args.model == 'bisenet':
        model = BiSeNet(1, 'resnet18').to(device)
    elif args.model == 'liteseg_mobilenet':
        model = LiteSegMobileNetV2(1, pretrained=False).to(device)
    elif args.model == 'liteseg_shufflenet':
        model = LiteSegShuffleNet(1, pretrained=False).to(device)
    elif args.model == 'esnet':
        model = ESNet(1).to(device)
    else:
        print(f'[INFO] Model {args.model} is not supported. Using default effunet')
        model = EfficientUnet().to(device)
    model.load_state_dict(torch.load(args.weights))
    print('[INFO] Model successfully loaded. Exporting to onnx...')
    model.eval()
    # trace model
    x = torch.rand(1, 3, args.img, args.img, device=device, requires_grad=True)
    y = model(x)
    # export
    torch.onnx.export(
    model,
    x,
    args.onnx_file,
    opset_version=args.opset,
    input_names=['image'],
    output_names=['mask'],
    do_constant_folding=True
    )
    print('[INFO] Model successfully exported. simplifying onnx file...')
    # simplify model
    onnx_model = onnx.load(args.onnx_file)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.onnx_file)
    print('[INFO] Final onnx file created')
    if args.checking:
        print('[INFO] Running onnxruntime inference...')
        ort_session = onnxruntime.InferenceSession(args.onnx_file)
        img = cv2.imread('test.jpg')
        img = cv2.resize(img[:, :, ::-1], (args.img, args.img))[None, ...] / 255.0
        ort_inputs = {ort_session.get_inputs()[0].name: img.astype(np.float32).transpose(0, 3, 1, 2)}
        ort_outs = ort_session.run(None, ort_inputs)
        mask = ort_outs[0]
        mask = (mask * 255).astype(np.uint8).squeeze()
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture('test_video.mp4')
        while cap.isOpened():
            ret, frame = cap.read()
            img = cv2.resize(frame[:, :, ::-1], (args.img, args.img))[None, ...] / 255.0
            ort_inputs = {ort_session.get_inputs()[0].name: img.astype(np.float32).transpose(0, 3, 1, 2)}
            ort_outs = ort_session.run(None, ort_inputs)
            mask = ort_outs[0]
            mask = (mask * 255).astype(np.uint8).squeeze()
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (args.img, args.img))
            mask = np.concatenate((frame, mask), axis=1)
            cv2.imshow('mask', mask)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()
