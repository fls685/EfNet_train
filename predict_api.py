"""
预测API：接收图片URL，返回分类结果
"""
import torch
import timm
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import requests
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# 全局变量
model = None
device = None
classes = ['back', 'detail', 'front', 'other']
transform = None


def load_model(checkpoint_path='checkpoints/best.pth'):
    """加载训练好的模型"""
    global model, device, classes, transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 创建模型
    model = timm.create_model(
        checkpoint['model_name'],
        pretrained=False,
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 类别
    classes = checkpoint['classes']

    # 数据预处理
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

    print(f"模型加载完成！")
    print(f"设备: {device}")
    print(f"类别: {classes}")
    print(f"验证准确率: {checkpoint['val_acc']:.4f}")


def download_image(url, timeout=10):
    """从URL下载图片"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # 转换为numpy数组
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    except Exception as e:
        raise Exception(f"下载图片失败: {str(e)}")


def predict_image(image):
    """预测图片类别"""
    global model, device, transform, classes

    # 预处理
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()

    # 所有类别的概率
    all_probs = {
        classes[i]: float(probabilities[0][i].item())
        for i in range(len(classes))
    }

    return predicted_class, confidence_score, all_probs


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': classes
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    预测接口

    请求体:
    {
        "url": "https://example.com/image.jpg"
    }

    返回:
    {
        "success": true,
        "predicted_class": "front",
        "confidence": 0.9523,
        "probabilities": {
            "back": 0.0123,
            "detail": 0.0234,
            "front": 0.9523,
            "other": 0.0120
        }
    }
    """
    try:
        # 获取请求参数
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': '请提供图片URL (url字段)'
            }), 400

        url = data['url']

        # 检查模型是否加载
        if model is None:
            return jsonify({
                'success': False,
                'error': '模型未加载'
            }), 500

        # 下载图片
        image = download_image(url)

        # 预测
        predicted_class, confidence, probabilities = predict_image(image)

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量预测接口

    请求体:
    {
        "urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
    }
    """
    try:
        data = request.get_json()

        if not data or 'urls' not in data:
            return jsonify({
                'success': False,
                'error': '请提供图片URL列表 (urls字段)'
            }), 400

        urls = data['urls']

        if not isinstance(urls, list):
            return jsonify({
                'success': False,
                'error': 'urls必须是列表'
            }), 400

        if model is None:
            return jsonify({
                'success': False,
                'error': '模型未加载'
            }), 500

        # 批量预测
        results = []
        for url in urls:
            try:
                image = download_image(url)
                predicted_class, confidence, probabilities = predict_image(image)
                results.append({
                    'url': url,
                    'success': True,
                    'predicted_class': predicted_class,
                    'confidence': float(confidence),
                    'probabilities': probabilities
                })
            except Exception as e:
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # 加载模型
    checkpoint_path = Path('checkpoints/best.pth')

    if not checkpoint_path.exists():
        print(f"错误: 模型文件不存在: {checkpoint_path}")
        print("请先训练模型或指定正确的checkpoint路径")
        exit(1)

    load_model(checkpoint_path)

    # 启动Flask服务
    print("\n" + "="*60)
    print("预测API已启动！")
    print("="*60)
    print("\n使用示例:")
    print("curl -X POST http://localhost:5000/predict \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"url\": \"https://example.com/image.jpg\"}'")
    print("\n" + "="*60 + "\n")

    app.run(host='0.0.0.0', port=5555, debug=False)
