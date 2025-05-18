# api_service.py
import traceback
import time
from datetime import datetime
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tool.parser import get_config
import os
import zipfile
import uuid
from threading import Thread
from collections import deque


class DetectionService:
    def __init__(self, cfg_path='config/configs.yaml'):
        # 加载配置
        self.cfg = get_config()
        self.cfg.merge_from_file(cfg_path)

        # 初始化模型
        self._init_model()
        self._warmup_model()

        # 初始化统计信息
        self.total_requests = 0
        self.avg_process_time = 0.0

        # 新增任务管理相关属性
        self.task_queue = deque()
        self.active_tasks = {}
        self.max_parallel = 2  # 最大并行任务数
        self.task_cleanup_interval = 3600  # 任务清理间隔（秒）

        # 启动任务处理线程
        self._start_task_processor()

    def _start_task_processor(self):
        """启动后台任务处理线程"""

        def task_processor():
            while True:
                # 处理队列中的任务
                if self.task_queue and len(self.active_tasks) < self.max_parallel:
                    task_id = self.task_queue.popleft()
                    task = self.active_tasks[task_id]
                    if task['type'] == 'video':
                        self._process_video_task(task)
                    elif task['type'] == 'batch_images':
                        self._process_batch_images_task(task)

                # 清理过期任务
                self._cleanup_old_tasks()
                time.sleep(1)

        Thread(target=task_processor, daemon=True).start()

    def _cleanup_old_tasks(self):
        """清理超过1小时的任务"""
        now = time.time()
        expired_tasks = [tid for tid, t in self.active_tasks.items()
                         if now - t['create_time'] > self.task_cleanup_interval]
        for tid in expired_tasks:
            del self.active_tasks[tid]

        # 新增视频处理接口方法

    def create_video_task(self, video_bytes, filename):
        """创建视频处理任务"""
        try:
            # 生成唯一任务ID
            task_id = str(uuid.uuid4())

            # 保存视频到临时目录
            temp_dir = os.path.join("temp", task_id)
            os.makedirs(temp_dir, exist_ok=True)

            video_path = os.path.join(temp_dir, secure_filename(filename))
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            # 创建任务记录
            self.active_tasks[task_id] = {
                'id': task_id,
                'type': 'video',
                'status': 'queued',
                'progress': 0.0,
                'create_time': time.time(),
                'video_path': video_path,
                'results': [],
                'error': None
            }

            # 加入处理队列
            self.task_queue.append(task_id)

            return {
                'status': 'success',
                'task_id': task_id,
                'message': '视频任务已创建'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'视频任务创建失败: {str(e)}'
            }

    def _process_video_task(self, task):
        """处理视频任务"""
        task['status'] = 'processing'
        cap = cv2.VideoCapture(task['video_path'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                results = self.model.predict(frame,
                                             imgsz=self.imgsz,
                                             conf=self.conf_thres,
                                             device=self.device,
                                             classes=self.classes)

                # 记录结果
                task['results'].append({
                    'frame': frame_count,
                    'detections': self._format_results(results),
                    'timestamp': datetime.now().isoformat()
                })

                # 更新进度
                frame_count += 1
                task['progress'] = frame_count / total_frames

            task['status'] = 'completed'

        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)

        finally:
            cap.release()
            # 清理临时文件
            if os.path.exists(task['video_path']):
                os.remove(task['video_path'])

        # 新增批量图片处理接口方法

    def create_batch_task(self, zip_bytes, filename):
        """创建批量图片处理任务"""
        try:
            task_id = str(uuid.uuid4())
            temp_dir = os.path.join("temp", task_id)
            os.makedirs(temp_dir, exist_ok=True)

            # 保存并解压ZIP文件
            zip_path = os.path.join(temp_dir, filename)
            with open(zip_path, 'wb') as f:
                f.write(zip_bytes)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # 获取所有图片文件
            image_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))

            if not image_files:
                raise ValueError("ZIP文件中未包含有效图片")

            # 创建任务记录
            self.active_tasks[task_id] = {
                'id': task_id,
                'type': 'batch_images',
                'status': 'queued',
                'progress': 0.0,
                'create_time': time.time(),
                'image_files': image_files,
                'processed': 0,
                'results': [],
                'error': None
            }

            self.task_queue.append(task_id)

            return {
                'status': 'success',
                'task_id': task_id,
                'message': '批量图片任务已创建',
                'total_images': len(image_files)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'批量任务创建失败: {str(e)}'
            }

    def _process_batch_images_task(self, task):
        """处理批量图片任务"""
        task['status'] = 'processing'
        total = len(task['image_files'])

        try:
            for idx, img_path in enumerate(task['image_files']):
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # 处理图片
                results = self.model.predict(img,
                                             imgsz=self.imgsz,
                                             conf=self.conf_thres,
                                             device=self.device,
                                             classes=self.classes)

                # 记录结果
                task['results'].append({
                    'filename': os.path.basename(img_path),
                    'detections': self._format_results(results),
                    'timestamp': datetime.now().isoformat()
                })

                # 更新进度
                task['processed'] = idx + 1
                task['progress'] = (idx + 1) / total

            task['status'] = 'completed'

        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)

        finally:
            # 清理临时文件
            if os.path.exists(os.path.dirname(img_path)):
                import shutil
                shutil.rmtree(os.path.dirname(img_path))

    def _init_model(self):
        """初始化目标检测模型"""
        cfg_model = self.cfg.MODEL
        self.weights = cfg_model.WEIGHT
        self.conf_thres = float(cfg_model.CONF)
        self.classes = eval(cfg_model.CLASSES)
        self.imgsz = int(cfg_model.IMGSIZE)
        self.device = cfg_model.DEVICE

        # 中文类别映射
        self.chinese_name = self.cfg.CONFIG.chinese_name

        print(f"正在加载模型: {self.weights}")
        self.model = YOLO(self.weights)
        print("模型加载完成")

    def _warmup_model(self):
        """模型预热"""
        dummy_image = np.zeros((300, 300, 3), dtype='uint8')
        self.model.predict(dummy_image,
                           imgsz=self.imgsz,
                           conf=self.conf_thres,
                           device=self.device,
                           classes=self.classes)
        print("模型预热完成")

    def process_image(self, image_bytes, filename="external_image.jpg"):
        """处理图像的核心方法"""
        try:
            # 记录开始时间
            start_time = time.time()

            # 转换图像数据
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无效的图片文件")

            # 执行预测
            results = self.model.predict(image,
                                         imgsz=self.imgsz,
                                         conf=self.conf_thres,
                                         device=self.device,
                                         classes=self.classes)

            # 格式化结果
            formatted_results = self._format_results(results)

            # 计算处理时间
            process_time = round(time.time() - start_time, 4)

            # 更新统计信息
            self._update_statistics(process_time)

            # 修改返回结构
            return {
                "status": "success",
                "detections": formatted_results,  # 原有detections数据
                "statistics": {
                    "total_objects": len(formatted_results),
                    "violence_detected": any(
                        d['class'] == 'violence' for d in formatted_results
                    )
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _format_results(self, results):
        """格式化检测结果"""
        formatted = []
        for idx, r in enumerate(results[0].boxes):
            cls_id = int(r.cls)
            cls_name = results[0].names[cls_id]

            # 转换为中文名称
            chinese_name = self.chinese_name.get(cls_name, cls_name)

            formatted.append({
                "id": idx + 1,
                "class": cls_name,
                "chinese_class": chinese_name,
                "confidence": round(float(r.conf), 4),
                "bbox": {
                    "xmin": int(r.xyxy[0][0]),
                    "ymin": int(r.xyxy[0][1]),
                    "xmax": int(r.xyxy[0][2]),
                    "ymax": int(r.xyxy[0][3])
                }
            })
        return formatted

    def _update_statistics(self, process_time):
        """更新服务统计信息"""
        self.total_requests += 1
        self.avg_process_time = round(
            (self.avg_process_time * (self.total_requests - 1) + process_time) / self.total_requests,
            4
        )


# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域请求

# 创建检测服务实例
service = DetectionService()


# 修改Flask路由部分
@app.route('/api/detect', methods=['POST'])
def detection_endpoint():
    """支持多图上传的检测接口"""
    # 获取多个图片文件
    files = request.files.getlist('images')  # 关键修改：获取文件列表

    if not files:
        return jsonify({
            "status": "error",
            "message": "未提供图片文件"
        }), 400

    results = []
    valid_count = 0

    try:
        # 遍历处理每个文件
        for file in files:
            if file.filename == '':
                continue

            try:
                # 处理单个图片
                image_bytes = file.read()

                # 调用原有处理逻辑
                result = service.process_image(
                    image_bytes,
                    filename=file.filename
                )

                # 记录成功结果
                if result['status'] == 'success':
                    valid_count += 1
                    results.append({
                        "filename": secure_filename(file.filename),
                        "status": "success",
                        "detections": result['detections'],
                        "statistics": result['statistics']
                    })
                else:
                    results.append({
                        "filename": secure_filename(file.filename),
                        "status": "error",
                        "message": result['message']
                    })

            except Exception as e:
                results.append({
                    "filename": secure_filename(file.filename),
                    "status": "error",
                    "message": f"处理失败: {str(e)}"
                })

        return jsonify({
            "status": "complete",
            "total_files": len(files),
            "processed_files": valid_count,
            "results": results
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"服务端错误: {str(e)}",
            "processed_files": valid_count,
            "results": results  # 返回已处理的结果
        }), 500

# 新增Flask路由
@app.route('/api/process/video', methods=['POST'])
def process_video():
    """视频处理接口"""
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "未提供视频文件"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "无效文件"}), 400

    try:
        result = service.create_video_task(
            video_bytes=file.read(),
            filename=file.filename
        )
        return jsonify(result), 200 if result['status'] == 'success' else 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"视频处理失败: {str(e)}"
        }), 500

@app.route('/api/tasks/<task_id>/results', methods=['GET'])
def get_task_results(task_id):
    """获取任务详细结果"""
    task = service.active_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "message": "任务不存在"}), 404

    if task['status'] != 'completed':
        return jsonify({
            "status": "error",
            "message": "任务尚未完成",
            "current_progress": task['progress']
        }), 425  # 过早的请求状态码

    # 构建详细响应
    response = {
        'task_id': task_id,
        'type': task['type'],
        'create_time': datetime.fromtimestamp(task['create_time']).isoformat(),
        'processing_time': time.time() - task['create_time'],
        'results': []
    }

    # 添加详细结果（分页示例）
    page = request.args.get('page', 1, type=int)
    per_page = 20

    if task['type'] == 'video':
        response['total_frames'] = len(task['results'])
        response['results'] = [{
            'frame': res['frame'],
            'timestamp': res['timestamp'],
            'detections': res['detections']
        } for res in task['results'][(page - 1) * per_page: page * per_page]]

    elif task['type'] == 'batch_images':
        response['total_images'] = len(task['results'])
        response['results'] = [{
            'filename': res['filename'],
            'timestamp': res['timestamp'],
            'detections': res['detections']
        } for res in task['results'][(page - 1) * per_page: page * per_page]]

    return jsonify(response)


# 新增同步视频处理接口
@app.route('/api/process/video/sync', methods=['POST'])
def process_video_sync():
    """同步视频处理接口（直接返回结果）"""
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "未提供视频文件"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "无效文件"}), 400

    try:
        # 生成唯一标识
        task_id = str(uuid.uuid4())
        create_time = time.time()

        # 创建临时目录
        temp_dir = os.path.join("temp", task_id)
        os.makedirs(temp_dir, exist_ok=True)

        # 保存视频文件
        video_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(video_path)

        # 直接处理视频
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                model_results = service.model.predict(
                    frame,
                    imgsz=service.imgsz,
                    conf=service.conf_thres,
                    device=service.device,
                    classes=service.classes
                )

                # 记录结果
                results.append({
                    'frame': frame_count,
                    'detections': service._format_results(model_results),
                    'timestamp': datetime.now().isoformat()
                })
                frame_count += 1

        finally:
            cap.release()
            # 清理临时文件
            if os.path.exists(video_path):
                os.remove(video_path)
            os.rmdir(temp_dir)

        # 构建响应（格式与异步接口一致）
        return jsonify({
            'task_id': task_id,
            'type': 'video',
            'create_time': datetime.fromtimestamp(create_time).isoformat(),
            'processing_time': round(time.time() - create_time, 2),
            'total_frames': len(results),
            'results': results
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"视频处理失败: {str(e)}"
        }), 500

if __name__ == '__main__':
    # 启动服务
    app.run(host='0.0.0.0',
            port=8000,
            threaded=True,
            debug=False)