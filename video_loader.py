import os
import cv2

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

class VideoManager:
    def __init__(self, default_video_path):
        self.default_video_path = default_video_path
        self.camera_map = {}
        self.build_index()

    def build_index(self):
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(backend_dir)

        target_path = self.default_video_path
        if not os.path.isabs(target_path):
            if os.path.exists(os.path.abspath(target_path)):
                target_path = os.path.abspath(target_path)
            elif os.path.exists(os.path.join(backend_dir, target_path)):
                target_path = os.path.join(backend_dir, target_path)
            else:
                target_path = os.path.join(workspace_root, target_path)

        print(f"🔍 [VideoManager] Active Video Target: {target_path} (Exists: {os.path.exists(target_path)})", flush=True)

        if os.path.exists(target_path):
            self.camera_map = {"AVENUE_FEED": [os.path.abspath(target_path)]}
        else:
            found = []
            for d in [backend_dir, workspace_root]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.lower().endswith(VIDEO_EXTENSIONS):
                            found.append(os.path.join(d, f))
            if found:
                self.camera_map = {"AVENUE_FEED": sorted(found)}
                print(f"🔍 [VideoManager] Fallback video found: {self.camera_map['AVENUE_FEED']}", flush=True)

    def get_videos(self, camera_id="AVENUE_FEED"):
        if not self.camera_map:
            self.build_index()
        return self.camera_map.get(camera_id.upper(), list(self.camera_map.values())[0] if self.camera_map else [])

class StreamManager:
    def __init__(self, video_manager):
        self.video_manager = video_manager
        self.caps = {}  # camera_id -> VideoCapture
        self.indices = {}  # camera_id -> current video index

    def get_frame(self, camera_id="AVENUE_FEED"):
        camera_id = camera_id.upper()
        videos = self.video_manager.get_videos(camera_id)
        if not videos or not os.path.exists(videos[0]):
            return None

        if camera_id not in self.caps:
            self.indices[camera_id] = 0
            self.caps[camera_id] = cv2.VideoCapture(videos[0])

        cap = self.caps[camera_id]
        ret, frame = cap.read()

        # If video reaches end -> seek back to frame 0 seamlessly for infinite loop!
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        # Secondary fallback if seek returns false
        if not ret or frame is None:
            cap.release()
            self.caps[camera_id] = cv2.VideoCapture(videos[0])
            ret, frame = self.caps[camera_id].read()

        if not ret or frame is None:
            return None

        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        return frame, videos[self.indices[camera_id]], msec

    def skip_frames(self, camera_id, num_frames):
        camera_id = camera_id.upper()
        if camera_id not in self.caps or num_frames <= 0:
            return

        videos = self.video_manager.get_videos(camera_id)
        if not videos:
            return

        cap = self.caps[camera_id]
        for _ in range(num_frames):
            ret = cap.grab()
            if not ret:
                # Seek back to frame 0 smoothly when video ends while skipping
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap.grab()
