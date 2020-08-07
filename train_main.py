# Person Detector
from Yolo_v2_pytorch.train_yolo import get_args as get_pd_args
from Yolo_v2_pytorch.train_yolo import train as pd_train

# TRACKING(FACE RECOGNITION) - not completed
# ========================================================
# learner = face_learner(conf)
# learner.train(conf, args.epochs)
# ========================================================

# PERSON DETECTOR
# ========================================================
pd_args = get_pd_args()
pd_train(pd_args)
# ========================================================

# MAKE YOUR LOADER IN YOUR TRAIN CLASS
# ========================================================
# ========================================================
