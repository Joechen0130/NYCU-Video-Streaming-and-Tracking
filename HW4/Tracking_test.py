import torch
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.optimize import linear_sum_assignment

#Reference: https://zhuanlan.zhihu.com/p/628015159

def linear_assignment(cost_matrix):
  """try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0])
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))"""
  
  x, y = linear_sum_assignment(cost_matrix)
  return np.array(list(zip(x, y)))
  
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return [], np.arange(len(detections)), []

  if(len(detections)==0):
    return [], [], []

  iou_matrix = iou_batch(detections, trackers)

  #print("IOU",iou_matrix)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      #print(np.stack(np.where(a), axis=1))
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      #print(linear_assignment(-iou_matrix))
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = []

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def iou_batch(bb_test, bb_gt):
  """
  compute iou
  """
  #print(bb_test)
  #print(bb_gt)
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  

def convert_bbox_to_z(bbox):
  """
  input: [left_top_x, left_top_y, right_bottom_x, right_bottm_y]
  return: [centre_x, centre_y, area, ratio]
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  centre_x = bbox[0] + w/2.
  centre_y = bbox[1] + h/2.
  area = w * h    #scale is just area
  ratio = w / float(h)
  return np.array([centre_x, centre_y, area, ratio]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
  """
  input: [centre_x, centre_y, area, ratio]
  return: [left_top_x, left_top_y, right_bottom_x, right_bottm_y]
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.  #measurement uncertainty/noise
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01  #Process uncertainty/noise
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0  # the continue time object tracked
        self.age = 0
        self.is_show = True

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0 #一旦更新表示Tracker已被追蹤
        self.history = []
        self.hits += 1  #物體被追蹤的次數（幀數）+1
        self.hit_streak += 1  #物體被連續追蹤的次數（幀數）+1
        self.kf.update(convert_bbox_to_z(bbox)) #更新狀態

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):  #防止邊界溢出
            self.kf.x[6] *= 0.0
        self.kf.predict()  #預測
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5)) #(x1,y1,x2,y2,id)
        to_del = []
        ret = []
        
        #預測結果超出範圍的Trackers直接刪除
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)

        #unmatched trackers with assigned detections
        """TODO:"""
        """for m in unmatched_trks:
            self.trackers[m[1]].update(dets[m[0], :])"""

        show_list = []

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            #print('ss', trk)
            #當前幀更新的Trackers同時 （被連續追蹤超過次數）的保留
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                show_list.append(trk.is_show)
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret), show_list
        return np.empty((0,5)), []


if __name__ == "__main__":
    # Model
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    #https://github.com/ultralytics/yolov5#pretrained-checkpoints
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
    model.conf = 0.6  # NMS confidence threshold
    model.iou = 0.3  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    #model.max_det = 1000  # maximum number of detections per image
    model.amp = False

    video_path = r"./demo_test.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output.mp4', fourcc, fps, (1280,720))

    tracker=Sort(max_age=30,min_hits=10,iou_threshold=0.1)

    totalCount=set()
    color_dict = {}

    while (True):
        detections=np.empty((0,6))

        ret, frame = cap.read()
        if ret == False: break
        # Inference
        results = model(frame[..., ::-1])
        # Results
        #xmin  ymin  xmax  ymax  confidence  class  name
        print(results.pandas().xyxy[0])

        for index, row in results.pandas().xyxy[0].iterrows():
            xmin,ymin,xmax,ymax,confidence,class_name = int(row["xmin"]),int(row["ymin"]),int(row["xmax"]),int(row["ymax"]),row["confidence"],row["name"]
            currentArray=np.array([xmin,ymin,xmax,ymax,confidence,0])
            detections=np.vstack((detections,currentArray))
        
        resultTracker, show_list = tracker.update(detections)
        for result in resultTracker:
            x1, y1, x2, y2 ,id_ = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            if id_ in color_dict:
                color = color_dict[int(id_)]
            else:
                color_dict[int(id_)] = list(np.random.random(size=3) * 256)
                color = color_dict[int(id_)]

            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 6)
            labelSize = cv2.getTextSize('ID-{}'.format(id_), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1),(x1+labelSize[0][0],y1-labelSize[0][1]), color, cv2.FILLED)
            cv2.putText(frame, 'ID-{}'.format(id_), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            cx,cy=x1+w//2,y1+h//2
            totalCount.add(int(id_))
            if len(totalCount) == 0:
              cv2.putText(frame,"0",(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),5)
            else:
              cv2.putText(frame,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),5)

        out.write(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        # ESC
        if key == 27:
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    print("*"*30)
    print(f"Count:{len(totalCount)}")
    print("*"*30)