import time
import numpy as np
import pandas as pd
import cv2



def to_csv(dataset,output_csv_path):
    # since a hand have 21 landmark, we gona generate x1,y1,z1,...,x21,y21,z21 

    columns = [[f'x{i}',f'y{i}',f'z{i}'] for i in range(1,22)]
    columns = np.array(columns).reshape(-1)
    main_df = pd.DataFrame()

    for label,landmarks in dataset.items():

        new_df  = pd.DataFrame(data=landmarks,columns=columns).assign(label=label)
        main_df = pd.concat([main_df,new_df])
    
    main_df.to_csv(output_csv_path)

    return main_df


def generate_dataset(landmarker,labels,output_csv_path,num_samples):

    dataset = {}
    cap     = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise Exception("CameraException : Failed open camera")

    for label in labels:

        print(f"Collecting Data for {label} start after (5 seconds) :")
        time.sleep(5)

        sample_count    = 0
        landmarks       = []
        wnd_name        =  "dataset Generator"

        while sample_count < num_samples:

            if sample_count % (num_samples // 2) == 0 and sample_count:

                print("Now other Hand (start after 5 seconds) : ")
                time.sleep(5)
            
            ret,frame = cap.read()

            if not ret: continue

            timestampms     = int(time.time() * 1000)
            frame           = cv2.flip(frame,1)
            rgb             = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_img          = landmarker.ToMpImage(rgb)
            result          = landmarker.getLandmarks(mp_img,timestampms)

            if result and result.hand_landmarks:

                landmark        = [[l.x,l.y,l.z] for l in result.hand_landmarks[0]]
                sample_count   += 1
                landmarks.append(landmark)
                print(f'progress : {sample_count} / {num_samples}')
            cv2.imshow(wnd_name,frame)

            
            if cv2.waitKey(20) & 0xff == ord('q') or cv2.getWindowProperty(wnd_name,cv2.WND_PROP_VISIBLE) < 1:

                raise Exception(f"DatasetGenerator : Interrupt Dataset Generation for {label}")

        dataset[label] = np.asarray(landmarks).reshape((len(landmarks), len(landmarks[0]) * len(landmarks[0][0]) ))  

    return to_csv(dataset,output_csv_path)






