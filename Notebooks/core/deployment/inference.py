import cv2
import numpy as np
import pandas as pd

from common.audit import path_exists
from common.db.stored_procedures import StoredProcedures
from core.datatype.frame_datum import FrameDatum
from database import database_interface as dbi


class InferenceModel:
    def __init__(self, model_file, labels, input_size):
        """
        This represents a model right before it gets promoted to a viNet Candidate
        @param model_file:
        @param labels:
        @param input_size:
        """
        assert model_file
        assert path_exists(model_file)
        assert labels
        assert path_exists(labels)

        self.__class_map = []
        with open(labels) as file:
            for line in file:
                line = line.strip()
                self.__class_map.append(line)
        file.close()

        self.__graph = model_file
        self.__labels_file = labels
        self.__net = None
        self._net_output = None
        self.__input_size = input_size
        self._top_one_prob = None

    @property
    def pb(self):
        return self.__graph

    @property
    def classmap(self):
        return self.__labels_file

    @property
    def input_size(self):
        return self.__input_size

    def resize(self, input_frame):
        """

        @param input_frame:
        @return:
        """
        return cv2.resize(input_frame, self.__input_size, interpolation=cv2.INTER_CUBIC)

    def predict(self, input_frame=None, scale_factor=1.0 / 127, mean=127.5, swaprb=True, crop=False):
        """
        TODO - verify forward pass with dummy input
        @param scale_factor:
        @param mean:
        @param swaprb:
        @param crop:
        @param input_frame:
        @return:
        """
        groundtruth = None

        assert isinstance(input_frame, (FrameDatum, tuple, pd.DataFrame))
        if isinstance(input_frame, tuple):
            assert len(input_frame) == 3
            frame_id = input_frame[0]
            frame_data = input_frame[2]
            groundtruth = input_frame[1]
            input_frame = FrameDatum(frame_id, frame_data, groundtruth)

        elif isinstance(input_frame, pd.DataFrame):
            frame_id = input_frame['frame_id'][0]
            frame_data = input_frame['frame_data'][0]
            groundtruth = input_frame['category'][0]

            input_frame = FrameDatum(frame_id, frame_data, groundtruth)

        input_frame = input_frame.cvmat

        try:
            net = cv2.dnn.readNetFromTensorflow(self.__graph)
            if net:
                # create blob
                blob = cv2.dnn.blobFromImage(input_frame,
                                             size=self.__input_size,
                                             scalefactor=scale_factor,
                                             swapRB=swaprb, mean=mean,
                                             crop=crop)
                # set input
                net.setInput(blob)
                out = net.forward()
                # Put efficiency information.
                t, _ = net.getPerfProfile()
                if out is not None:
                    # Reshape the Output so its a single dimensional vector
                    out = out.reshape(len(out[0][:]))
                    # Convert the scores to class probabilities between 0-1 by applying softmax
                    expanded = np.exp(out - np.max(out))
                    self._net_output = expanded / expanded.sum()
                    self._top_one_prob = round(np.max(self._net_output) * 100, 2)
                    top_one_index = np.argmax(self._net_output)
                    top_one_label = self.__class_map[top_one_index]
                    t = (t * 1000.0) / cv2.getTickFrequency()
                    if top_one_label == groundtruth:
                        print(f"GT:{groundtruth}, PRED:{top_one_label}: {self._top_one_prob}% : \u2713, time: {t}ms")
                    else:
                        print(f"GT:{groundtruth}, PRED:{top_one_label}: {self._top_one_prob}% : X, time: {t}ms")
                    return True
                else:
                    return False
            else:
                return False

        except (Exception, BaseException) as e:
            print(e)
            return False


if __name__ == '__main__':
    model_dir = r"E:\viNet_RnD\Research\Candidates\GWA"
    model = r"E:\viNet_RnD\Research\Candidates\GWA\MobileNetV2\viNet_3.0_MobileNetV2_cp.ckpt-e07_GWA.RnD.Candidate.0.pb"
    # model = r"E:\viNet_RnD\Research\Candidates\GWA\V2\viNet_2.8_Goldwind_2.3m.Operational.pb"
    label = r"E:\viNet_RnD\Research\Candidates\GWA\MobileNetV2\viNet_3.0_MobileNetV2_cp.ckpt-e07_GWA.RnD_ClassMap.txt"
    in_size = (224, 224)
    im = InferenceModel(model, label, in_size)

    server = dbi.PgsqlInterface()
    server.connect(r'C:\ProgramData\viNet\config\database.ini')

    gt = open(label).readlines()
    gt = [t.strip() for t in gt]
    frames = pd.DataFrame()

    frames = [StoredProcedures.get_frames(t, 20, connection=server.connection) for t in gt]
    frames = pd.concat(frames)

    for index, row in frames.iterrows():
        frame = (row[0], row[1], row[2])
        im.predict(input_frame=frame, mean=128, swaprb=True, scale_factor=1.0/128, crop=False)
