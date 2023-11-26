import cv2
import os

class get_frame_from_video:
    def __init__(self):
        self.name = 'ParkSeongJun'
        #
        self.save = False
        self.show = True
        self.glasses = True
        self.video_capture = True
        self.rotate = True
        #
        self.save_path, self.video_path = self.get_path()
        self.count = 1

    def get_path(self):
        save_path = '/storage/jysuh/gallery' + self.name
        if self.glasses:
            path = '/storage/jysuh/Video/wear_glass.mp4'
        else:
            path = '/storage/jysuh/Video/not_wear_glass.mp4'

        return save_path, path

    def make_gallery(self):
        if not os.path.exists(self.save_path):
            os.makedirs("/".join(self.save_path.split('/')[:-1]), exist_ok=True)

    def capture(self):
        capture = cv2.VideoCapture(self.video_path)
        #
        while self.video_capture:
            success, self.vid_img = capture.read()
            self.vid_img = [cv2.resize(self.vid_img, dsize=(640,640))]
            fps = capture.get(cv2.CAP_PROP_FPS)
            #
            if self.rotate:
                self.vid_img[0] = cv2.rotate(self.vid_img[0], cv2.ROTATE_90_CLOCKWISE)
            #
            if success:
                if self.save:
                    self.img_save()
                #
                if self.show:
                    self.img_show()
                    print(fps)
                #
                self.count += 1
            else:
                cv2.destroyAllWindows()
                break

    def img_save(self):
        cv2.imwrite(self.save_path.split('/')[:-1] + '_{}'.format(self.count), self.vid_img[0])

    def img_show(self):
        cv2.imshow('frame', self.vid_img[0])
        cv2.waitKey(10)



if __name__ == '__main__':
    img = get_frame_from_video()
    img.make_gallery()
    img.capture()
