import glob

if __name__ == '__main__':
    number_of_images = {}
    path = []

    origin_train_path = '/storage/sjpark/VGGFace2/train/*/*.jpg'
    origin_test_path = '/storage/sjpark/VGGFace2/test/*/*.jpg'

    CBAM_train_path = '/storage/sjpark/VGGFace2/Extracted_Face/CBAM_960/Train/*/*.jpg'
    CBAM_test_path = '/storage/sjpark/VGGFace2/Extracted_Face/CBAM_960/Test/*/*.jpg'

    CW_train_path = '/storage/sjpark/VGGFace2/Extracted_Face/CW_960/Train/*/*.jpg'
    CW_test_path = '/storage/sjpark/VGGFace2/Extracted_Face/CW_960/Test/*/*.jpg'

    SW_train_path = '/storage/sjpark/VGGFace2/Extracted_Face/SW_960/Train/*/*.jpg'
    SW_test_path = '/storage/sjpark/VGGFace2/Extracted_Face/SW_960/Test/*/*.jpg'

    path = [origin_train_path, origin_test_path, CBAM_train_path, CBAM_test_path, CW_train_path, CW_test_path, SW_train_path, SW_test_path]

    for path_ in path:
        images = glob.glob(path_)

        if path_.split('/')[4] != 'Extracted_Face':
            name = 'origin'
            if path_.split('/')[4] == 'train':
                task = 'Train'
            else:
                task = 'Test'
        else:
            name = path_.split('/')[5]
            if path_.split('/')[6] == 'Train':
                task = 'Train'
            else:
                task = 'Test'

        key = name + '_' + task
        number_of_images[key] = len(images)

    for key, item in number_of_images.items():
        if key.split('_')[0] != 'origin':
            if key.split('_')[2] == 'Train':
                number_of_images[key] = item / number_of_images['origin_Train'] * 100
                number_of_images[key] = round(number_of_images[key], 3)
            else:
                number_of_images[key] = item / number_of_images['origin_Test'] * 100
                number_of_images[key] = round(number_of_images[key], 3)

    print(number_of_images)

