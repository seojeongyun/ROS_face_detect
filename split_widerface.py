import os
import glob
import shutil
import json

if __name__ == '__main__':
    easy = []
    medium = []
    hard = []

    base_path = '/storage/hrlee/widerface/'
    annotaions_path = '/storage/hrlee/widerface/annotations/instances_val_easy.json'

    type = 'annotations' # images or labels or annotations
    task = 'val' # train or val or
    level = 'medium' # easy or medium or hard

    classes = glob.glob(base_path + 'images' + '/' + 'train_' + level + '/*')

    if type == 'images':
        for i in range(len(classes)):
            if level == 'easy':
                easy.append(classes[i].split('/')[6])

            elif level == 'medium':
                medium.append(classes[i].split('/')[6])

            elif level == 'hard':
                hard.append(classes[i].split('/')[6])

            else:
                NotImplementedError

        val_path = '/storage/hrlee/widerface/' + type + '/val' + '_' + level
        files = glob.glob(val_path + '/*')
        files_list = []

        for i in range(len(files)):
            files_list.append(files[i].split('/')[6])

        if level == 'easy':
            for i in range(len(easy)):
                if easy[i] in files_list:
                    files_list.remove(easy[i])

        elif level == 'medium':
            for i in range(len(medium)):
                if medium[i] in files_list:
                    files_list.remove(medium[i])

        elif level == 'hard':
            for i in range(len(hard)):
                if hard[i] in files_list:
                    files_list.remove(hard[i])

        else:
            NotImplementedError

        files_list.sort()
        removed_path_list = []

        for i in range(len(files_list)):
            removed_path = [s for s in files if files_list[i] in s]
            removed_path_list.append(removed_path[0])

        print(removed_path_list)
        for i in range(len(removed_path_list)):
            shutil.rmtree(removed_path_list[i])

    elif type == 'labels':
        for i in range(len(classes)):
            if level == 'easy':
                easy.append(classes[i].split('/')[6])

            elif level == 'medium':
                medium.append(classes[i].split('/')[6])

            elif level == 'hard':
                hard.append(classes[i].split('/')[6])

            else:
                NotImplementedError

        val_path = '/storage/hrlee/widerface/' + type + '/val' + '_' + level
        files = glob.glob(val_path + '/*')
        files_list = []

        for i in range(len(files)):
            files_list.append(files[i].split('/')[6])

        if level == 'easy':
            for i in range(len(easy)):
                if easy[i] in files_list:
                    files_list.remove(easy[i])

        elif level == 'medium':
            for i in range(len(medium)):
                if medium[i] in files_list:
                    files_list.remove(medium[i])

        elif level == 'hard':
            for i in range(len(hard)):
                if hard[i] in files_list:
                    files_list.remove(hard[i])

        else:
            NotImplementedError

        files_list.sort()
        removed_path_list = []

        for i in range(len(files_list)):
            removed_path = [s for s in files if files_list[i] in s]
            removed_path_list.append(removed_path[0])

        print(removed_path_list)
        for i in range(len(removed_path_list)):
            shutil.rmtree(removed_path_list[i])

    elif type == 'annotations':
        annotaions_path = '/storage/hrlee/widerface/annotations/instances_val.json'

        with open(annotaions_path) as file:
            datas = json.load(file)
            # json_test = datas['users']
            print(datas)

