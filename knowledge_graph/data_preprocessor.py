import datetime
import glob
import json
import re
import os

remove_ext = '.mp4'


def shot_time_main_dir(dirname, ext):
    files = glob.glob(dirname + '*.' + ext)
    output = []
    for file in files:
        temp = shot_time_main(file)
        output.append(temp)
    return output


def shot_time_main(filename):
    input_file = filename
    with open(input_file, 'r') as f:
        d = json.load(f)
    f_name = d['file_name'].replace(remove_ext, '')
    # print(f_name)

    shot_dic = {}
    shots = d['shot_results']
    for shot in shots:
        shot_id = int(shot['shot_id'].replace('SHOT_', ''))
        st = shot['start_time']
        et = shot['end_time']
        if shot_id in shot_dic:
            print(shot_id, 'shot id error')
        else:
            shot_dic[shot_id] = {'st': st, 'et': et}
    output = {"ep": f_name, 'shots': shot_dic}
    return output


def ess_main(filename):
    input_file = filename
    ess_dic = {}
    with open(input_file, 'r') as f:
        d = json.load(f)
    for item in d['data']:
        ess_dic[item['ep_id']] = item['scenes']
    time_data = shot_time_main_dir('../AnotherMissOh_ShotList2/', 'json')
    time_dic = {}
    for item in time_data:
        if item['ep'] in time_dic:
            # print(item['ep'])
            pass
        else:
            time_dic[item['ep']] = item['shots']
    output = {}
    for k, v in ess_dic.items():
        ep_item = time_dic[k]
        temp_ep = []
        for sce in v:
            shots = sce['shots']
            temp_shot = []
            for shot in shots:
                time = ep_item[int(shot)]
                temp = {shot: time}
                temp_shot.append(temp)
            sce_id = sce['scene_id']
            temp_scen = {sce_id: temp_shot}
            temp_ep.append(temp_scen)
        output[k] = temp_ep

    with open(filename.replace('ESS', 'ESST'), 'w', encoding='UTF-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def add_scene_num_subtitle(filename):
    input_file = filename
    en_sub = load_sub('data/input/edited_AnotherMissOh_Subtitle/')
    sub = [en_sub]
    sub_lang = ['eng']
    with open(input_file, 'r') as f:
        d = json.load(f)
    output = []
    for i, item in enumerate(sub):
        output_item = {}
        for ep, scenes in d.items():
            print('working ' + sub_lang[i] + '-' + ep + '...')
            scripts = item[ep]['script']
            for script in scripts:
                st = datetime.datetime.strptime(script['st'], '%H:%M:%S.%f')
                et = datetime.datetime.strptime(script['et'], '%H:%M:%S.%f')
                # st = datetime.datetime.strptime(script['st'].split('.')[0], '%H:%M:%S')
                # et = datetime.datetime.strptime(script['et'].split('.')[0], '%H:%M:%S')
                # st = st + datetime.timedelta(seconds=2)
                # et = et + datetime.timedelta(seconds=2)

                scene_st, scene_et = '', ''
                scene_num = ''
                # scene: dict
                for scene in scenes:
                    # print(scene)
                    for key in scene.keys():
                        # print(key)
                        scene_num = key
                        shots = scene[key]
                        shot_first = shots[0]
                        shot_last = shots[-1]
                        for k in shot_first.keys():
                            temp = shot_first[k]
                            scene_st = datetime.datetime.strptime(temp['st'], '%H:%M:%S;%f')
                            # scene_st = datetime.datetime.strptime(temp['st'].split(';')[0], '%H:%M:%S')
                            # scene_st = scene_st - datetime.timedelta(seconds=2)
                        for k in shot_last.keys():
                            temp = shot_last[k]
                            scene_et = datetime.datetime.strptime(temp['et'], '%H:%M:%S;%f')
                            # scene_et = datetime.datetime.strptime(temp['et'].split(';')[0], '%H:%M:%S')
                            # scene_et = scene_et - datetime.timedelta(seconds=2)
                    if scene_st <= st and et <= scene_et:
                        # print(script['utter'])
                        # print(ep, scene_num)
                        # print(scene_st, scene_et)
                        if 'scene_num' in script:
                            print(script['scene_num'], scene_num)
                        else:
                            script['scene_num'] = scene_num
                    else:
                        pass

            isSortedScript(ep, scripts)
            scripts = sortedSceneID(scripts)
            output_item[ep] = {'script': scripts}
        output.append(output_item)
    return output


def sortedSceneID(scripts):
    scene_id = 1
    scene_num_list = []
    s_num = '-10'
    for script in scripts:
        if 'scene_num' in script:
            s_num = script['scene_num']
        else:
            script['scene_num'] = s_num
            pass
        scene_num_list.append(int(s_num))
    # print(scene_num_list)
    scene_num_dic = {}
    for script in scripts:
        if 'scene_num' in script:
            s_num = script['scene_num']
            if s_num in scene_num_dic:
                temp = scene_num_dic[s_num]
                temp.append(script)
                scene_num_dic[s_num] = temp
            else:
                scene_num_dic[s_num] = [script]
        else:
            # no scene number (error)
            print(script)

    # return scripts
    return scene_num_dic


def isSortedScript(ep, scripts):
    scene_num_list = []
    for script in scripts:
        s_num = -10
        try:
            s_num = script['scene_num']
        except KeyError as e:
            # s_num = -1
            pass
        scene_num_list.append(int(s_num))
    while -10 in scene_num_list:
        scene_num_list.remove(-10)
    is_sorted = all(x <= y for x, y in zip(scene_num_list[:-1], scene_num_list[1:]))

    print_list = []
    prev = -1
    for num in scene_num_list:
        if prev != num:
            print_list.append(num)
            prev = num
    if not is_sorted:
        pass

def add_time_subtitle():
    sub_list = add_scene_num_subtitle('data/input/AnotherMissOh_ESST-list.json')
    en_sub = sub_list[0]
    dir_name = 'data/input/AnotherMissOh_Scene_Subtitle/'
    try:
        os.mkdir(dir_name)
    except:
        pass

    for i, sub in enumerate(sub_list):
        for ep, item in sub.items():
            outputfile = dir_name + ep[:-2] + '_ep' + ep[-2:] + '.json'
            result = []
            for s_id, us in item['script'].items():
                scene = {
                    "scene": us,
                    "scene_number": s_id
                }
                result.append(scene)
            with open(outputfile, 'w', encoding='UTF-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)


def load_sub(dirname):
    files = glob.glob(dirname + '*.json')
    prefix = 'AnotherMissOh'
    output = {}
    for file in files:
        with open(file, 'r', encoding='UTF-8') as f:
            d = json.load(f)
        file = file.replace('_subtitle', '')
        ep_num = file[-7:-5]
        ep_id = prefix + ep_num
        # print(d)
        output[ep_id] = d
    return output


if __name__ == "__main__":
    add_time_subtitle()
