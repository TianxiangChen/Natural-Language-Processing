import os
import json
import numpy as np

# My imports
import string
import multiprocessing
import csv
indir = '/u/cs401/A1/data/'
feats_dir = '/u/cs401/A1/feats/'
wordlist_dir = '/u/cs401/Wordlists/'
# indir = '../data/'
# feats_dir = '../feats/'
# wordlist_dir = '../../Wordlists/'


def create_dic(filename):
    dic = {}
    with open(filename) as f:
        line = 1
        for i in f:
            temp_list = i.split('\n')
            dic[temp_list[0]] = line
            line += 1
    return dic


def load_LIWC_ids(feats_dir):
    alt_filename = feats_dir + 'Alt_IDs.txt'
    left_filename = feats_dir + 'Left_IDs.txt'
    right_filename = feats_dir + 'Right_IDs.txt'
    center_filename = feats_dir + 'Center_IDs.txt'

    alt_id_dic = create_dic(alt_filename)
    left_id_dic = create_dic(left_filename)
    right_id_dic = create_dic(right_filename)
    center_id_dic = create_dic(center_filename)

    return alt_id_dic, left_id_dic, right_id_dic, center_id_dic
alt_id_dic, left_id_dic, right_id_dic, center_id_dic = load_LIWC_ids(feats_dir)


def load_LIWC_feats(feats_dir):
    alt_filename = feats_dir + 'Alt_feats.dat.npy'
    left_filename = feats_dir + 'Left_feats.dat.npy'
    right_filename = feats_dir + 'Right_feats.dat.npy'
    center_filename = feats_dir + 'Center_feats.dat.npy'

    alt_LIWC_features = np.load(alt_filename)
    left_LIWC_features = np.load(left_filename)
    right_LIWC_features = np.load(right_filename)
    center_LIWC_features = np.load(center_filename)

    return alt_LIWC_features, left_LIWC_features, right_LIWC_features, center_LIWC_features
alt_LIWC_features, left_LIWC_features, right_LIWC_features, center_LIWC_features = load_LIWC_feats(feats_dir)


def preproc( comment ):
    ''' This function checks a single comment is empty post, "[removed]" or others

    Parameters:
        comment : string, the body of a comment

    Returns:
        a number which:
        0: empty post
        1: [removed]
        2: others
    '''

    modComm = ''
    modComm = comment

    modComm = modComm.strip()
    modComm = modComm.replace('\n', ' ')
    modComm = " ".join(modComm.split())

    if modComm == '':
        return 0
    elif modComm == '[removed]':
        return 1
    else:
        return 2


def process_single_file(file,subdir, empty_post_queue, removed_post_queque, cat_size_queue):
    fullFile = os.path.join(subdir, file)
    print("Processing " + fullFile)

    data = json.load(open(fullFile))
    data_len = len(data)
    empty_post_list = []
    removed_post_list = []
    for i in range(data_len):
        j = json.loads(data[i])
        empty_post = {}
        removed_post = {}
        result_id = preproc(j['body'])
        if result_id == 0: # empty post
            empty_post['id'] = j['id']
            empty_post['cat'] = file
            empty_post_list.append(empty_post)
        elif result_id == 1: # removed post
            removed_post['id'] = j['id']
            removed_post['cat'] = file
            removed_post_list.append(removed_post)
        print("Finish {}/{} -- {}".format(i + 1, data_len, file))

    if len(empty_post_list) > 0:
        empty_post_queue.append(empty_post_list)
    if len(removed_post_list) > 0:
        removed_post_queque.append(removed_post_list)
    dic = {}
    dic[file] = data_len
    cat_size_queue.append(dic)

    return 0


def main():
    Output_empty_post = []
    Output_removed_post = []
    cat_size = {}

    for subdir, dirs, files in os.walk(indir):
        empty_post_result = multiprocessing.Manager().list()
        removed_post_result = multiprocessing.Manager().list()
        cat_size_result = multiprocessing.Manager().list()

        jobs = [multiprocessing.Process(target = process_single_file, args =(file, subdir, empty_post_result, removed_post_result, cat_size_result )) for file in files]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        for result in empty_post_result:
            Output_empty_post = Output_empty_post + result
        for result in removed_post_result:
            Output_removed_post = Output_removed_post + result
        for result in cat_size_result:
            cat_size.update(result)
    total_data_size = sum(cat_size.values())

    # 0: Left, 1: Center, 2: Right, 3: Alt
    empty_post_distribution = [0] * 4
    removed_post_distribution = [0] * 4

    for post in Output_empty_post:
        if post['cat'] == 'Left':
            empty_post_distribution[0] += 1
        elif post['cat'] == 'Center':
            empty_post_distribution[1] += 1
        elif post['cat'] == 'Right':
            empty_post_distribution[2] += 1
        else:
            empty_post_distribution[3] += 1

    for post in Output_removed_post:
        if post['cat'] == 'Left':
            removed_post_distribution[0] += 1
        elif post['cat'] == 'Center':
            removed_post_distribution[1] += 1
        elif post['cat'] == 'Right':
            removed_post_distribution[2] += 1
        else:
            removed_post_distribution[3] += 1

    num_empty_post = len(Output_empty_post)
    num_removed_post = len(Output_removed_post)
    print("Num of empty posts: {} / {} total posts".format(num_empty_post, total_data_size))
    print("Num of removed posts: {} / {} total posts".format(num_removed_post, total_data_size))
    print("There toals size for each catagory is:")
    print("Left: {}, Center {}, Right {}, Alt {}".format(cat_size['Left'], cat_size['Center'], cat_size['Right'], cat_size['Alt']))
    print("Empty post distribution: {}".format(empty_post_distribution))
    print("Removed post distribution: {}".format(removed_post_distribution))


    feats_empty_post = np.zeros((num_empty_post, 144))
    for i, reddit in enumerate(Output_empty_post):
        cat = reddit['cat']
        if cat == 'Alt':
            feats_empty_post[i, :] = alt_LIWC_features[alt_id_dic[reddit['id']]]
        elif cat == 'Left':
            feats_empty_post[i, :] = left_LIWC_features[left_id_dic[reddit['id']]]
        elif cat == 'Right':
            feats_empty_post[i, :] = right_LIWC_features[right_id_dic[reddit['id']]]
        elif cat == 'Center':
            feats_empty_post[i, :] = center_LIWC_features[center_id_dic[reddit['id']]]
        else:
            print("for {}th datum from input file, the catagory {} is defined wrongly.".format(i, cat))

    empty_post_LIWC_feature_mean = np.mean(feats_empty_post, axis=0)
    empty_post_LIWC_feature_var = np.var(feats_empty_post, axis=0)
    print("Mean for LIWC/Receptiviti features for empty post is: {}".format(empty_post_LIWC_feature_mean))
    print("Vairance for LIWC/Receptiviti features for empty post is: {}".format(empty_post_LIWC_feature_var))

    feats_removed_post = np.zeros((num_removed_post, 144))
    for i, reddit in enumerate(Output_removed_post):
        cat = reddit['cat']
        if cat == 'Alt':
            feats_removed_post[i, :] = alt_LIWC_features[alt_id_dic[reddit['id']]]
        elif cat == 'Left':
            feats_removed_post[i, :] = left_LIWC_features[left_id_dic[reddit['id']]]
        elif cat == 'Right':
            feats_removed_post[i, :] = right_LIWC_features[right_id_dic[reddit['id']]]
        elif cat == 'Center':
            feats_removed_post[i, :] = center_LIWC_features[center_id_dic[reddit['id']]]
        else:
            print("for {}th datum from input file, the catagory {} is defined wrongly.".format(i, cat))


    removed_post_LIWC_feature_mean = np.mean(feats_removed_post, axis=0)
    removed_post_LIWC_feature_var = np.var(feats_removed_post, axis=0)
    print("Mean for LIWC/Receptiviti features for removed post is: {}".format(removed_post_LIWC_feature_mean))
    print("Variance for LIWC/Receptiviti features for removed post is: {}".format(removed_post_LIWC_feature_var))

    if os.path.isfile('a1_bonus_post_analysis.csv'):
        os.remove('a1_bonus_post_analysis.csv')

    with open('a1_bonus_post_analysis.csv', 'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(["Num of empty posts: {} / {} total posts".format(num_empty_post, total_data_size)])
        csvwriter.writerow(["Num of removed posts: {} / {} total posts".format(num_removed_post, total_data_size)])
        csvwriter.writerow(["There toals size for each catagory is:"])
        csvwriter.writerow(["Left: {}, Center {}, Right {}, Alt {}".format(cat_size['Left'], cat_size['Center'], cat_size['Right'], cat_size['Alt'])])
        csvwriter.writerow(["Empty post distribution: {}".format(empty_post_distribution)])
        csvwriter.writerow(["Removed post distribution: {}".format(removed_post_distribution)])
        csvwriter.writerow(["Mean for LIWC/Receptiviti features for empty post"])
        csvwriter.writerow(empty_post_LIWC_feature_mean)
        csvwriter.writerow(["Vairance for LIWC/Receptiviti features for empty post"])
        csvwriter.writerow(empty_post_LIWC_feature_var)
        csvwriter.writerow(["Mean for LIWC/Receptiviti features for removed post"])
        csvwriter.writerow(removed_post_LIWC_feature_mean)
        csvwriter.writerow(["Variance for LIWC/Receptiviti features for removed post is"])
        csvwriter.writerow(removed_post_LIWC_feature_var)

    return 0

if __name__ == "__main__":
    main()
