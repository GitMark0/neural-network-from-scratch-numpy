import h5py

from nn import NN2L


def load_dataset(file_name):
    dataset_dict = {}
    keys = []
    file_name = file_name
    f = h5py.File(file_name, 'r')
    for key in f.keys():
        keys.append(key)
        group = f[key]
        data = group.value
        dataset_dict[key] = data.copy()
    return keys, dataset_dict


if __name__ == '__main__':
    # 0:train_set; 1:test_set
    # eg. data[0][1]['train_set_x'] - returns train_x
    data = []
    for dataset_name in ('train_catvnoncat.h5', 'test_catvnoncat.h5'):
        data.append(load_dataset(dataset_name))
    train_x = data[0][1]['train_set_x'] / 255
    train_y = data[0][1]['train_set_y']

    train_x = train_x.reshape(-1, train_x.shape[0])
    features_num = train_x.shape[0]

    nn = NN2L(features_num)
    nn.train(train_x, train_y)
    print('Hello')

