import csv
import os

import numpy as np


if __name__ == "__main__":
    data_dir = 'Data/chebi-20'
    descriptions = {}
    mol2vecs = {}

    training_cids = []
    with open(os.path.join(data_dir, "training.txt")) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames=[
                                'cid', 'mol2vec', 'desc'])
        for n, line in enumerate(reader):
            descriptions[line['cid']] = line['desc']
            mol2vecs[line['cid']] = line['mol2vec']
            training_cids.append(line['cid'])

    valid_cids = []
    with open(os.path.join(data_dir, "val.txt")) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames=[
                                'cid', 'mol2vec', 'desc'])
        for n, line in enumerate(reader):
            descriptions[line['cid']] = line['desc']
            mol2vecs[line['cid']] = line['mol2vec']
            valid_cids.append(line['cid'])

    test_cids = []
    with open(os.path.join(data_dir, "test.txt")) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames=[
                                'cid', 'mol2vec', 'desc'])
        for n, line in enumerate(reader):
            descriptions[line['cid']] = line['desc']
            mol2vecs[line['cid']] = line['mol2vec']
            test_cids.append(line['cid'])

    def mol_to_vec(cids, mol2vecs):
        transformed = []
        for cid in cids:
            mol_embed = np.fromstring(mol2vecs[cid], sep=" ")
            transformed.append({'cid': cid, 'mol2vec': mol_embed})
        return transformed

    Y_train = mol_to_vec(training_cids, mol2vecs)
    Y_valid = mol_to_vec(valid_cids, mol2vecs)
    Y_test = mol_to_vec(test_cids, mol2vecs)

    Y_cand = Y_train + Y_valid + Y_test

    X_train = [descriptions[item['cid']] for item in Y_train]
    X_valid = [descriptions[item['cid']] for item in Y_valid]
    X_test = [descriptions[item['cid']] for item in Y_test]

    save_dir = data_dir
    np.save(os.path.join(save_dir, f'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, f'y_train.npy'), Y_train)

    np.save(os.path.join(save_dir, f'X_valid.npy'), X_valid)
    np.save(os.path.join(save_dir, f'y_valid.npy'), Y_valid)

    np.save(os.path.join(save_dir, f'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, f'y_test.npy'), Y_test)

    np.save(os.path.join(save_dir, f'y_cand.npy'), Y_cand)
