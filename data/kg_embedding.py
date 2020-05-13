import csv
import logging
import os
import argparse
import random
from sklearn.model_selection import train_test_split

logger = logging.getLogger('kg_embedding.py')


def kg_to_openke(kg_path: str, saved_dir: str, seed: int):
    '''
    @todo not implemented fully
    Given path to knowledge graph: it creates following files to train entity & relation embeddings
    :param path:
    :type path:
    :return:
    :rtype:
    '''
    english = '/c/en'
    entity_id_file = os.path.join(saved_dir, 'entity2id.txt')
    relation_id_file = os.path.join(saved_dir, 'relation2id.txt')
    train_file = os.path.join(saved_dir, 'train2id.txt')
    valid_file = os.path.join(saved_dir, 'valid2id.txt')
    test_file = os.path.join(saved_dir, 'test2id.txt')

    entities = {}
    entity_id = 0
    relations = {}
    relation_id = 0
    triples = []

    logging.info('Converting knowledge graph locating {} into OpenKE format.')
    with open(kg_path) as assertions:
        assertion_reader = csv.reader(assertions, delimiter='\t')
        for row in assertion_reader:
            relation = row[1]
            entity1 = row[2]
            entity2 = row[3]

            if entity1.startswith(english) and entity2.startswith(english):
                if entity1 not in entities:
                    entities[entity1] = entity_id
                    entity_id += 1
                if entity2 not in entities:
                    entities[entity2] = entity_id
                    entity_id += 1
                if relation not in relations:
                    relations[relation] = relation_id
                    relation_id += 1

                triples.append([entities[entity1], entities[entity2], relations[relation]])

    random.Random(seed).shuffle(triples)

    logging.info('Split triples as 60% of train, %20 of dev, %20 of test')
    train, dev = train_test_split(triples, test_size=0.40)
    dev, test = train_test_split(dev, test_size=0.50)

    logging.debug('Entity tuples are saving.')
    write_tuples(entities, entity_id_file)
    logging.debug('Relation tuples are saving.')
    write_tuples(relations, relation_id_file)
    logging.debug('Triples are saving.')
    write_triples(train_file, train)
    write_triples(valid_file, dev)
    write_triples(test_file, test)
    logging.info('Conversion is completed.')

    n_n(saved_dir, train_file, valid_file, test_file)


def write_triples(triple_file, triples):
    with open(triple_file, 'w') as output_file:
        cw = csv.writer(output_file, delimiter='\t')
        num_triples = len(triples)
        logging.info('Number of triples {}'.format(num_triples))
        cw.writerow([num_triples])
        for triple in triples:
            cw.writerow(triple)


def write_tuples(tuples, tuple_id_file):
    with open(tuple_id_file, 'w') as output_file:
        cw = csv.writer(output_file, delimiter='\t')
        num_tuples = len(tuples)
        logging.info('Number of tuples {}'.format(num_tuples))
        cw.writerow([num_tuples])
        for entity_id, tuple_name in tuples.items():
            cw.writerow([entity_id, tuple_name])


# reference script is taken from OpenKE github repository, and is adjusted
def n_n(file_dir: str, triple_fpath: str, valid_fpath: str, test_fpath: str):
    lef = {}
    rig = {}
    rellef = {}
    relrig = {}

    triple = open(triple_fpath, "r")
    valid = open(valid_fpath, "r")
    test = open(test_fpath, "r")

    tot = (int)(triple.readline())
    for i in range(tot):
        content = triple.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(valid.readline())
    for i in range(tot):
        content = valid.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    tot = (int)(test.readline())
    for i in range(tot):
        content = test.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    test.close()
    valid.close()
    triple.close()

    type_constrain_fpath = os.path.join(file_dir, "type_constrain.txt")
    f = open(type_constrain_fpath, "w")
    f.write("%d\n" % (len(rellef)))
    for i in rellef:
        f.write("%s\t%d" % (i, len(rellef[i])))
        for j in rellef[i]:
            f.write("\t%s" % (j))
        f.write("\n")
        f.write("%s\t%d" % (i, len(relrig[i])))
        for j in relrig[i]:
            f.write("\t%s" % (j))
        f.write("\n")
    f.close()

    rellef = {}
    totlef = {}
    relrig = {}
    totrig = {}
    # lef: (h, r)
    # rig: (r, t)
    for i in lef:
        if not i[1] in rellef:
            rellef[i[1]] = 0
            totlef[i[1]] = 0
        rellef[i[1]] += len(lef[i])
        totlef[i[1]] += 1.0

    for i in rig:
        if not i[0] in relrig:
            relrig[i[0]] = 0
            totrig[i[0]] = 0
        relrig[i[0]] += len(rig[i])
        totrig[i[0]] += 1.0

    s11 = 0
    s1n = 0
    sn1 = 0
    snn = 0
    f = open(test_fpath, "r")
    tot = (int)(f.readline())
    for i in range(tot):
        content = f.readline()
        h, t, r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            s11 += 1
        if (rign >= 1.5 and lefn < 1.5):
            s1n += 1
        if (rign < 1.5 and lefn >= 1.5):
            sn1 += 1
        if (rign >= 1.5 and lefn >= 1.5):
            snn += 1
    f.close()

    f = open(test_fpath, "r")

    f11 = open(os.path.join(file_dir, "1-1.txt"), "w")
    f1n = open(os.path.join(file_dir, "1-n.txt"), "w")
    fn1 = open(os.path.join(file_dir, "n-1.txt"), "w")
    fnn = open(os.path.join(file_dir, "n-n.txt"), "w")
    fall = open(os.path.join(file_dir, "test2id_all.txt"), "w")
    tot = (int)(f.readline())

    fall.write("%d\n" % (tot))
    f11.write("%d\n" % (s11))
    f1n.write("%d\n" % (s1n))
    fn1.write("%d\n" % (sn1))
    fnn.write("%d\n" % (snn))
    for i in range(tot):
        content = f.readline()
        h, t, r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            f11.write(content)
            fall.write("0" + "\t" + content)
        if (rign >= 1.5 and lefn < 1.5):
            f1n.write(content)
            fall.write("1" + "\t" + content)
        if (rign < 1.5 and lefn >= 1.5):
            fn1.write(content)
            fall.write("2" + "\t" + content)
        if (rign >= 1.5 and lefn >= 1.5):
            fnn.write(content)
            fall.write("3" + "\t" + content)
    fall.close()
    f.close()
    f11.close()
    f1n.close()
    fn1.close()
    fnn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A module for converting KGs into OpenKE format')
    parser.add_argument('--kg_path', type=str, help='Directory locating KG')
    parser.add_argument('--saved_dir', type=str, help='Directory for saving OpenKE files')
    parser.add_argument('--seed', type=int, help='Assign seed for reproducability')
    args = parser.parse_args()

    kg_to_openke(args.kg_path, args.saved_dir, args.seed)
