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
    dev_file = os.path.join(saved_dir, 'dev2id.txt')
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
                    entity_id += 1
                    entities[entity1] = entity_id
                if entity2 not in entities:
                    entity_id += 1
                    entities[entity2] = entity_id
                if relation not in relations:
                    relation_id += 1
                    relations[relation] = relation_id

                triples.append([entities[entity1], relations[relation], entities[entity2]])

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
    write_triples(dev_file, dev)
    write_triples(test_file, test)
    logging.info('Conversion is completed.')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A module for converting KGs into OpenKE format')
    parser.add_argument('--kg_path', type=str, help='Directory locating KG')
    parser.add_argument('--saved_dir', type=str, help='Directory for saving OpenKE files')
    parser.add_argument('--seed', type=int, help='Assign seed for reproducability')
    args = parser.parse_args()

    kg_to_openke(args.kg_path, args.saved_dir, args.seed)
