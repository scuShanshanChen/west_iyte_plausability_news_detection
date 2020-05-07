import csv
import logging
import os
import argparse

logger = logging.getLogger('kg_embedding.py')


def kg_to_openke(kg_path: str, saved_dir: str):
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
    triple_file = os.path.join(saved_dir, 'triples.txt')

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

    logging.debug('Entity tuples are saving.')
    write_tuples(entities, entity_id_file)
    logging.debug('Relation tuples are saving.')
    write_tuples(relations, relation_id_file)
    logging.debug('Triples are saving.')
    write_triples(triple_file, triples)
    logging.info('Conversion is completed.')


def write_triples(triple_file, triples):
    with open(triple_file, 'w') as output_file:
        cw = csv.writer(output_file, delimiter='\t')
        num_triples = len(triples)
        logging.info('Number of triples {}'.format(num_triples))
        cw.write_row([num_triples])
        for triple in triples:
            cw.write_row(triple)


def write_tuples(tuples, tuple_id_file):
    with open(tuple_id_file, 'w') as output_file:
        cw = csv.writer(output_file, delimiter='\t')
        num_tuples = len(tuples)
        logging.info('Number of triples {}'.format(num_tuples))
        cw.write_row([num_tuples])
        for entity_id, tuple_name in tuples.items():
            cw.write_row([tuple_name, entity_id])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A module for converting KGs into OpenKE format')
    parser.add_argument('--kg_path', type=str, help='Directory locating KG')
    parser.add_argument('--saved_dir', type=str, help='Directory for saving OpenKE files')
    args = parser.parse_args()

    kg_to_openke(args.kg_path, args.saved_dir)
