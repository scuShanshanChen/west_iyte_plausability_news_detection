import csv


def kg_to_openke(kg_path: str, triple_path):
    '''
    @todo not implemented fully
    Given path to knowledge graph: it creates following files to train entity & relation embeddings
    :param path:
    :type path:
    :return:
    :rtype:
    '''
    english = '/c/en'

    entities = {}
    entity_id = 0
    relations = {}
    relation_id = 0
    triples = []

    with open(kg_path) as assertions:
        assertion_reader = csv.reader(assertions, delimiter='\t')
        for row in assertion_reader:
            relation = row[1]
            entity1 = row[2]
            entity2 = row[3]

            if entity1.startswith(english) and entity2.startswith(english):
                if not entity1 in entities:
                    entity_id += 1
                    entities[entity1] = entity_id
                if not entity2 in entities:
                    entity_id += 1
                    entities[entity2] = entity_id
                if not relation in relations:
                    relation_id += 1
                    relations[relation] = relation_id

                triples.append([entities[entity1], relations[relation], entities[entity2]])

    with open(triple_path, 'w') as output_file:
        cw = csv.writer(output_file, delimiter='\t')
        cw.write_row([len(triples)])
        for triple in triples:
            cw.write_row(triple)

    # @todo add entity2id.txt and relation2id.txt
